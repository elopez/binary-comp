"""Conservative i386 COFF extension checks, without rewriting either object.

Only explicitly named, self-contained public COMDAT functions may be added.
Existing contributions retain their order, raw bytes and link identities.
"""
from collections import Counter
import hashlib
import struct

from .coff import COFF, is_function_type, u16, u32


COMDAT = 0x1000
_RELOCATION_WIDTHS = {0: 0, 1: 2, 2: 2, 6: 4, 7: 4, 10: 2, 11: 4, 13: 1, 20: 4}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _View:
    def __init__(self, coff, sections, added, allow_source_path_change, allow_nonselecting_checksum_change):
        self.coff, self.sections, self.added = coff, sections, added
        self.allow_source_path_change = allow_source_path_change
        self.allow_nonselecting_checksum_change = allow_nonselecting_checksum_change
        self.file_paths = {}
        self.section_definitions = {}
        for index, symbol in coff.symbols.items():
            section = symbol["section"]
            _require(-2 <= section <= len(coff.sections), "Invalid symbol section")
            _require(symbol["storage"] != 103 or symbol["aux_count"] > 0,
                     "File symbol requires filename auxiliaries")
            if section > 0:
                _require(symbol["value"] <= len(coff.sections[section - 1]["data"]),
                         "Symbol outside its section")
            if (section > 0 and symbol["storage"] == 3 and symbol["type"] == 0
                    and symbol["value"] == 0 and symbol["aux_count"] == 1
                    and symbol["name"] == coff.sections[section - 1]["name"]):
                _require(section not in self.section_definitions,
                         "Multiple section-definition records")
                self.section_definitions[section] = index
        for section, entry in enumerate(coff.sections, 1):
            header = 20 + (section - 1) * 40
            _require(not u32(coff.data, header + 28) and not u16(coff.data, header + 34),
                     "COFF line tables are unsupported")
            _require(not entry["flags"] & 0x01000000, "Relocation overflow is unsupported")
            for offset, index, kind in entry["relocations"]:
                _require(kind in _RELOCATION_WIDTHS, "Unsupported i386 relocation kind")
                _require(offset + _RELOCATION_WIDTHS[kind] <= len(entry["data"]),
                         "Relocation outside its section")
                _require(index in coff.symbols, "Relocation targets a missing or auxiliary symbol")
            if entry["flags"] & COMDAT:
                members = [i for i, s in coff.symbols.items() if s["section"] == section]
                _require(section in self.section_definitions
                         and members[0] == self.section_definitions[section],
                         "COMDAT requires its leading section-definition record")
                aux = coff.auxiliary_records[members[0]]
                _require(len(members) >= 2 or aux[14] == 5,
                         "Non-associative COMDAT requires a leader symbol")
        # Check every auxiliary record, including unreferenced metadata.
        for index in coff.symbols:
            self.key(index)
        for index, symbol in coff.symbols.items():
            if symbol["storage"] == 105:
                seen = set()
                current = index
                while coff.symbols[current]["storage"] == 105:
                    _require(current not in seen, "Cyclic weak fallback")
                    seen.add(current)
                    current = coff.symbols[current]["weak_fallback"]
            section = symbol["section"]
            if self.section_definitions.get(section) == index:
                seen = set()
                while True:
                    aux = coff.auxiliary_records[self.section_definitions[section]]
                    if aux[14] != 5:
                        break
                    _require(section not in seen, "Cyclic COMDAT association")
                    seen.add(section)
                    section = u16(aux, 12)

    def is_added_reference(self, symbol):
        return (symbol["name"] in self.added and symbol["storage"] == 2
                and symbol["value"] == 0 and symbol["aux_count"] == 0
                and (symbol["section"] == 0 and (symbol["type"] == 0
                     or is_function_type(symbol["type"]))
                     or symbol["section"] > 0 and is_function_type(symbol["type"])))

    def primary(self, index, *, reference=False):
        _require(index in self.coff.symbols, "Auxiliary reference targets a missing symbol")
        symbol = self.coff.symbols[index]
        if reference and self.is_added_reference(symbol):
            return ("added-function", symbol["name"])
        section = symbol["section"]
        private = (section > 0 and symbol["storage"] in (3, 6)
                   and symbol["type"] == 0 and symbol["aux_count"] == 0)
        count = None if self.allow_source_path_change and symbol["storage"] == 103 else symbol["aux_count"]
        return (None if private else symbol["name"], self.sections[section] if section > 0
                else section, symbol["value"], symbol["type"], symbol["storage"],
                count)

    def reference(self, index):
        # Index-bearing auxiliary fields may not disambiguate duplicate records
        # by an arbitrary table position. Reject that uncommon shape.
        key = self.primary(index, reference=True)
        _require(sum(self.primary(i, reference=True) == key for i in self.coff.symbols) == 1,
                 "Ambiguous auxiliary symbol reference")
        return key

    def key(self, index, *, relocation=False):
        symbol = self.coff.symbols[index]
        if relocation and self.is_added_reference(symbol):
            return ("added-function", symbol["name"])
        aux = self.coff.auxiliary_records.get(index, b"")
        section = symbol["section"]
        if self.section_definitions.get(section) == index:
            entry = self.coff.sections[section - 1]
            length, relocations, lines, checksum, association, selection = struct.unpack_from(
                "<IHHIHB", aux)
            _require(length == len(entry["data"]) and relocations == len(entry["relocations"])
                     and lines == 0 and aux[15:] == bytes(3),
                     "Invalid section-definition auxiliary record")
            _require(selection in range(1, 7) if entry["flags"] & COMDAT else selection == 0,
                     "Unsupported COMDAT selection")
            if selection == 5:
                _require(association in self.section_definitions
                         and self.coff.sections[association - 1]["flags"] & COMDAT,
                         "Invalid COMDAT association")
                association = self.sections[association]
            # Number has no index semantics for non-associative selections;
            # retain it exactly instead of guessing that it names a section.
            if self.allow_nonselecting_checksum_change and selection in (0, 5):
                checksum = None
            aux_key = ("section", length, relocations, lines, checksum, association, selection)
        elif symbol["storage"] == 105:
            _require(section == 0 and symbol["value"] == 0 and len(aux) == 18
                     and aux[8:] == bytes(10), "Unsupported weak-external metadata")
            aux_key = ("weak", self.reference(symbol["weak_fallback"]), symbol["weak_search"])
        elif aux and symbol["storage"] == 103:
            _require(symbol["name"] == ".file" and section == -2
                     and symbol["type"] == 0 and symbol["value"] == 0,
                     "Unsupported file-symbol metadata")
            if self.allow_source_path_change:
                end = aux.find(b"\0")
                _require(end > 0 and aux[end:] == bytes(len(aux) - end),
                         "Source filename must be NUL-terminated and zero-padded")
                path = aux[:end].decode("ascii")
                basename = path.replace("\\", "/").rsplit("/", 1)[-1]
                _require(basename not in ("", ".", ".."), "Invalid source basename")
                aux_key = ("file", basename)
                self.file_paths[index] = path
            else:
                aux_key = ("file", aux)
        elif aux and symbol["storage"] == 2 and section > 0 and is_function_type(symbol["type"]):
            _require(len(aux) == 18, "Unsupported function auxiliaries")
            tag, size, lines, following = struct.unpack_from("<IIII", aux)
            _require(not lines and aux[16:] == bytes(2)
                     and symbol["value"] + size <= len(self.coff.sections[section - 1]["data"]),
                     "Unsupported function-definition metadata")
            aux_key = ("function", self.reference(tag) if tag else None, size,
                       self.reference(following) if following else None)
        else:
            _require(not aux, "Unsupported auxiliary record")
            aux_key = None
        return self.primary(index), aux_key


def compare_coff_extension(baseline: COFF, candidate: COFF, added_functions=(), *,
                           allow_source_path_change=False, allow_nonselecting_checksum_change=False):
    """Return a JSON-compatible witness, or raise ValueError on any difference.

    Inputs are parsed native i386 objects. Symbol/section indices and private
    untyped local spellings may change; no content matching or rewriting occurs.
    Unsupported auxiliary/debug shapes fail closed. Physical file offsets,
    timestamps and unused string-table encodings are outside this semantic check.
    The source-path opt-in permits only genuine .file paths with the same
    basename and reports every changed path; runtime identities remain strict.
    A separate opt-in permits reported section-checksum changes for ordinary
    and associative contributions after their payload/link checks pass.
    """
    names = list(added_functions)
    _require(all(isinstance(name, str) and name for name in names)
             and len(names) == len(set(names)), "Added function names must be distinct nonempty strings")
    names = set(names)
    _require(u16(baseline.data, 18) == u16(candidate.data, 18), "COFF header flags differ")
    added = {}
    excluded_symbols = set()
    for name in sorted(names):
        _require(not any(s["name"] == name and (s["section"] != 0 or s["value"] != 0
                     or s["storage"] != 2 or s["aux_count"] != 0
                     or not (s["type"] == 0 or is_function_type(s["type"])))
                     for s in baseline.symbols.values()), "Added function already has baseline ownership")
        definitions = [(i, s) for i, s in candidate.symbols.items()
                       if s["name"] == name and s["section"] > 0]
        _require(len(definitions) == 1, f"Expected one added function definition: {name}")
        index, symbol = definitions[0]
        section = symbol["section"]
        _require(1 <= section <= len(candidate.sections), "Invalid added function section")
        entry = candidate.sections[section - 1]
        _require(symbol["value"] == 0 and symbol["storage"] == 2
                 and is_function_type(symbol["type"]) and symbol["aux_count"] == 0,
                 "Added function must be a sole typed public definition at offset zero without auxiliaries")
        _require(entry["name"] == ".text" and entry["data"] and not entry["relocations"]
                 and (entry["flags"] & ~0x00F00000) == 0x60001020
                 and (entry["flags"] >> 20 & 15) != 15,
                 "Added function requires nonempty relocation-free executable .text COMDAT")
        header = 20 + (section - 1) * 40
        _require(candidate.data[header + 8:header + 16] == bytes(8),
                 "Added function has nonzero virtual metadata")
        members = [(i, s) for i, s in candidate.symbols.items() if s["section"] == section]
        _require(len(members) == 2 and members[1][0] == index
                 and members[0][1]["name"] == ".text" and members[0][1]["type"] == 0
                 and members[0][1]["storage"] == 3 and members[0][1]["value"] == 0
                 and members[0][1]["aux_count"] == 1,
                 "Added section must contain only its section definition and public function")
        _require(section not in added, "Added functions share a section")
        added[section] = name
        excluded_symbols.update(i for i, _ in members)
    retained = [i for i in range(1, len(candidate.sections) + 1) if i not in added]
    _require(len(retained) == len(baseline.sections), "Unexpected section count")
    section_map = dict(zip(retained, range(1, len(retained) + 1)))
    left = _View(baseline, {i: i for i in range(1, len(baseline.sections) + 1)}, names,
                 allow_source_path_change, allow_nonselecting_checksum_change)
    right = _View(candidate, {**section_map, **{i: ("added", name) for i, name in added.items()}},
                  names, allow_source_path_change, allow_nonselecting_checksum_change)
    for section in added:
        aux = candidate.auxiliary_records[right.section_definitions[section]]
        _require(aux[14] != 5, "Added function cannot be an associative COMDAT")
    for candidate_section, baseline_section in section_map.items():
        original, current = baseline.sections[baseline_section - 1], candidate.sections[candidate_section - 1]
        label = f"section {baseline_section} -> {candidate_section}"
        _require(original["name"] == current["name"] and original["flags"] == current["flags"],
                 f"Section name or flags differ: {label}")
        _require(baseline.data[20 + (baseline_section - 1) * 40 + 8:20 + (baseline_section - 1) * 40 + 16]
                 == candidate.data[20 + (candidate_section - 1) * 40 + 8:20 + (candidate_section - 1) * 40 + 16],
                 f"Section virtual metadata differs: {label}")
        _require(original["data"] == current["data"], f"Raw section bytes differ: {label}")
        def relocations(view, entry):
            return [(offset, kind, view.key(index, relocation=True))
                    for offset, index, kind in entry["relocations"]]
        _require(relocations(left, original) == relocations(right, current),
                 f"Relocations or target identities differ: {label}")
        if (original["flags"] & COMDAT
                and baseline.auxiliary_records[left.section_definitions[baseline_section]][14] != 5):
            leader = next(i for i, s in baseline.symbols.items()
                          if s["section"] == baseline_section and i != left.section_definitions[baseline_section])
            other = next(i for i, s in candidate.symbols.items()
                         if s["section"] == candidate_section and i != right.section_definitions[candidate_section])
            _require(left.key(leader) == right.key(other), f"COMDAT leader differs: {label}")
    before = Counter(left.key(i) for i in baseline.symbols)
    after = Counter(right.key(i) for i in candidate.symbols if i not in excluded_symbols)
    promoted = 0
    for index, symbol in baseline.symbols.items():
        key = left.key(index)
        if left.is_added_reference(symbol) and symbol["section"] == 0 and before[key] > after[key]:
            # One new definition can replace one declaration, not erase an
            # arbitrary number of unchanged undefined records.
            _require(sum(s["name"] == symbol["name"] for s in baseline.symbols.values()) == 1,
                     "Ambiguous added-function declaration replacement")
            before[key] -= 1
            promoted += 1
    _require(before == after, "Symbol identities, multiplicities or auxiliary metadata differ")
    section_checksum_changes = []
    if allow_nonselecting_checksum_change:
        for candidate_section, baseline_section in section_map.items():
            if baseline_section not in left.section_definitions:
                continue
            old = baseline.auxiliary_records[left.section_definitions[baseline_section]]
            new = candidate.auxiliary_records[right.section_definitions[candidate_section]]
            if u32(old, 8) != u32(new, 8):
                section_checksum_changes.append({"baseline_section": baseline_section,
                                                 "candidate_section": candidate_section,
                                                 "selection": old[14],
                                                 "before": u32(old, 8), "after": u32(new, 8)})
    source_path_changes = []
    if allow_source_path_change:
        before_paths, after_paths = {}, {}
        for view, paths in ((left, before_paths), (right, after_paths)):
            for index, path in view.file_paths.items():
                key = view.key(index)
                _require(key not in paths, "Ambiguous source-file metadata")
                paths[key] = path
        for key, path in before_paths.items():
            if path != after_paths[key]:
                source_path_changes.append({"before": path, "after": after_paths[key]})
    return {
        "equivalent": True,
        "baseline_sha256": hashlib.sha256(baseline.data).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.data).hexdigest(),
        "preserved_sections": len(baseline.sections),
        "preserved_bytes": sum(len(s["data"]) for s in baseline.sections),
        "preserved_relocations": sum(len(s["relocations"]) for s in baseline.sections),
        "preserved_symbols": len(baseline.symbols) - promoted,
        "promoted_undefined_symbols": promoted,
        "source_path_changes": source_path_changes,
        "section_checksum_changes": section_checksum_changes,
        "section_map": [{"baseline": b, "candidate": c} for c, b in section_map.items()],
        "added_functions": [{"name": name, "section": section,
                             "bytes": len(candidate.sections[section - 1]["data"])}
                            for section, name in sorted(added.items())],
    }
