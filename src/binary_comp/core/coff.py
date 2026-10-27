"""Strict i386 COFF objects, explicit relocations and complete code/data windows.

The parser retains native symbol indices, weak fallbacks and local labels. The
comparison helpers accept explicit symbol/address bindings and a reference
object with ``read(va, size)``; they never normalize arbitrary operands.
Complete code regions additionally require ``section_for_va(va)`` returning
section flags, start/end and rawsize, plus strictly file-backed reads. No
compiler, linker, project layout or build command is assumed here.
"""
from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
import struct

from .binary import compare_binary


# A symbol's requested size is metadata; materializing arbitrary 32-bit values
# as host allocations is not safe. Larger contributions remain inspectable.
MAX_COMMON_BYTES = 16 * 1024 * 1024


def u16(data, offset):
    return struct.unpack_from("<H", data, offset)[0]


def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def is_function_type(value):
    """Test the first COFF derived-type field, preserving any base return type."""
    return (value & 0x30) == 0x20


def _is_common_symbol(symbol):
    kind = symbol["type"]
    return (symbol["section"] == 0 and symbol["value"] > 0
            and symbol["storage"] == 2 and symbol.get("aux_count", 0) == 0
            and not is_function_type(kind)
            and ((kind & 0xF) != 1 or (kind & 0x30) == 0x10))


def _is_ia32_alignment(data):
    """Recognize only short, side-effect-free compiler alignment instructions."""
    offset = 0
    while offset < len(data):
        for instruction in (b"\x90", b"\x8b\xff", b"\x8d\x49\x00"):
            if data.startswith(instruction, offset):
                offset += len(instruction)
                break
        else:
            return False
    return True


class COFF:
    """MSVC i386 object, retaining original symbol indices for relocations."""

    def __init__(self, path):
        self.data = Path(path).read_bytes()
        self._parse()

    @classmethod
    def from_bytes(cls, data):
        """Parse an in-memory object without a temporary file."""
        result = cls.__new__(cls)
        result.data = bytes(data)
        result._parse()
        return result

    def _require_range(self, offset, size, label):
        if not 0 <= offset <= offset + size <= len(self.data):
            raise ValueError(f"Truncated COFF {label}")

    def _parse(self):
        self._require_range(0, 20, "header")
        if u16(self.data, 0) != 0x14C or u16(self.data, 16) != 0:
            raise ValueError("Expected i386 COFF object without optional header")
        symbol_table, count = struct.unpack_from("<II", self.data, 8)
        self._require_range(20, u16(self.data, 2) * 40, "section headers")
        self._require_range(symbol_table, count * 18, "symbol table")
        strings = symbol_table + count * 18
        self._require_range(strings, 4, "string table")
        string_size = u32(self.data, strings)
        # CVTRES writes zero for an empty table, but still emits its size field.
        if string_size == 0:
            string_size = 4
        if string_size < 4 or strings + string_size > len(self.data):
            raise ValueError("Truncated COFF string table")

        def name_at(pos):
            name = self.data[pos:pos + 8]
            if name[:4] == b"\0" * 4:
                offset = u32(name, 4)
                if not 4 <= offset < string_size:
                    raise ValueError("Bad COFF string offset")
                end = self.data.index(b"\0", strings + offset, strings + string_size)
                name = self.data[strings + offset:end]
            return name.rstrip(b"\0").decode("ascii")

        self.sections = []
        for i in range(u16(self.data, 2)):
            pos = 20 + i * 40
            size, raw, relocs = struct.unpack_from("<III", self.data, pos + 16)
            nrelocs = u16(self.data, pos + 32)
            flags = u32(self.data, pos + 36)
            if flags & 0x80:  # IMAGE_SCN_CNT_UNINITIALIZED_DATA: size is allocation, not file bytes.
                if raw or nrelocs or flags & 0x60:
                    raise ValueError("Malformed uninitialized COFF section")
                section_data = bytes(size)
            else:
                if (size and not raw) or raw + size > len(self.data):
                    raise ValueError("Truncated COFF section or missing raw data")
                section_data = self.data[raw:raw + size]
            if (nrelocs and not relocs) or relocs + nrelocs * 10 > len(self.data):
                raise ValueError("Truncated COFF relocations")
            self.sections.append({
                "name": self.data[pos:pos + 8].rstrip(b"\0").decode("ascii"),
                "data": section_data,
                "flags": flags,
                "relocations": [struct.unpack_from("<IIH", self.data, relocs + j * 10)
                                for j in range(nrelocs)],
            })
        self.symbols = {}
        self.named = {}
        i = 0
        while i < count:
            pos = symbol_table + i * 18
            name = name_at(pos)
            value, section, kind, storage, aux = struct.unpack_from("<IhHBB", self.data, pos + 8)
            if i + aux >= count:
                raise ValueError("Truncated COFF auxiliary symbols")
            symbol = {"name": name, "value": value, "section": section,
                      "type": kind, "storage": storage, "aux_count": aux}
            if storage == 105:  # IMAGE_SYM_CLASS_WEAK_EXTERNAL
                if aux != 1:
                    raise ValueError(f"Malformed weak external {name}")
                symbol["weak_fallback"] = u32(self.data, pos + 18)
                symbol["weak_search"] = u32(self.data, pos + 22)
                if symbol["weak_search"] not in (1, 2, 3):
                    raise ValueError(f"Unsupported weak external search mode for {name}")
            self.symbols[i] = symbol
            if (storage in (2, 105) or (storage in (3, 6) and section > 0
                    and (aux == 0 or is_function_type(kind)))):
                previous = self.named.get(name)
                if previous and previous["section"] > 0 and section > 0:
                    raise ValueError(f"Ambiguous symbol definitions for {name}")
                if (previous and _is_common_symbol(previous) and _is_common_symbol(symbol)
                        and previous["value"] != value):
                    raise ValueError(f"Conflicting COMMON sizes for {name}")
                # MSVC can emit an unresolved reference followed by its COMDAT
                # definition. Preserve both indices; resolve names to the definition.
                if (not previous or section > 0 or
                        (previous["section"] == 0 and not _is_common_symbol(previous)
                         and (_is_common_symbol(symbol) or storage == 105))):
                    self.named[name] = symbol
            i += 1 + aux

    def resolve_symbol(self, index):
        """Resolve this object's strong definitions and declared weak fallbacks."""
        seen = set()
        while True:
            if index in seen:
                raise ValueError("Cyclic COFF weak external fallback")
            seen.add(index)
            if index not in self.symbols:
                raise ValueError(f"Invalid COFF symbol index {index}")
            symbol = self.symbols[index]
            symbol = self.named.get(symbol["name"], symbol)
            if symbol["section"] > 0 or "weak_fallback" not in symbol:
                return symbol
            index = symbol["weak_fallback"]

    def definition_extent(self, name):
        """Return a COMMON request size or a complete section-backed extent."""
        common = self.common_size(name)
        if common is not None:
            return common
        symbol, section = self.definition(name)
        start = symbol["value"]
        end = min((s["value"] for s in self.symbols.values()
                   if s["section"] == symbol["section"] and s["value"] > start),
                  default=len(section["data"]))
        return end - start

    def common_size(self, name):
        """Return a sectionless COMMON contribution's size, or None.

        This describes one object's request, not the final linker's coalesced
        allocation, placement, or ownership. Undefined externs are not data.
        """
        symbol = self.named[name]
        return symbol["value"] if _is_common_symbol(symbol) else None

    def materialize_common(self, name):
        """Materialize a bounded COMMON request as logical zeros, not a section."""
        size = self.common_size(name)
        if size is None:
            raise ValueError(f"Not a COMMON data definition: {name}")
        if size > MAX_COMMON_BYTES:
            raise ValueError(f"COMMON contribution exceeds materialization limit: {name}")
        return bytes(size)

    def definition(self, name):
        """Return an actual section-backed definition; COMMON has no section."""
        symbol = self.named[name]
        if not 1 <= symbol["section"] <= len(self.sections):
            raise ValueError(f"Undefined symbol {name}")
        return symbol, self.sections[symbol["section"] - 1]

    def _section_functions(self, section):
        return [s for s in self.symbols.values()
                if 1 <= s["section"] <= len(self.sections)
                and self.sections[s["section"] - 1] is section
                and is_function_type(s["type"])]

    def code_region(self, name, entries, *, anchor_offset=0):
        """Select a complete multi-entry contribution from its native records."""
        anchor, section = self.definition(name)
        if (type(anchor_offset) is not int
                or not 0 <= anchor_offset < len(section["data"])):
            raise ValueError("Invalid code region anchor_offset")
        if (not section["flags"] & 0x20 or not section["flags"] & 0x20000000
                or not is_function_type(anchor["type"]) or anchor["value"] != anchor_offset):
            raise ValueError("Code region needs a typed executable anchor at anchor_offset")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Code region needs an exhaustive typed-entry list")
        expected = {}
        for entry in entries:
            if (not isinstance(entry, Mapping) or not isinstance(entry.get("symbol"), str)
                    or not entry["symbol"] or type(entry.get("offset")) is not int
                    or not 0 <= entry["offset"] < len(section["data"])):
                raise ValueError("Invalid code region entry")
            if entry["symbol"] in expected:
                raise ValueError("Duplicate code region entry")
            expected[entry["symbol"]] = entry["offset"]
        functions = self._section_functions(section)
        actual = {}
        for symbol in functions:
            if (symbol["storage"] not in (2, 3) or symbol["name"] in actual
                    or not 0 <= symbol["value"] < len(section["data"])):
                raise ValueError("Unsupported native code region entry")
            actual[symbol["name"]] = symbol["value"]
        if actual != expected or name not in expected:
            raise ValueError("Code region entries differ from native typed records")
        if len(set(actual.values())) < 2:
            raise ValueError("Code region requires distinct typed entry offsets")
        return section

    def inline_table(self, name, size, *, index_entries=None, alignment=4):
        """Bound native local case pointers or byte indices in a /Gy code section."""
        if type(alignment) is not int or alignment not in (1, 2, 4):
            raise ValueError(f"Invalid inline_alignment for {name}")
        if index_entries is not None and alignment != 4:
            raise ValueError(f"inline_alignment requires a native pointer table: {name}")
        symbol, section = self.definition(name)
        if (not section["flags"] & 0x20 or symbol["storage"] != 6
                or is_function_type(symbol["type"])):
            raise ValueError(f"Expected an inline code-section data label: {name}")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid inline table size: {name}")
        functions = self._section_functions(section)
        if len(functions) != 1 or functions[0]["value"] != 0:
            raise ValueError("Inline data requires one /Gy function per section")
        start, end = symbol["value"], symbol["value"] + size
        boundary = start + self.definition_extent(name)
        if not 0 < start < end <= boundary <= len(section["data"]):
            raise ValueError(f"Invalid inline table extent: {name}")
        # A following native symbol can bound four-byte table alignment. The
        # final definition can extend only to its next 16-byte boundary.
        tail = section["data"][end:boundary]
        if (boundary != len(section["data"]) and tail
                and boundary != (end + 3) // 4 * 4):
            raise ValueError(f"Excessive or unaligned inline table padding: {name}")
        if boundary > (end + 15) // 16 * 16 or not _is_ia32_alignment(tail):
            raise ValueError(f"Unknown or excessive inline table padding: {name}")
        if any(offset < 0 or offset + 4 > len(section["data"])
               for offset, _, _ in section["relocations"]):
            raise ValueError(f"Relocation outside inline table section: {name}")
        if any(offset < start < offset + 4 or offset < end < offset + 4
               for offset, _, _ in section["relocations"]):
            raise ValueError(f"Relocation crosses inline table boundary: {name}")
        relocations = sorted((offset - start, index, kind)
                             for offset, index, kind in section["relocations"]
                             if start <= offset < end)
        if index_entries is None:
            if (start % alignment or size % 4
                    or [offset for offset, _, _ in relocations] != list(range(0, size, 4))
                    or any(kind != 0x6 for _, _, kind in relocations)):
                raise ValueError(f"Inline table needs a complete contiguous DIR32 payload: {name}")
            for offset, index, _ in relocations:
                target = self.resolve_symbol(index)
                if (target["section"] != symbol["section"] or target["storage"] != 6
                        or is_function_type(target["type"]) or not 0 <= target["value"] < start):
                    raise ValueError(f"Inline table target is not a local case label: {name}")
                if u32(section["data"], start + offset):
                    raise ValueError(f"Inline case-pointer addend must be zero: {name}")
        elif (type(index_entries) is not int or not 0 < index_entries <= 256
              or relocations or any(value >= index_entries for value in section["data"][start:end])):
            raise ValueError(f"Invalid native byte-index table: {name}")
        if any(end <= offset < boundary for offset, _, _ in section["relocations"]):
            raise ValueError(f"Relocation in inline table padding: {name}")
        return {"symbol": symbol, "owner": functions[0], "section": section,
                "start": start, "end": end, "data": section["data"][start:end],
                "relocations": relocations, "alignment_padding": tail,
                "trailing_nop_padding": tail.count(b"\x90")}

    def function(self, name):
        symbol, section = self.definition(name)
        if not section["flags"] & 0x20:
            raise ValueError(f"Not a code symbol: {name}")
        if is_function_type(symbol["type"]):
            functions = self._section_functions(section)
            if any(s["value"] != 0 for s in functions):
                raise ValueError("Comparison requires all /Gy function entries at section offset zero")
            return section
        # VC6 emits EH cleanup/handler entries as labels in a shared .text$x
        # section. Bound each complete entry by the next symbol or section end.
        if symbol["storage"] != 6 or symbol["type"] != 0:
            raise ValueError(f"Not a function or compiler code label: {name}")
        if any(s["section"] == symbol["section"] and is_function_type(s["type"])
               for s in self.named.values()):
            raise ValueError(f"Local function labels are not separate code entries: {name}")
        start = symbol["value"]
        end = start + self.definition_extent(name)
        if not 0 <= start < end <= len(section["data"]):
            raise ValueError(f"Invalid code label extent: {name}")
        if any(offset < 0 or offset + 4 > len(section["data"])
               for offset, _, _ in section["relocations"]):
            raise ValueError(f"Relocation outside code section: {name}")
        relocations = [(offset - start, index, kind)
                       for offset, index, kind in section["relocations"]
                       if offset < end and offset + 4 > start]
        return dict(section, data=section["data"][start:end], relocations=relocations)


class BodyExtentError(ValueError):
    """The complete compiler function does not fit the configured body boundary."""


def _checked_origin(origin):
    if type(origin) is not int or not 0 <= origin <= 0xFFFFFFFF:
        raise ValueError(f"Origin outside i386 address domain: {origin!r}")
    return origin


def _binding_address(value, name):
    address = int(value, 16)
    if not 0 <= address <= 0xFFFFFFFF:
        raise ValueError(f"Binding outside i386 address domain: {name}")
    return address


def _validate_function_alias_bindings(coff, section, origin, bindings):
    """Co-located entry names cannot acquire different configured origins."""
    functions = coff._section_functions(section)
    if any(s["value"] != 0 for s in functions):
        raise ValueError("Comparison requires all /Gy function entries at section offset zero")
    if len(functions) <= 1:
        return
    for symbol in functions:
        name = symbol["name"]
        if name in bindings and _binding_address(bindings[name]["va"], name) != origin:
            raise ValueError(f"Co-located function alias address differs from origin: {name}")


def inline_table_alignment(binding, name):
    """Validate the optional alignment contract only for explicit pointer tables."""
    if "inline_alignment" not in binding:
        return 4
    alignment = binding["inline_alignment"]
    if type(alignment) is not int or alignment not in (1, 2, 4):
        raise ValueError(f"Invalid inline_alignment for {name}")
    if (binding.get("inline_data") is not True
            or type(binding.get("data_size")) is not int or binding["data_size"] <= 0
            or {"data_hex", "inline_index_for", "executable_data"} & binding.keys()):
        raise ValueError(f"inline_alignment requires inline_data=true and a complete pointer table: {name}")
    return alignment


def apply_fixups(coff, raw, relocations, origin, bindings, reference, visiting=None,
                 *, common_definitions=None, code_regions=None):
    _checked_origin(origin)
    if code_regions is not None:
        code_regions.check_context(coff, bindings, reference, common_definitions)
    if visiting is None:
        validate_common_definitions(coff, bindings, reference, common_definitions, code_regions=code_regions)
    body = bytearray(raw)
    occupied = set()
    for offset, index, kind in relocations:
        if offset < 0 or offset + 4 > len(body) or occupied.intersection(range(offset, offset + 4)):
            raise ValueError("Relocation outside object or overlapping relocation")
        occupied.update(range(offset, offset + 4))
        symbol = coff.resolve_symbol(index)
        name = symbol["name"]
        if name not in bindings:
            raise ValueError(f"Unresolved relocation: {name}")
        binding = bindings[name]
        inline_table_alignment(binding, name)
        address = _binding_address(binding["va"], name)
        if symbol["section"] != 0 and not 1 <= symbol["section"] <= len(coff.sections):
            raise ValueError(f"Unsupported relocation symbol section: {name}")
        if symbol["section"] > 0 and coff.sections[symbol["section"] - 1]["flags"] & 0x20:
            if code_regions is not None:
                code_regions.validate_target(coff, symbol, address, visiting)
            else:
                _validate_function_alias_bindings(
                    coff, coff.sections[symbol["section"] - 1],
                    address if is_function_type(symbol["type"]) else address - symbol["value"], bindings)
        if "data_hex" in binding or "data_size" in binding:
            validate_data_binding(coff, name, bindings, reference, visiting,
                                  common_definitions=common_definitions, code_regions=code_regions)
        elif (_is_common_symbol(symbol) or (symbol["section"] != 0 and
              not coff.sections[symbol["section"] - 1]["flags"] & 0x20)):
            raise ValueError(f"Defined data symbol needs complete data validation: {name}")
        addend = u32(body, offset)
        if kind == 0x6:  # IMAGE_REL_I386_DIR32
            value = address + addend
        elif kind == 0x14:  # IMAGE_REL_I386_REL32
            value = address + addend - (origin + offset + 4)
        else:
            raise ValueError(f"Unsupported relocation type {kind:#x}")
        struct.pack_into("<I", body, offset, value & 0xFFFFFFFF)
    return bytes(body), len(relocations)


def validate_common_definitions(coff, bindings, reference, common_definitions=None, *, code_regions=None):
    """Eagerly validate explicit same-size strong providers for native COMMONs.

    Providers are already parsed objects. No file lookup, symbol search or
    section synthesis takes place in the core. All use the caller's bindings.
    """
    if code_regions is not None:
        code_regions.check_context(coff, bindings, reference, common_definitions)
    for name, binding in bindings.items():
        inline_table_alignment(binding, name)
    if common_definitions is None:
        return
    if not isinstance(common_definitions, Mapping):
        raise ValueError("COMMON definitions must map names to COFF providers")
    for name, provider in common_definitions.items():
        if not isinstance(name, str) or not name or not isinstance(provider, COFF):
            raise ValueError("Invalid COMMON provider record")
        common = coff.common_size(name) if name in coff.named else None
        if common is None:
            raise ValueError(f"COMMON provider needs a recognized consumer COMMON: {name}")
        if (not isinstance(bindings.get(name), Mapping)
                or not ({"data_size", "data_hex"} & bindings[name].keys())):
            raise ValueError(f"COMMON provider needs complete data validation: {name}")
        try:
            symbol, section = provider.definition(name)
        except (KeyError, ValueError) as error:
            raise ValueError(f"COMMON provider needs a same-name strong data definition: {name}") from error
        if (symbol["storage"] != 2 or symbol["aux_count"] != 0
                or is_function_type(symbol["type"])
                or ((symbol["type"] & 0xF) == 1 and (symbol["type"] & 0x30) != 0x10)
                or section["flags"] & 0x20 or not section["flags"] & 0xC0
                or "weak_fallback" in symbol):
            raise ValueError(f"COMMON provider needs a same-name strong data definition: {name}")
        if provider.definition_extent(name) != common:
            raise ValueError(f"COMMON provider extent differs from requested size: {name}")
    # A configured but unreferenced mapping must not hide wrong provider bytes,
    # missing dependencies, a bare address or an invalid contribution extent.
    for name in common_definitions:
        validate_data_binding(coff, name, bindings, reference, set(),
                              common_definitions=common_definitions, code_regions=code_regions)


def validate_data_binding(coff, name, bindings, reference, visiting=None,
                          *, common_definitions=None, code_regions=None):
    """Compare a complete data definition, including all checked pointer fixups."""
    if code_regions is not None:
        code_regions.check_context(coff, bindings, reference, common_definitions)
    if visiting is None:
        validate_common_definitions(coff, bindings, reference, common_definitions, code_regions=code_regions)
        visiting = set()
    identity = id(coff), name
    if identity in visiting:
        raise ValueError(f"Cyclic data binding: {name}")
    binding = bindings[name]
    inline_table_alignment(binding, name)
    address = _binding_address(binding["va"], name)
    common = coff.common_size(name)
    if common is not None and common > MAX_COMMON_BYTES:
        raise ValueError(f"COMMON contribution exceeds materialization limit: {name}")
    if "data_hex" in binding:
        if "data_size" in binding:
            raise ValueError(f"Ambiguous data size for {name}")
        expected = bytes.fromhex(binding["data_hex"])
        size = len(expected)
    else:
        size = binding["data_size"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid data size for {name}")
    if common is not None:
        if "inline_data" in binding or "inline_index_for" in binding:
            raise ValueError(f"COMMON data has no inline section: {name}")
        if size != common:
            raise ValueError(f"Configured data extent differs from COMMON contribution: {name}")
        _validate_common_range(name, bindings, address, size)
        if common_definitions is not None and name in common_definitions:
            visiting.add(identity)
            try:
                return validate_data_binding(common_definitions[name], name, bindings,
                                             reference, visiting,
                                             common_definitions=common_definitions, code_regions=code_regions)
            finally:
                visiting.remove(identity)
    if "data_hex" in binding:
        if not expected or reference.read(address, size) != expected:
            raise ValueError(f"Original constant differs for {name}")
    else:
        expected = reference.read(address, size)
    if common is not None:
        actual = coff.materialize_common(name)
        if not compare_binary(expected, actual).exact:
            raise ValueError(f"Rebuilt data differs from original: {name}")
        return actual, 0
    definition, section = coff.definition(name)
    inline = "inline_data" in binding
    if "inline_index_for" in binding and not inline:
        raise ValueError(f"Byte indices require an explicit inline table: {name}")
    if inline:
        table = inline_table(coff, name, bindings)
        owner = bindings.get(table["owner"]["name"], {})
        if "va" not in owner or type(owner.get("size")) is not int or owner["size"] <= 0:
            raise ValueError(f"Inline table needs its configured function extent: {name}")
        origin = _binding_address(owner["va"], table["owner"]["name"])
        if address != origin + table["start"]:
            raise ValueError(f"Inline table address differs from native code offset: {name}")
        padding = table["alignment_padding"]
        if padding and reference.read(address + size, len(padding)) != padding:
            raise ValueError(f"Inline table alignment differs from original: {name}")
        for _, index, _ in table["relocations"]:
            target = coff.resolve_symbol(index)
            target_binding = bindings.get(target["name"])
            if target_binding is None:
                raise ValueError(f"Unresolved relocation: {target['name']}")
            if (target["value"] >= owner["size"]
                    or _binding_address(target_binding["va"], target["name"]) != origin + target["value"]):
                raise ValueError(f"Inline case target differs from native code offset: {target['name']}")
        if "inline_index_for" in binding:
            validate_data_binding(coff, binding["inline_index_for"], bindings, reference,
                                  visiting | {identity}, common_definitions=common_definitions, code_regions=code_regions)
    elif is_function_type(definition["type"]) or section["flags"] & 0x20:
        raise ValueError(f"Expected a data definition for {name}")
    if not inline and coff.definition_extent(name) != size:
        raise ValueError(f"Configured data extent differs from COFF definition: {name}")
    start, end = definition["value"], definition["value"] + size
    raw = section["data"][start:end]
    if len(raw) != size:
        raise ValueError(f"Truncated data definition: {name}")
    if any(offset < 0 or offset + 4 > len(section["data"])
           for offset, _, _ in section["relocations"]):
        raise ValueError(f"Relocation outside data section: {name}")
    relocations = [(offset - start, index, kind)
                   for offset, index, kind in section["relocations"]
                   if offset < end and offset + 4 > start]
    visiting.add(identity)
    try:
        actual, fixups = apply_fixups(coff, raw, relocations, address, bindings, reference,
                                     visiting, common_definitions=common_definitions, code_regions=code_regions)
    finally:
        visiting.remove(identity)
    if not compare_binary(expected, actual).exact:
        raise ValueError(f"Rebuilt data differs from original: {name}")
    return actual, fixups


def _validate_common_range(name, bindings, address, size):
    """Keep an explicit COMMON contribution bounded and separate from owners."""
    if address + size > 0x100000000:
        raise ValueError(f"COMMON contribution outside i386 address domain: {name}")
    for other, binding in bindings.items():
        if other == name:
            continue
        if "data_size" in binding:
            length = binding["data_size"]
        elif "data_hex" in binding:
            length = len(bytes.fromhex(binding["data_hex"]))
        else:
            continue
        if type(length) is not int or length <= 0:
            raise ValueError(f"Invalid data size for {other}")
        start = _binding_address(binding["va"], other)
        if address < start + length and start < address + size:
            raise ValueError(f"COMMON contribution overlaps configured data: {name}, {other}")


def inline_table(coff, name, bindings):
    """Require explicit native table roles; never infer data by decoding bytes."""
    binding = bindings[name]
    alignment = inline_table_alignment(binding, name)
    if (binding.get("inline_data") is not True or "data_size" not in binding
            or "data_hex" in binding):
        raise ValueError(f"Inline table requires inline_data=true and data_size: {name}")
    relation = binding.get("inline_index_for")
    if "inline_index_for" not in binding:
        return coff.inline_table(name, binding["data_size"], alignment=alignment)
    if (not isinstance(relation, str) or relation not in bindings
            or "inline_index_for" in bindings[relation]):
        raise ValueError(f"Invalid inline byte-index relation: {name}")
    pointers = inline_table(coff, relation, bindings)
    table = coff.inline_table(name, binding["data_size"], index_entries=len(pointers["data"]) // 4)
    if table["section"] is not pointers["section"] or pointers["end"] != table["start"]:
        raise ValueError(f"Byte-index table must follow its native pointer table: {name}")
    return table


def code_window(coff, section, origin, size, bindings, reference, *, diagnostic=False,
                common_definitions=None, code_regions=None):
    """Bound code and native inline data once for both the gate and reviews."""
    _checked_origin(origin)
    validate_common_definitions(coff, bindings, reference, common_definitions, code_regions=code_regions)
    if code_regions is not None:
        code_regions.validate_code_window(section, origin, size)
    _validate_function_alias_bindings(coff, section, origin, bindings)
    raw = section["data"]
    tables = []
    for name, binding in bindings.items():
        if "inline_index_for" in binding and "inline_data" not in binding:
            raise ValueError(f"Byte indices require an explicit inline table: {name}")
        if "inline_data" not in binding:
            continue
        _, table_section = coff.definition(name)
        if table_section is section:
            tables.append((name, inline_table(coff, name, bindings)))
    if tables:
        tables.sort(key=lambda item: item[1]["start"])
        for (_, previous), (_, following) in zip(tables, tables[1:]):
            if previous["end"] + len(previous["alignment_padding"]) != following["start"]:
                raise ValueError("Inline data definitions must cover the complete native table suffix")
        last = tables[-1][1]
        if last["end"] + len(last["alignment_padding"]) != len(raw):
            raise ValueError("Unconfigured data follows native inline table")
        start = tables[0][1]["start"]
        if size > start:
            raise ValueError("Configured code extent overlaps native inline data")
        mapping = dict(bindings)
        owner = tables[0][1]["owner"]["name"]
        mapping[owner] = {"va": hex(origin), "size": size}
        table_regions = (None if code_regions is None else
                         code_regions.for_code_window(mapping, owner, origin, size))
        if not diagnostic:
            alignment = inline_table_alignment(bindings[tables[0][0]], tables[0][0])
            if start != (size + alignment - 1) // alignment * alignment:
                raise BodyExtentError(f"Inline table offset {start} incompatible with original code extent {size}")
            gap = raw[size:start]
            if (any(byte != 0x90 for byte in gap)
                    or reference.read(origin + size, len(gap)) != gap):
                raise BodyExtentError("Non-NOP or changed alignment before inline table")
        for name, _ in tables:
            validate_data_binding(coff, name, mapping, reference,
                                  common_definitions=common_definitions, code_regions=table_regions)
        end = start if diagnostic else size
        relocations = [(offset, index, kind) for offset, index, kind in section["relocations"]
                       if offset < start]
        if any(offset < 0 or offset + 4 > end for offset, _, _ in relocations):
            raise ValueError("Relocation crosses code/inline alignment boundary")
        return {"data": raw[:end], "relocations": relocations, "bindings": mapping,
                "code_regions": table_regions,
                "inline_data_bytes": sum(len(t["data"]) for _, t in tables),
                "inline_data_symbols": [name for name, _ in tables],
                "inline_alignment_bytes": sum(len(t["alignment_padding"]) for _, t in tables),
                "excluded_nop_padding": sum(t["trailing_nop_padding"] for _, t in tables)
                                        + (0 if diagnostic else start - size)}
    if diagnostic:
        return {"data": raw, "relocations": section["relocations"], "bindings": bindings}
    # Only bounded NOP alignment outside the complete original body is excluded.
    if len(raw) < size or len(raw) > (size + 15) // 16 * 16:
        raise BodyExtentError(f"Function length {len(raw)} incompatible with original {size}")
    if any(x != 0x90 for x in raw[size:]):
        raise BodyExtentError("Non-NOP instructions extend beyond original function body")
    return {"data": raw[:size], "relocations": section["relocations"], "bindings": bindings,
            "excluded_nop_padding": len(raw) - size}


def relocate(coff, section, origin, size, bindings, reference, *, common_definitions=None, code_regions=None):
    window = code_window(coff, section, origin, size, bindings, reference,
                         common_definitions=common_definitions, code_regions=code_regions)
    return apply_fixups(coff, window["data"], window["relocations"], origin,
                        window["bindings"], reference, common_definitions=common_definitions,
                        code_regions=window.get("code_regions", code_regions))


def compare_function(coff, function, version, bindings, reference, *, common_definitions=None, code_regions=None):
    validate_common_definitions(coff, bindings, reference, common_definitions, code_regions=code_regions)
    target = function[version]
    address, size = _binding_address(target["va"], function["symbol"]), target["size"]
    section = coff.function(function["symbol"])
    actual, relocations = relocate(coff, section, address, size, bindings, reference,
                                   common_definitions=common_definitions, code_regions=code_regions)
    expected = reference.read(address, size)
    return compare_binary(expected, actual).exact, actual, expected, relocations


class CodeRegions:
    """Explicit complete regions for one COFF owner and its binding namespace.

    This is validation context, not permission to relax the function gate.
    Every use rechecks the complete relevant region; no byte result is cached.
    Provider-owned regions require a separate ownership model and are rejected.
    """

    def __init__(self, coff, regions, version, bindings, reference, *, common_definitions=None):
        self.coff = coff
        self.bindings = deepcopy(bindings)
        self.reference = reference
        if common_definitions is not None and not isinstance(common_definitions, Mapping):
            raise ValueError("COMMON definitions must map names to COFF providers")
        self.common_definitions = None if common_definitions is None else dict(common_definitions)
        self.version = version
        for name, binding in self.bindings.items():
            inline_table_alignment(binding, name)
        self.regions = {}
        self.sections = {}
        for region in regions:
            if version not in region:
                continue
            name = region["symbol"]
            if name in self.regions:
                raise ValueError("Duplicate code region owner")
            section = self.coff.code_region(name, region.get("entries"),
                                            anchor_offset=region.get("anchor_offset", 0))
            identity = id(section)
            if identity in self.sections:
                raise ValueError("Duplicate COFF code contribution ownership")
            self._metadata(region)
            record = deepcopy(region)
            self.regions[name] = record
            self.sections[identity] = record

    def check_context(self, coff, bindings, reference, common_definitions):
        if (bindings != self.bindings or reference is not self.reference
                or common_definitions != self.common_definitions
                or (coff is not self.coff and not any(
                    coff is provider for provider in (self.common_definitions or {}).values()))):
            raise ValueError("Code region context differs from comparison inputs")

    def for_code_window(self, bindings, owner, origin, size):
        """Retain only the ordinary inline-table owner's checked extent update."""
        symbol, section = self.coff.definition(owner)
        if (not is_function_type(symbol["type"]) or symbol["value"] != 0
                or id(section) in self.sections):
            raise ValueError("Invalid ordinary code owner for region dependencies")
        expected = deepcopy(self.bindings)
        previous = expected.get(owner)
        if previous is not None:
            if ({"data_size", "data_hex", "inline_data", "inline_index_for", "inline_alignment"} & previous.keys()
                    or _binding_address(previous["va"], owner) != origin):
                raise ValueError("Code region context differs from code owner address or role")
        expected[owner] = {"va": hex(origin), "size": size}
        if bindings != expected:
            raise ValueError("Code region context differs from code window bindings")
        return CodeRegions(self.coff, list(self.regions.values()), self.version,
                           bindings, self.reference, common_definitions=self.common_definitions)

    def validate_code_window(self, section, origin, size):
        """Ordinary code cannot also own any declared complete region bytes."""
        if id(section) in self.sections:
            raise ValueError("Duplicate COFF code contribution ownership")
        for region in self.regions.values():
            start = _binding_address(region[self.version]["va"], region["symbol"])
            if origin < start + region[self.version]["size"] and start < origin + size:
                raise ValueError("Ordinary code overlaps a complete code region")

    def _metadata(self, region):
        section = self.coff.code_region(region["symbol"], region.get("entries"),
                                        anchor_offset=region.get("anchor_offset", 0))
        target = region[self.version]
        origin = _binding_address(target["va"], region["symbol"])
        size = target["size"]
        if type(size) is not int or size <= 0 or origin + size > 0x100000000:
            raise ValueError("Invalid complete code region range")
        if size != len(section["data"]):
            raise ValueError("Code region size differs from complete COFF contribution")
        for other in self.regions.values():
            if other["symbol"] == region["symbol"]:
                continue
            start = _binding_address(other[self.version]["va"], other["symbol"])
            if origin < start + other[self.version]["size"] and start < origin + size:
                raise ValueError("Overlapping original code regions")
        if not callable(getattr(self.reference, "section_for_va", None)):
            raise ValueError("Code regions require original section metadata")
        backing = self.reference.section_for_va(origin)
        if (backing is None or not backing.flags & 0x20000000
                or origin + size > backing.end
                or origin + size > backing.start + backing.rawsize):
            raise ValueError("Code region original range must be executable and file-backed")
        self.reference.read(origin, size)  # Strict references reject virtual-only tails.
        for entry in region["entries"]:
            binding = self.bindings.get(entry["symbol"])
            if (binding is None or "data_size" in binding or "data_hex" in binding
                    or _binding_address(binding["va"], entry["symbol"]) != origin + entry["offset"]):
                raise ValueError(f"Code region entry needs its exact explicit address: {entry['symbol']}")
        for name, binding in self.bindings.items():
            symbol = self.coff.named.get(name)
            if symbol and symbol["section"] > 0 and self.coff.sections[symbol["section"] - 1] is section:
                if {"inline_data", "inline_index_for", "inline_alignment"} & binding.keys():
                    raise ValueError("Inline data roles are unsupported in code regions")
            if "data_hex" in binding or "data_size" in binding:
                length = (len(bytes.fromhex(binding["data_hex"])) if "data_hex" in binding
                          else binding["data_size"])
                address = _binding_address(binding["va"], name)
                if type(length) is not int or length <= 0:
                    raise ValueError("Invalid complete data extent")
                if origin < address + length and address < origin + size:
                    raise ValueError("Code region overlaps configured complete data")
        return section

    def compare(self, name, visiting=None):
        region = self.regions[name]
        section = self._metadata(region)
        if self.sections.get(id(section)) is not region:
            raise ValueError("Code region context differs from its native contribution")
        origin = _binding_address(region[self.version]["va"], name)
        visiting = set() if visiting is None else visiting
        identity = ("code region", id(self.coff), id(section))
        if identity in visiting:
            raise ValueError("Recursive code region comparison")
        visiting.add(identity)
        try:
            actual, fixups = apply_fixups(
                self.coff, section["data"], section["relocations"], origin,
                self.bindings, self.reference, visiting,
                common_definitions=self.common_definitions, code_regions=self)
        finally:
            visiting.remove(identity)
        expected = self.reference.read(origin, len(section["data"]))
        return compare_binary(expected, actual).exact, actual, expected, fixups

    def validate_target(self, coff, symbol, address, visiting):
        section = coff.sections[symbol["section"] - 1]
        if coff is not self.coff:
            if any(s["value"] != 0 for s in coff._section_functions(section)):
                raise ValueError("Provider-owned code regions are unsupported")
            _validate_function_alias_bindings(
                coff, section, address if is_function_type(symbol["type"]) else address - symbol["value"],
                self.bindings)
            return
        region = self.sections.get(id(section))
        if region is None:
            _validate_function_alias_bindings(
                coff, section, address if is_function_type(symbol["type"]) else address - symbol["value"],
                self.bindings)
            return
        if self._metadata(region) is not section:
            raise ValueError("Code region context differs from its native contribution")
        origin = _binding_address(region[self.version]["va"], region["symbol"])
        if not 0 <= symbol["value"] < len(section["data"]) or address != origin + symbol["value"]:
            raise ValueError("Code region target differs from its native offset")
        identity = ("code region", id(coff), id(section))
        if visiting is None or identity not in visiting:
            if not self.compare(region["symbol"], visiting)[0]:
                raise ValueError("Referenced complete code region differs from original")


def compare_code_region(coff, region, version, bindings, reference, *,
                        common_definitions=None, code_regions=None):
    """Compare one complete multi-entry section without a function-body claim."""
    if code_regions is None:
        code_regions = CodeRegions(coff, [region], version, bindings, reference,
                                   common_definitions=common_definitions)
    code_regions.check_context(coff, bindings, reference, common_definitions)
    if (code_regions.coff is not coff or code_regions.version != version
            or code_regions.regions.get(region["symbol"]) != region):
        raise ValueError("Code region context differs from its COFF owner")
    validate_common_definitions(coff, bindings, reference, common_definitions,
                                code_regions=code_regions)
    result = code_regions.compare(region["symbol"])
    if result[0]:
        for name, binding in bindings.items():
            if "data_size" in binding or "data_hex" in binding:
                validate_data_binding(coff, name, bindings, reference,
                                      common_definitions=common_definitions, code_regions=code_regions)
    return result
