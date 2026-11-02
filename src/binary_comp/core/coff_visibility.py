"""Expose explicitly named static COMDAT functions with storage-byte edits only."""
import hashlib

from .coff import COFF, is_function_type, u32
from .coff_extension import compare_coff_extension


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def globalize_coff_functions(coff: COFF, names):
    """Return (native object bytes, metadata witness), without mutating the input.

    Only unique, auxiliary-free static function leaders at offset zero in
    independent executable code COMDATs are supported. All bytes except their
    declared primary-symbol storage classes remain exact. This operation makes
    no claim about source recovery, instruction equality or a linked executable.
    """
    _require(not isinstance(names, (str, bytes)), "Function names must be a collection")
    names = list(names)
    _require(names and all(isinstance(name, str) and name for name in names)
             and len(names) == len(set(names)), "Function names must be distinct nonempty strings")
    # Reparse the authoritative bytes rather than trusting mutable cached views.
    coff = COFF.from_bytes(coff.data)
    compare_coff_extension(coff, coff)
    table = u32(coff.data, 8)
    table_end = table + u32(coff.data, 12) * 18
    protected = [(0, 20 + len(coff.sections) * 40)]
    for number, section in enumerate(coff.sections):
        header = 20 + number * 40
        if not section["flags"] & 0x80:
            raw, size = u32(coff.data, header + 20), u32(coff.data, header + 16)
            if size:
                protected.append((raw, raw + size))
        if section["relocations"]:
            raw = u32(coff.data, header + 24)
            protected.append((raw, raw + len(section["relocations"]) * 10))
    _require(not any(table < end and start < table_end for start, end in protected),
             "Symbol table overlaps headers, raw section data or relocations")
    functions = []
    for name in sorted(names):
        matches = [(i, s) for i, s in coff.symbols.items() if s["name"] == name]
        _require(len(matches) == 1, f"Expected one unambiguous primary symbol: {name}")
        index, symbol = matches[0]
        section_number = symbol["section"]
        _require(symbol["storage"] == 3 and is_function_type(symbol["type"])
                 and symbol["aux_count"] == 0 and symbol["value"] == 0
                 and 1 <= section_number <= len(coff.sections),
                 f"Expected a static typed function at section offset zero: {name}")
        section = coff.sections[section_number - 1]
        _require(section["data"] and section["flags"] & 0x20001020 == 0x20001020
                 and not section["flags"] & 0xC0,
                 f"Expected an executable code COMDAT: {name}")
        members = [(i, s) for i, s in coff.symbols.items() if s["section"] == section_number]
        typed = [i for i, s in members if is_function_type(s["type"])]
        _require(len(members) >= 2 and members[1][0] == index and typed == [index]
                 and coff.auxiliary_records[members[0][0]][14] != 5,
                 f"Function must be the sole typed independent COMDAT leader: {name}")
        offset = table + index * 18 + 16
        _require(coff.data[offset] == 3, "Unexpected native storage-class byte")
        functions.append({"name": name, "symbol_index": index, "storage_offset": offset,
                          "old_storage": 3, "new_storage": 2, "section": section_number,
                          "type": symbol["type"], "value": symbol["value"]})
    output = bytearray(coff.data)
    for function in functions:
        output[function["storage_offset"]] = 2
    output = bytes(output)
    changed = {i for i, (before, after) in enumerate(zip(coff.data, output)) if before != after}
    _require(len(output) == len(coff.data)
             and changed == {f["storage_offset"] for f in functions},
             "Unexpected object-byte changes")
    result = COFF.from_bytes(output)
    expected_symbols = {i: dict(symbol) for i, symbol in coff.symbols.items()}
    for function in functions:
        expected_symbols[function["symbol_index"]]["storage"] = 2
    _require(result.sections == coff.sections and result.auxiliary_records == coff.auxiliary_records
             and result.symbols == expected_symbols, "Unexpected reparsed object changes")
    compare_coff_extension(result, result)
    return output, {
        "operation": "coff-function-visibility",
        "input_sha256": hashlib.sha256(coff.data).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "changed_bytes": len(changed),
        "raw_sections_unchanged": True,
        "relocations_unchanged": True,
        "functions": functions,
    }
