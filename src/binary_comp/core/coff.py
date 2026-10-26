"""Strict i386 COFF objects, explicit relocations and complete code/data windows.

The parser retains native symbol indices, weak fallbacks and local labels. The
comparison helpers accept explicit symbol/address bindings and a reference
object with ``read(va, size)``; they never normalize arbitrary operands. No
compiler, linker, project layout or build command is assumed here.
"""
from pathlib import Path
import struct

from .binary import compare_binary


def u16(data, offset):
    return struct.unpack_from("<H", data, offset)[0]


def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def is_function_type(value):
    """Test the first COFF derived-type field, preserving any base return type."""
    return (value & 0x30) == 0x20


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
                      "type": kind, "storage": storage}
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
                # MSVC can emit an unresolved reference followed by its COMDAT
                # definition. Preserve both indices; resolve names to the definition.
                if (not previous or section > 0 or
                        (previous["section"] == 0 and storage == 105)):
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
        """Bound a complete definition by the next symbol or its section end."""
        symbol, section = self.definition(name)
        start = symbol["value"]
        end = min((s["value"] for s in self.symbols.values()
                   if s["section"] == symbol["section"] and s["value"] > start),
                  default=len(section["data"]))
        return end - start

    def definition(self, name):
        symbol = self.named[name]
        if not 1 <= symbol["section"] <= len(self.sections):
            raise ValueError(f"Undefined symbol {name}")
        return symbol, self.sections[symbol["section"] - 1]

    def _section_functions(self, section):
        return [s for s in self.symbols.values()
                if 1 <= s["section"] <= len(self.sections)
                and self.sections[s["section"] - 1] is section
                and is_function_type(s["type"])]

    def inline_table(self, name, size, *, index_entries=None):
        """Bound native local case pointers or byte indices in a /Gy code section."""
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
            if (start % 4 or size % 4
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
    if len(functions) <= 1:
        return
    if any(s["value"] != 0 for s in functions):
        raise ValueError("Comparison requires all /Gy function entries at section offset zero")
    for symbol in functions:
        name = symbol["name"]
        if name in bindings and _binding_address(bindings[name]["va"], name) != origin:
            raise ValueError(f"Co-located function alias address differs from origin: {name}")


def apply_fixups(coff, raw, relocations, origin, bindings, reference, visiting=None):
    _checked_origin(origin)
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
        address = _binding_address(binding["va"], name)
        if symbol["section"] != 0 and not 1 <= symbol["section"] <= len(coff.sections):
            raise ValueError(f"Unsupported relocation symbol section: {name}")
        if symbol["section"] > 0 and is_function_type(symbol["type"]):
            _validate_function_alias_bindings(
                coff, coff.sections[symbol["section"] - 1], address, bindings)
        if "data_hex" in binding or "data_size" in binding:
            validate_data_binding(coff, name, bindings, reference, visiting)
        elif (symbol["section"] != 0 and
              not coff.sections[symbol["section"] - 1]["flags"] & 0x20):
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


def validate_data_binding(coff, name, bindings, reference, visiting=None):
    """Compare a complete data definition, including all checked pointer fixups."""
    visiting = set() if visiting is None else visiting
    if name in visiting:
        raise ValueError(f"Cyclic data binding: {name}")
    binding = bindings[name]
    address = _binding_address(binding["va"], name)
    if "data_hex" in binding:
        if "data_size" in binding:
            raise ValueError(f"Ambiguous data size for {name}")
        expected = bytes.fromhex(binding["data_hex"])
        if not expected or reference.read(address, len(expected)) != expected:
            raise ValueError(f"Original constant differs for {name}")
        size = len(expected)
    else:
        size = binding["data_size"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid data size for {name}")
        expected = reference.read(address, size)
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
            validate_data_binding(coff, binding["inline_index_for"], bindings, reference, visiting | {name})
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
    visiting.add(name)
    try:
        actual, fixups = apply_fixups(coff, raw, relocations, address, bindings, reference, visiting)
    finally:
        visiting.remove(name)
    if not compare_binary(expected, actual).exact:
        raise ValueError(f"Rebuilt data differs from original: {name}")
    return actual, fixups


def inline_table(coff, name, bindings):
    """Require explicit native table roles; never infer data by decoding bytes."""
    binding = bindings[name]
    if (binding.get("inline_data") is not True or "data_size" not in binding
            or "data_hex" in binding):
        raise ValueError(f"Inline table requires inline_data=true and data_size: {name}")
    relation = binding.get("inline_index_for")
    if "inline_index_for" not in binding:
        return coff.inline_table(name, binding["data_size"])
    if (not isinstance(relation, str) or relation not in bindings
            or "inline_index_for" in bindings[relation]):
        raise ValueError(f"Invalid inline byte-index relation: {name}")
    pointers = inline_table(coff, relation, bindings)
    table = coff.inline_table(name, binding["data_size"], index_entries=len(pointers["data"]) // 4)
    if table["section"] is not pointers["section"] or pointers["end"] != table["start"]:
        raise ValueError(f"Byte-index table must follow its native pointer table: {name}")
    return table


def code_window(coff, section, origin, size, bindings, reference, *, diagnostic=False):
    """Bound code and native inline data once for both the gate and reviews."""
    _checked_origin(origin)
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
        mapping[tables[0][1]["owner"]["name"]] = {"va": hex(origin), "size": size}
        if not diagnostic:
            if start != (size + 3) // 4 * 4:
                raise BodyExtentError(f"Inline table offset {start} incompatible with original code extent {size}")
            gap = raw[size:start]
            if (any(byte != 0x90 for byte in gap)
                    or reference.read(origin + size, len(gap)) != gap):
                raise BodyExtentError("Non-NOP or changed alignment before inline table")
        for name, _ in tables:
            validate_data_binding(coff, name, mapping, reference)
        end = start if diagnostic else size
        relocations = [(offset, index, kind) for offset, index, kind in section["relocations"]
                       if offset < start]
        if any(offset < 0 or offset + 4 > end for offset, _, _ in relocations):
            raise ValueError("Relocation crosses code/inline alignment boundary")
        return {"data": raw[:end], "relocations": relocations, "bindings": mapping,
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


def relocate(coff, section, origin, size, bindings, reference):
    window = code_window(coff, section, origin, size, bindings, reference)
    return apply_fixups(coff, window["data"], window["relocations"], origin,
                        window["bindings"], reference)


def compare_function(coff, function, version, bindings, reference):
    target = function[version]
    address, size = _binding_address(target["va"], function["symbol"]), target["size"]
    section = coff.function(function["symbol"])
    actual, relocations = relocate(coff, section, address, size, bindings, reference)
    expected = reference.read(address, size)
    return compare_binary(expected, actual).exact, actual, expected, relocations
