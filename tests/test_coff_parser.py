"""Distributable COFF metadata fixtures; no compiler or external binary is needed."""
from __future__ import annotations

import struct

import pytest

from binary_comp.core.coff import (
    COFF, apply_fixups, code_window, compare_function, is_function_type, validate_data_binding,
)


def object_bytes(name: str = "_entry", *, aux: bytes = b"", storage: int = 2, callee: bool = False, kind: int = 0x20) -> bytes:
    text = b"\xc3" + b"\x90" * 15
    raw_pointer = 60
    symbol_pointer = raw_pointer + len(text)
    strings = b""
    if len(name) > 8:
        encoded_name = struct.pack("<II", 0, 4)
        strings = name.encode("ascii") + b"\0"
    else:
        encoded_name = name.encode("ascii").ljust(8, b"\0")
    symbol = encoded_name + struct.pack("<IhHBB", 0, 1, kind, storage, len(aux) // 18)
    if callee:
        symbol += b"_callee\0" + struct.pack("<IhHBB", 0, 0, 0, 2, 0)
    header = struct.pack("<HHIIIHH", 0x14C, 1, 0, symbol_pointer, 1 + len(aux) // 18 + callee, 0, 0)
    section = struct.pack("<8sIIIIIIHHI", b".text", 0, 0, len(text), raw_pointer,
                          0, 0, 0, 0, 0x60001020)
    return header + section + text + symbol + aux + struct.pack("<I", 4 + len(strings)) + strings


def test_memory_and_path_parsing_preserve_the_same_native_symbol_indices(tmp_path):
    data = object_bytes("_long_function_name")
    path = tmp_path / "sample.obj"
    path.write_bytes(data)
    memory = COFF.from_bytes(memoryview(data))
    disk = COFF(path)
    assert memory.data == disk.data == data
    assert memory.sections == disk.sections
    assert memory.symbols == disk.symbols
    assert memory.named == disk.named
    assert memory.resolve_symbol(0)["name"] == "_long_function_name"
    assert memory.function("_long_function_name")["data"] == b"\xc3" + b"\x90" * 15


def test_cvtres_zero_length_empty_string_table_preserves_short_names(tmp_path):
    canonical = COFF.from_bytes(object_bytes())
    data = object_bytes()[:-4] + bytes(4)
    path = tmp_path / "empty-strings.obj"
    path.write_bytes(data)
    for coff in (COFF.from_bytes(data), COFF(path)):
        assert coff.sections == canonical.sections
        assert coff.symbols == canonical.symbols
        assert coff.named == canonical.named


@pytest.mark.parametrize("missing", [1, 2, 3, 4])
def test_empty_string_table_still_requires_its_complete_size_field(missing):
    data = object_bytes()[:-4] + bytes(4)
    with pytest.raises(ValueError, match="Truncated COFF string table"):
        COFF.from_bytes(data[:-missing])


@pytest.mark.parametrize("size", [1, 2, 3, 5, 0xFFFFFFFF])
def test_invalid_nonzero_string_table_lengths_are_not_treated_as_empty(size):
    data = object_bytes()[:-4] + struct.pack("<I", size)
    with pytest.raises(ValueError, match="Truncated COFF string table"):
        COFF.from_bytes(data)


@pytest.mark.parametrize("offset", [0, 3, 4, 5, 0xFFFFFFFF])
def test_zero_length_table_cannot_resolve_long_names_from_undeclared_bytes(offset):
    data = bytearray(object_bytes("_long_function_name"))
    symbol_table, count = struct.unpack_from("<II", data, 8)
    struct.pack_into("<I", data, symbol_table + 4, offset)
    struct.pack_into("<I", data, symbol_table + count * 18, 0)
    with pytest.raises(ValueError, match="Bad COFF string offset"):
        COFF.from_bytes(data)


@pytest.mark.parametrize("offset, value, label", [
    (8, 0x100000, "symbol table"),
    (12, 0x100000, "symbol table"),
    (40, 0x100000, "section or missing raw data"),
])
def test_zero_length_table_keeps_symbol_and_section_ranges_bounded(offset, value, label):
    data = bytearray(object_bytes()[:-4] + bytes(4))
    struct.pack_into("<I", data, offset, value)
    with pytest.raises(ValueError, match=f"Truncated COFF {label}"):
        COFF.from_bytes(data)


@pytest.mark.parametrize("size", [0, 19, 59, 93, 97])
def test_truncated_headers_sections_symbols_and_strings_raise_value_error(size):
    with pytest.raises(ValueError, match="Truncated COFF"):
        COFF.from_bytes(object_bytes()[:size])


@pytest.mark.parametrize("offset, value", [(8, 0x100000), (12, 0x100000), (94, 0x100000)])
def test_metadata_offsets_and_counts_cannot_read_outside_the_object(offset, value):
    data = bytearray(object_bytes())
    struct.pack_into("<I", data, offset, value)
    with pytest.raises(ValueError, match="Truncated COFF"):
        COFF.from_bytes(data)


@pytest.mark.parametrize("offset, value", [(0, 0x8664), (16, 0xE0)])
def test_unsupported_machine_and_optional_headers_are_explicit(offset, value):
    data = bytearray(object_bytes())
    struct.pack_into("<H", data, offset, value)
    with pytest.raises(ValueError, match="i386 COFF object without optional header"):
        COFF.from_bytes(data)


def test_section_table_and_raw_data_bounds_are_checked():
    for offset, value in ((2, 0x100), (40, 0x100000)):
        data = bytearray(object_bytes())
        struct.pack_into("<H" if offset == 2 else "<I", data, offset, value)
        with pytest.raises(ValueError, match="Truncated COFF"):
            COFF.from_bytes(data)


def test_relocations_need_a_real_complete_file_range():
    for pointer in (0, 0x100000):
        data = bytearray(object_bytes())
        struct.pack_into("<I", data, 44, pointer)
        struct.pack_into("<H", data, 52, 1)
        with pytest.raises(ValueError, match="Truncated COFF relocations"):
            COFF.from_bytes(data)


def test_invalid_long_name_offsets_and_unterminated_names_fail():
    data = bytearray(object_bytes("_long_function_name"))
    struct.pack_into("<I", data, 80, 0x100000)
    with pytest.raises(ValueError, match="Bad COFF string offset"):
        COFF.from_bytes(data)
    data = bytearray(object_bytes("_long_function_name"))
    data[-1] = ord("x")
    with pytest.raises(ValueError):
        COFF.from_bytes(data)


def test_auxiliary_records_must_fit_and_weak_search_modes_are_checked():
    data = bytearray(object_bytes())
    data[93] = 1
    with pytest.raises(ValueError, match="Truncated COFF auxiliary symbols"):
        COFF.from_bytes(data)
    with pytest.raises(ValueError, match="Malformed weak external"):
        COFF.from_bytes(object_bytes(storage=105))
    auxiliary = struct.pack("<II", 0, 99) + b"\0" * 10
    with pytest.raises(ValueError, match="Unsupported weak external search mode"):
        COFF.from_bytes(object_bytes(aux=auxiliary, storage=105))


@pytest.mark.parametrize("delta", [-(1 << 32), 1 << 32])
@pytest.mark.parametrize("kind", [6, 0x14])
def test_explicit_binding_cannot_alias_a_valid_va_by_integer_wraparound(delta, kind):
    coff = COFF.from_bytes(object_bytes(callee=True))
    bindings = {"_callee": {"va": hex(0x401080 + delta)}}
    with pytest.raises(ValueError, match="Binding outside i386 address domain"):
        apply_fixups(coff, bytes(4), [(0, 1, kind)], 0x401000, bindings, None)


@pytest.mark.parametrize("origin", [-1, -0x100000000, 0x100000000, 0x100401000])
def test_origins_outside_i386_domain_fail_even_without_relocations(origin):
    coff = COFF.from_bytes(object_bytes())
    with pytest.raises(ValueError, match="Origin outside i386 address domain"):
        apply_fixups(coff, b"\xc3", [], origin, {}, None)
    with pytest.raises(ValueError, match="Origin outside i386 address domain"):
        code_window(coff, coff.function("_entry"), origin, 1, {}, None)
    function = {"symbol": "_entry", "sample": {"va": hex(origin), "size": 1}}
    with pytest.raises(ValueError, match="Binding outside i386 address domain"):
        compare_function(coff, function, "sample", {}, None)


@pytest.mark.parametrize("kind, origin, address, addend, expected", [
    (6, 0, 0xFFFFFFFF, 1, 0),
    (6, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF),
    (0x14, 0xFFFFFFFC, 4, 0xFFFFFFFF, 3),
    (0x14, 0, 0, 0xFFFFFFFC, 0xFFFFFFF8),
])
def test_native_addends_keep_modular_dir32_and_rel32_arithmetic(kind, origin, address, addend, expected):
    coff = COFF.from_bytes(object_bytes(callee=True))
    bindings = {"_callee": {"va": hex(address)}}
    actual, fixups = apply_fixups(coff, struct.pack("<I", addend), [(0, 1, kind)],
                                  origin, bindings, None)
    assert actual == struct.pack("<I", expected)
    assert fixups == 1


@pytest.mark.parametrize("address", ["-0x100000000", "0x100000000"])
def test_data_bindings_validate_address_domain_before_any_reference_read(address):
    coff = COFF.from_bytes(object_bytes())
    with pytest.raises(ValueError, match="Binding outside i386 address domain"):
        validate_data_binding(coff, "_entry", {"_entry": {"va": address, "data_hex": "c3"}}, None)


@pytest.mark.parametrize("kind, expected", [(0x20, True), (0x24, True), (0x120, True),
                                            (0, False), (4, False), (0x30, False), (0x34, False)])
def test_function_type_checks_only_the_native_derived_function_field(kind, expected):
    assert is_function_type(kind) is expected


def test_function_return_base_type_int_is_accepted():
    coff = COFF.from_bytes(object_bytes(kind=0x24))
    assert coff.function("_entry") is coff.sections[0]


@pytest.mark.parametrize("storage", [2, 6])
def test_native_array_type_is_not_a_function_or_compiler_code_label(storage):
    coff = COFF.from_bytes(object_bytes(kind=0x30, storage=storage))
    with pytest.raises(ValueError, match="Not a function or compiler code label"):
        coff.function("_entry")


def test_untyped_native_compiler_code_labels_still_have_bounded_entries():
    coff = COFF.from_bytes(object_bytes(kind=0, storage=6))
    assert coff.function("_entry")["data"] == b"\xc3" + b"\x90" * 15
