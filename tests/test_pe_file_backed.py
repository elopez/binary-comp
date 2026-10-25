"""Strict reference reads and Rich provenance use synthetic executable fixtures."""
from __future__ import annotations

import struct

import pytest

from binary_comp.core.pe import FileBackedPEImage, PEImage, decode_rich_records
from conftest import DATA_VA, TEXT_VA, write_tiny_pe


def test_file_backed_reader_preserves_complete_raw_reads_and_linker_bytes(tmp_path):
    path = tmp_path / "sample.exe"
    write_tiny_pe(path)
    data = bytearray(path.read_bytes())
    data[0x9A:0x9C] = bytes((3, 4))
    path.write_bytes(data)
    image = FileBackedPEImage(path)
    assert image.linker == (3, 4)
    assert image.read(TEXT_VA, 1) == b"\xb8"
    assert image.read(DATA_VA, 6) == b"hello\0"
    for address, size in ((DATA_VA, -1), (TEXT_VA + 0x1FF, 2), (TEXT_VA - 1, 1)):
        with pytest.raises(ValueError):
            image.read(address, size)


def test_existing_loader_view_still_zero_fills_virtual_tail_while_strict_reads_fail(tmp_path):
    path = tmp_path / "sample.exe"
    write_tiny_pe(path)
    data = bytearray(path.read_bytes())
    section_table = 0x80 + 24 + 0xE0
    struct.pack_into("<I", data, section_table + 40 + 8, 0x40)
    path.write_bytes(data)
    assert PEImage(path).read(DATA_VA + 0x20, 4) == bytes(4)
    with pytest.raises(ValueError, match="no complete file-backed section"):
        FileBackedPEImage(path).read(DATA_VA + 0x20, 4)


def test_overlapping_loader_virtual_tail_cannot_replace_later_file_backed_bytes(tmp_path):
    path = tmp_path / "sample.exe"
    write_tiny_pe(path)
    data = bytearray(path.read_bytes())
    section_table = 0x80 + 24 + 0xE0
    struct.pack_into("<I", data, section_table + 8, 0x1400)
    path.write_bytes(data)
    # The legacy loader view deliberately chooses its first virtual section.
    assert PEImage(path).read(DATA_VA, 6) == bytes(6)
    # Strict comparison must read the raw section it actually validated.
    assert FileBackedPEImage(path).read(DATA_VA, 6) == b"hello\0"


@pytest.mark.parametrize("offset, value", [(0x84, 0x8664), (0x98, 0x20B)])
def test_strict_reader_rejects_other_machine_or_optional_header_formats(tmp_path, offset, value):
    path = tmp_path / "sample.exe"
    write_tiny_pe(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<H", data, offset, value)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Expected i386 PE32 image"):
        FileBackedPEImage(path)


def test_strict_reader_rejects_raw_sections_past_end_of_file(tmp_path):
    path = tmp_path / "sample.exe"
    write_tiny_pe(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, 0x80 + 24 + 0xE0 + 20, len(data))
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Truncated PE section"):
        FileBackedPEImage(path)


def test_truncated_pe_metadata_reports_value_error(tmp_path):
    path = tmp_path / "sample.exe"
    write_tiny_pe(path)
    path.write_bytes(path.read_bytes()[:0x180])
    with pytest.raises(ValueError):
        FileBackedPEImage(path)


def rich_fixture(tmp_path):
    path = tmp_path / "sample.exe"
    write_tiny_pe(path)
    data = bytearray(path.read_bytes())
    key = 0x12345678
    plain = b"DanS" + bytes(12) + struct.pack("<IIII", (3 << 16) | 101, 2, (7 << 16) | 202, 4)
    for offset in range(0, len(plain), 4):
        value = struct.unpack_from("<I", plain, offset)[0] ^ key
        struct.pack_into("<I", data, 0x40 + offset, value)
    end = 0x40 + len(plain)
    data[end:end + 8] = b"Rich" + struct.pack("<I", key)
    return path, data, end


def test_rich_records_are_generic_provenance_and_absence_is_distinct(tmp_path):
    path, data, _ = rich_fixture(tmp_path)
    path.write_bytes(data)
    expected = [{"product": 3, "build": 101, "count": 2}, {"product": 7, "build": 202, "count": 4}]
    assert decode_rich_records(data) == expected
    assert FileBackedPEImage(path).rich_records() == expected
    write_tiny_pe(path)
    assert FileBackedPEImage(path).rich_records() == []


@pytest.mark.parametrize("mutation", ["signature", "record_size", "unaligned", "missing_key"])
def test_malformed_rich_records_are_rejected(tmp_path, mutation):
    _, data, end = rich_fixture(tmp_path)
    if mutation == "signature":
        data[0x40:0x44] = bytes(4)
    elif mutation == "record_size":
        data[end - 4:end + 4] = data[end:end + 8]
    elif mutation == "unaligned":
        data[end + 1:end + 9] = data[end:end + 8]
        data[end] = 0
    else:
        struct.pack_into("<I", data, 0x3C, end + 4)
    with pytest.raises(ValueError, match="Malformed Rich header"):
        decode_rich_records(data)
