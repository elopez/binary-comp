from __future__ import annotations

import pytest

from binary_comp.analyzers.byte_compare import compare_function_bytes
from binary_comp.analyzers.report import (
    SimilarityReportOptions,
    format_similarity_report,
    generate_similarity_report,
)
from binary_comp.config import BuildConfig, ProjectTarget
from binary_comp.core.pe import PEImage

from conftest import TEXT_VA, write_tiny_pe


pytest.importorskip("capstone")

from binary_comp.analyzers.function_compare import disassemble_function  # noqa: E402


MOV_CMP_RET = b"\xB8\x07\x00\x00\x00\x83\xF8\x07\xC3"     # mov eax,7; cmp eax,7; ret
MOV_ECX_RET = b"\xB9\x07\x00\x00\x00\x83\xF8\x07\xC3"     # mov ecx,7; cmp eax,7; ret
XOR_RET = b"\x33\xC0\xC3"                                 # xor eax,eax; ret


def load(tmp_path, name, code):
    path = tmp_path / name
    write_tiny_pe(path, code)
    image = PEImage(str(path))
    result = disassemble_function(image, TEXT_VA, [TEXT_VA, TEXT_VA + 0x10], 0x100, frozenset({"nop"}))
    return image, result.instructions


def test_identical_functions_compare_exact(tmp_path):
    original = load(tmp_path, "original.exe", MOV_CMP_RET)
    rebuilt = load(tmp_path, "rebuilt.exe", MOV_CMP_RET)

    comparison = compare_function_bytes(*original, *rebuilt, TEXT_VA + 0x10, TEXT_VA + 0x10)

    assert comparison.exact
    assert comparison.label == "EXACT"
    assert comparison.original_size == 0x10


def test_a_different_register_is_a_byte_difference(tmp_path):
    original = load(tmp_path, "original.exe", MOV_CMP_RET)
    rebuilt = load(tmp_path, "rebuilt.exe", MOV_ECX_RET)

    comparison = compare_function_bytes(*original, *rebuilt, TEXT_VA + 0x10, TEXT_VA + 0x10)

    assert not comparison.exact
    assert comparison.same_size
    assert comparison.label == "BYTES 1/3"
    assert comparison.first_difference == 0


def test_a_shorter_function_reports_its_size_delta(tmp_path):
    original = load(tmp_path, "original.exe", MOV_CMP_RET)
    rebuilt = load(tmp_path, "rebuilt.exe", XOR_RET)

    comparison = compare_function_bytes(*original, *rebuilt)

    assert comparison.label == "SIZE -6"


def test_report_adds_byte_status_and_summary(fixture_root, sample_binaries):
    original, rebuilt = sample_binaries
    target = ProjectTarget(
        name="full",
        original_exe=str(original),
        rebuilt_exe=str(rebuilt),
        map_path=str(fixture_root / "rebuilt.map"),
        source_dirs=(str(fixture_root / "src"),),
        code_dir=str(fixture_root / "code"),
        build=BuildConfig(),
    )

    report = generate_similarity_report(target, SimilarityReportOptions(build=False, compare_bytes=True))
    text = format_similarity_report(report)

    assert [row.bytes.label for row in report.rows] == ["EXACT"]
    assert "Byte comparison: 1" in text
    assert "  Exact: 1" in text


def test_pe_image_reads_highlow_base_relocations(tmp_path):
    import struct

    path = tmp_path / "relocated.exe"
    block = struct.pack("<IIHH", 0x1000, 12, 0x3004, 0x0000)
    write_tiny_pe(path, MOV_CMP_RET, {0x14: block})
    data = bytearray(path.read_bytes())
    optional_header = 0x80 + 24
    struct.pack_into("<II", data, optional_header + 96 + 5 * 8, 0x2014, len(block))
    path.write_bytes(bytes(data))

    assert PEImage(str(path)).relocations() == frozenset({TEXT_VA + 4})
