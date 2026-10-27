"""Explicit pointer-table alignment uses synthetic complete COFF contributions."""
import hashlib
import json
import struct

import pytest

from binary_comp.cli import main
from binary_comp.core.coff import (
    COFF, CodeRegions, apply_fixups, code_window, compare_function, validate_data_binding,
)
from binary_comp.core.pe import FileBackedPEImage
from conftest import DATA_VA, TEXT_VA
from test_coff_contract import DATA_FLAGS, coff_bytes, entry, reference
from test_coff_regions import make_shared


def switch_fixture(tmp_path, *, alignment=1, gap=b"", tail=b"\x90" * 3):
    # Two returning cases end at byte 21. The table is native data immediately
    # after the final RET, with no instruction or relocation bytes omitted.
    code = bytes.fromhex("8b 44 24 04 ff 24 85 00 00 00 00 31 c0 40 c3 b8 01 00 00 00 c3")
    start = len(code) + len(gap)
    raw = code + gap + bytes(8) + tail
    symbols = [{"name": "dispatch", "section": 1, "type": 0x20},
               {"name": "$case0", "section": 1, "value": 11, "storage": 6},
               {"name": "$case1", "section": 1, "value": 15, "storage": 6},
               {"name": "$table", "section": 1, "value": start, "storage": 6}]
    relocations = [(7, 3, 6), (start, 1, 6), (start + 4, 2, 6)]
    coff = COFF.from_bytes(coff_bytes([{"data": raw, "relocations": relocations}], symbols))
    expected = bytearray(raw)
    for offset, index, _ in relocations:
        struct.pack_into("<I", expected, offset, TEXT_VA + symbols[index]["value"])
    bindings = {s["name"]: {"va": hex(TEXT_VA + s.get("value", 0))} for s in symbols}
    bindings["dispatch"]["size"] = len(code)
    bindings["$table"].update(data_size=8, inline_data=True, inline_alignment=alignment)
    return coff, entry("dispatch", TEXT_VA, len(code)), bindings, reference(tmp_path, expected)


def check(fixture):
    coff, function, bindings, image = fixture
    return compare_function(coff, function, "sample", bindings, image)


@pytest.mark.parametrize("alignment,gap", [(1, b""), (2, b"\x90"), (4, b"\x90" * 3)])
def test_explicit_alignment_preserves_complete_code_table_and_padding(tmp_path, alignment, gap):
    coff, function, bindings, image = fixture = switch_fixture(
        tmp_path, alignment=alignment, gap=gap, tail=b"")
    exact, rebuilt, original, fixups = check(fixture)
    assert exact and rebuilt == original and len(rebuilt) == 21 and rebuilt[-1] == 0xC3
    assert fixups == 1
    table = coff.inline_table("$table", 8, alignment=alignment)
    assert len(table["data"]) == 8 and len(table["relocations"]) == 2
    assert validate_data_binding(coff, "$table", bindings, image) == (
        struct.pack("<II", TEXT_VA + 11, TEXT_VA + 15), 2)
    window = code_window(coff, coff.function("dispatch"), TEXT_VA, 21, bindings, image)
    assert window["inline_data_bytes"] == 8 and window["excluded_nop_padding"] == len(gap)


def test_odd_pointer_offset_requires_explicit_alignment_and_keeps_default_four(tmp_path):
    coff, _, bindings, _ = fixture = switch_fixture(tmp_path)
    with pytest.raises(ValueError, match="contiguous DIR32"):
        coff.inline_table("$table", 8)
    del bindings["$table"]["inline_alignment"]
    with pytest.raises(ValueError, match="contiguous DIR32"):
        check(fixture)
    aligned = switch_fixture(tmp_path, alignment=4, gap=b"\x90" * 3, tail=b"")
    del aligned[2]["$table"]["inline_alignment"]
    assert check(aligned)[0]


@pytest.mark.parametrize("alignment", [None, False, True, 0, -1, 3, 8, 1.0, "1", [], {}])
def test_core_rejects_malformed_alignment_even_on_unreferenced_bindings(tmp_path, alignment):
    coff, function, bindings, image = switch_fixture(tmp_path)
    with pytest.raises(ValueError, match="inline_alignment"):
        coff.inline_table("$table", 8, alignment=alignment)
    bindings["$table"]["inline_alignment"] = alignment
    with pytest.raises(ValueError, match="inline_alignment"):
        compare_function(coff, function, "sample", bindings, image)
    with pytest.raises(ValueError, match="inline_alignment"):
        validate_data_binding(coff, "$table", bindings, image)
    with pytest.raises(ValueError, match="inline_alignment"):
        apply_fixups(coff, b"", [], TEXT_VA, bindings, image)


@pytest.mark.parametrize("mutation", ["bare", "false", "no_size", "hex", "indices", "executable", "false_executable"])
def test_alignment_cannot_be_reused_as_another_binding_role(tmp_path, mutation):
    coff, function, bindings, image = switch_fixture(tmp_path)
    table = bindings["$table"]
    if mutation == "bare":
        del table["inline_data"]
    elif mutation == "false":
        table["inline_data"] = False
    elif mutation == "no_size":
        del table["data_size"]
    elif mutation == "hex":
        table["data_hex"] = "00" * 8
    elif mutation == "indices":
        table["inline_index_for"] = "$other"
    elif mutation == "executable":
        table["executable_data"] = True
    else:
        table["executable_data"] = False
    with pytest.raises(ValueError, match="inline_alignment"):
        compare_function(coff, function, "sample", bindings, image)
    with pytest.raises(ValueError, match="inline_alignment"):
        validate_data_binding(coff, "$table", bindings, image)


def test_alignment_does_not_convert_common_or_ordinary_storage_into_inline_data(tmp_path):
    for symbol, section in [({"name": "buffer", "value": 8}, None),
                            ({"name": "buffer", "section": 1},
                             {"name": ".rdata", "data": bytes(8), "flags": DATA_FLAGS})]:
        coff = COFF.from_bytes(coff_bytes([] if section is None else [section], [symbol]))
        bindings = {"buffer": {"va": hex(DATA_VA), "data_size": 8,
                               "inline_data": True, "inline_alignment": 1}}
        with pytest.raises(ValueError, match="COMMON data|code-section data label"):
            validate_data_binding(coff, "buffer", bindings, reference(tmp_path, b"\xc3", bytes(8)))


@pytest.mark.parametrize("mutation", [
    "partial", "oversize", "missing_fixup", "duplicate_fixup", "crossing_fixup", "wrong_kind",
    "addend", "wrong_case", "missing_case", "nonlocal_case", "typed_case", "alias_owner",
    "nonlocal_table", "ordinary_section", "tail_bytes", "tail_fixup", "excess_tail", "unowned_suffix",
])
def test_unaligned_table_keeps_full_payload_fixup_ownership_and_suffix_guards(tmp_path, mutation):
    coff, _, bindings, _ = fixture = switch_fixture(tmp_path)
    section = coff.sections[0]
    raw = bytearray(section["data"])
    if mutation == "partial":
        bindings["$table"]["data_size"] = 4
    elif mutation == "oversize":
        bindings["$table"]["data_size"] = 12
    elif mutation == "missing_fixup":
        section["relocations"].pop()
    elif mutation == "duplicate_fixup":
        section["relocations"].append(section["relocations"][-1])
    elif mutation == "crossing_fixup":
        section["relocations"].append((20, 1, 6))
    elif mutation == "wrong_kind":
        section["relocations"][-1] = (25, 2, 0x14)
    elif mutation == "addend":
        raw[21] = 1
    elif mutation == "wrong_case":
        bindings["$case0"]["va"] = hex(TEXT_VA + 12)
    elif mutation == "missing_case":
        del bindings["$case0"]
    elif mutation == "nonlocal_case":
        coff.named["$case0"]["storage"] = 2
    elif mutation == "typed_case":
        coff.named["$case0"]["type"] = 0x20
    elif mutation == "alias_owner":
        coff.symbols[4] = {"name": "alias", "section": 1, "value": 0, "type": 0x20, "storage": 2}
    elif mutation == "nonlocal_table":
        coff.named["$table"]["storage"] = 2
    elif mutation == "ordinary_section":
        section["flags"] = DATA_FLAGS
    elif mutation == "tail_bytes":
        raw[29:] = b"\x8d\x49\0"  # Recognized padding must still equal the original.
    elif mutation == "tail_fixup":
        raw += b"\x90"
        section["relocations"].append((29, 1, 6))
    elif mutation == "excess_tail":
        raw += b"\x90" * 16
    else:
        coff.symbols[4] = coff.named["$after"] = {
            "name": "$after", "section": 1, "value": 29, "type": 0, "storage": 6}
    section["data"] = bytes(raw)
    with pytest.raises(ValueError):
        check(fixture)


def test_explicit_alignment_never_skips_a_boundary_or_changed_pre_table_byte(tmp_path):
    fixture = switch_fixture(tmp_path, alignment=1, gap=b"\x90", tail=b"")
    with pytest.raises(ValueError, match="incompatible with original code extent"):
        check(fixture)
    fixture = switch_fixture(tmp_path, alignment=2, gap=b"\x90", tail=b"")
    fixture[0].sections[0]["data"] = fixture[0].sections[0]["data"][:21] + b"\xcc" + fixture[0].sections[0]["data"][22:]
    with pytest.raises(ValueError, match="Non-NOP"):
        check(fixture)


def test_alignment_is_not_a_byte_index_table_option(tmp_path):
    coff, _, _, _ = switch_fixture(tmp_path)
    with pytest.raises(ValueError, match="native pointer table"):
        coff.inline_table("$table", 8, index_entries=2, alignment=1)


def test_region_context_rejects_stray_alignment_and_cannot_erase_an_owner_role(tmp_path):
    coff, region, bindings, image = make_shared(tmp_path)
    bindings["first"]["inline_alignment"] = 1
    with pytest.raises(ValueError, match="inline_alignment"):
        CodeRegions(coff, [region], "retail", bindings, image)
    del bindings["first"]["inline_alignment"]
    context = CodeRegions(coff, [region], "retail", bindings, image)
    context.bindings["caller"] = {"va": hex(TEXT_VA + 0x80), "inline_alignment": 1}
    updated = dict(context.bindings)
    updated["caller"] = {"va": hex(TEXT_VA + 0x80), "size": 6}
    with pytest.raises(ValueError, match="address or role"):
        context.for_code_window(updated, "caller", TEXT_VA + 0x80, 6)


def test_unaligned_table_requires_the_complete_original_file_backed_range(tmp_path):
    coff, function, bindings, image = switch_fixture(tmp_path)
    raw = bytearray(image.data)
    pe = struct.unpack_from("<I", raw, 0x3C)[0]
    sections = pe + 24 + struct.unpack_from("<H", raw, pe + 20)[0]
    struct.pack_into("<I", raw, sections + 16, 25)  # Only the first table pointer is file-backed.
    path = tmp_path / "truncated.exe"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        compare_function(coff, function, "sample", bindings, FileBackedPEImage(path))


@pytest.fixture
def project(tmp_path):
    coff, function, bindings, image = switch_fixture(tmp_path)
    (tmp_path / "Synthetic.obj").write_bytes(coff.data)
    modules = [{"name": "Synthetic", "functions": [function], "bindings": {"sample": bindings}}]
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps(modules))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"targets": {"sample": {
        "kind": "i386-coff", "original_exe": str(image.path), "object_dir": str(tmp_path),
        "original_sha256": hashlib.sha256(image.data).hexdigest()}},
        "coff": {"modules": "modules.json"}}))
    return config, manifest, tmp_path / "report", modules


def args(project, *, score=False, filtered=False):
    config, _, output, _ = project
    command = ["compare", "dispatch"] if score or filtered else ["report"]
    command += ["--config", str(config), "--target", "sample", "--no-build", "--module", "Synthetic"]
    if score:
        command += ["--score", "--object", str(config.parent / "Synthetic.obj")]
    else:
        command += ["--output", str(output)]
    return command


@pytest.mark.parametrize("filtered", [False, True])
def test_cli_report_and_filtered_comparison_count_only_complete_code_and_data(project, capsys, filtered):
    pytest.importorskip("capstone")
    assert main(args(project, filtered=filtered)) == 0
    result = json.loads((project[2] / "report.json").read_text())
    row, = result["code"]
    assert row["body_size"] == 21 and row["decoded_bytes"] == {"original": 21, "rebuilt": 21}
    assert row["relocated_bytes"]["exact"] and not row["raw_bytes"]["exact"]
    assert row["checked_fixups"] == 1 and row["inline_data_bytes"] == 8
    assert row["inline_alignment_bytes"] == 3
    data, = result["data"]
    assert data["exact"] and data["size"] == 8 and data["checked_fixups"] == 2
    assert main(args(project, score=True)) == 0
    assert capsys.readouterr().out.endswith("0\n")


@pytest.mark.parametrize("mutation", ["default", "bad_alignment", "bad_role", "bad_table", "bad_code", "short_code", "overlap"])
@pytest.mark.parametrize("filtered", [False, True])
def test_cli_rechecks_alignment_and_never_retains_stale_exact_results(project, capsys, mutation, filtered):
    pytest.importorskip("capstone")
    assert main(args(project, filtered=filtered)) == 0
    capsys.readouterr()
    _, manifest, output, modules = project
    table = modules[0]["bindings"]["sample"]["$table"]
    if mutation == "default":
        del table["inline_alignment"]
    elif mutation == "bad_alignment":
        table["inline_alignment"] = True
    elif mutation == "bad_role":
        modules[0]["bindings"]["sample"]["$case0"]["inline_alignment"] = 1
    elif mutation in ("short_code", "overlap"):
        size = 20 if mutation == "short_code" else 22
        modules[0]["functions"][0]["sample"]["size"] = size
        modules[0]["bindings"]["sample"]["dispatch"]["size"] = size
    else:
        path = project[0].parent / "Synthetic.obj"
        raw = bytearray(path.read_bytes())
        raw[60 + (21 if mutation == "bad_table" else 13)] ^= 1
        path.write_bytes(raw)
    manifest.write_text(json.dumps(modules))
    assert main(args(project, filtered=filtered)) > 0
    if mutation in ("bad_alignment", "bad_role"):
        assert not (output / "report.json").exists() and not (output / "report.txt").exists()
    else:
        result = json.loads((output / "report.json").read_text())
        assert not result["code"][0].get("relocated_bytes", {}).get("exact", False)
        if mutation in ("default", "bad_table"):
            assert not result["data"][0]["exact"]
    capsys.readouterr()
    expected_status = 0 if mutation in ("bad_code", "short_code") else 2
    assert main(args(project, score=True)) == expected_status
    captured = capsys.readouterr()
    assert captured.out == ("1\n" if expected_status == 0 else "")
    assert bool(captured.err) == (expected_status == 2)
