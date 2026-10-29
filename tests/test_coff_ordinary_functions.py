"""Ordinary multi-function sections, built entirely from synthetic records."""
import hashlib
import json
from pathlib import Path
import struct

import pytest

from binary_comp.cli import main
from binary_comp.core.coff import (
    BodyExtentError, COFF, CodeRegions, apply_fixups, code_window,
    compare_function, validate_data_binding,
)
from conftest import DATA_VA, TEXT_VA
from test_coff_contract import DATA_FLAGS, coff_bytes, entry, reference


def ordinary_bytes():
    # Two calls have already been resolved within the section: no native fixup
    # exists for them, so their displacements must survive comparison unchanged.
    first = b"\xe8\x0b\0\0\0\xc3" + b"\x90" * 10
    second = b"\xa1" + bytes(4) + b"\xe8" + bytes(4) + b"\xe8\xe1\xff\xff\xff\xc3"
    third = b"\x68" + bytes(4) + b"\x58\xc3" + b"\x90" * 9
    return coff_bytes([
        {"data": first + second + third,
         "relocations": [(17, 5, 6), (22, 7, 0x14), (33, 2, 6)]},
        {"name": ".data", "flags": DATA_FLAGS, "data": struct.pack("<II", 7, 0),
         "relocations": [(4, 2, 6)]},
    ], [
        {"name": "first", "section": 1, "type": 0x20},
        {"name": "second", "section": 1, "type": 0x24, "value": 16},
        {"name": "alias", "section": 1, "type": 0x20, "value": 16},
        {"name": "third", "section": 1, "type": 0x20, "value": 32},
        {"name": "case", "section": 1, "storage": 6, "value": 26},
        {"name": "number", "section": 2},
        {"name": "pointer", "section": 2, "value": 4},
        {"name": "external"},
    ])


@pytest.fixture
def ordinary(tmp_path):
    c = COFF.from_bytes(ordinary_bytes())
    original = bytearray(c.sections[0]["data"])
    struct.pack_into("<I", original, 17, DATA_VA)
    struct.pack_into("<i", original, 22, 0x80 - 26)
    struct.pack_into("<I", original, 33, TEXT_VA + 16)
    b = {name: {"va": hex(TEXT_VA + offset)} for name, offset in
         (("first", 0), ("second", 16), ("alias", 16), ("third", 32), ("case", 26))}
    b.update(number={"va": hex(DATA_VA), "data_size": 4},
             pointer={"va": hex(DATA_VA + 4), "data_size": 4},
             external={"va": hex(TEXT_VA + 0x80)})
    image = reference(tmp_path, original, struct.pack("<II", 7, TEXT_VA + 16))
    return c, b, image


def check(fixture, name="second", size=16):
    c, b, image = fixture
    return compare_function(c, entry(name, int(b[name]["va"], 16), size), "sample", b, image)


@pytest.mark.parametrize("name,size,fixups", [("first", 6, 0), ("second", 16, 2),
                                              ("alias", 16, 2), ("third", 7, 1)])
def test_typed_entries_bound_complete_functions_and_rebase_only_real_fixups(ordinary, name, size, fixups):
    c, _, _ = ordinary
    window = c.function(name)
    start = c.named[name]["value"]
    assert c.function_owner(window) == (c.sections[0], start, start + 16)
    assert window["data"] == c.sections[0]["data"][start:start + 16]
    exact, rebuilt, original, count = check(ordinary, name, size)
    assert exact and rebuilt == original and len(rebuilt) == size and count == fixups
    if name in ("second", "alias"):
        assert window["relocations"] == [(1, 5, 6), (6, 7, 0x14)]
        assert window["data"][10:15] == original[10:15]  # unrelocated CALL
    with pytest.raises(ValueError, match="not separate code entries"):
        c.function("case")


def test_static_typed_entry_with_auxiliary_record_is_a_real_boundary(tmp_path):
    c = COFF.from_bytes(coff_bytes([{"data": b"\xc3" + b"\x90" * 15 + b"\xc3"}], [
        {"name": "first", "section": 1, "type": 0x20, "storage": 3},
        {"name": "second", "section": 1, "type": 0x20, "storage": 3,
         "value": 16, "aux": bytes(18)},
        {"name": "label", "section": 1, "storage": 6, "value": 1},
    ]))
    assert c.function("first")["data"] == b"\xc3" + b"\x90" * 15
    assert c.function("second")["data"] == b"\xc3"
    assert compare_function(c, entry("second", TEXT_VA + 16, 1), "sample", {},
                            reference(tmp_path, c.sections[0]["data"]))[0]


@pytest.mark.parametrize("origin", [8, 0xfffffff0])
def test_complete_native_section_origin_cannot_underflow_or_overflow(ordinary, origin):
    c, b, image = ordinary
    with pytest.raises(ValueError, match="outside i386 address domain"):
        compare_function(c, entry("second", origin, 16), "sample", b, image)


@pytest.mark.parametrize("mutation", ["COMDAT", "prefix", "outside", "storage", "not executable",
                                      "not code", "straddle", "outside relocation", "overlap"])
def test_unsupported_ownership_and_cross_function_fixups_fail_closed(ordinary, mutation):
    c, _, _ = ordinary
    section = c.sections[0]
    if mutation == "COMDAT":
        section["flags"] |= 0x1000
    elif mutation == "prefix":
        c.named["first"]["value"] = 1
    elif mutation == "outside":
        c.named["third"]["value"] = 48
    elif mutation == "storage":
        c.named["third"]["storage"] = 6
    elif mutation == "not executable":
        section["flags"] &= ~0x20000000
    elif mutation == "not code":
        section["flags"] &= ~0x20
    else:
        section["relocations"].append(({"straddle": 13, "outside relocation": 47,
                                         "overlap": 18}[mutation], 5, 6))
    with pytest.raises(ValueError):
        check(ordinary)


@pytest.mark.parametrize("name", ["first", "second", "alias", "third", "case"])
def test_all_explicit_same_section_bindings_must_share_the_native_origin(ordinary, name):
    c, b, image = ordinary
    b[name]["va"] = hex(int(b[name]["va"], 16) + 1)
    with pytest.raises(ValueError, match="same-section binding differs"):
        compare_function(c, entry("second", TEXT_VA + 16, 16), "sample", b, image)
    with pytest.raises(ValueError, match="same-section binding differs"):
        validate_data_binding(c, "pointer", b, image)
    with pytest.raises(ValueError, match="same-section binding differs"):
        apply_fixups(c, bytes(4), [(0, 2, 6)], DATA_VA + 4, b, image)


@pytest.mark.parametrize("offset", [0, 1, 26, 27, 31])
def test_unrelocated_instruction_and_relative_layout_bytes_remain_authoritative(ordinary, offset):
    c, _, _ = ordinary
    raw = bytearray(c.sections[0]["data"])
    raw[offset] ^= 1
    c.sections[0]["data"] = bytes(raw)
    result = check(ordinary, "first", 6) if offset < 16 else check(ordinary)
    assert not result[0]


@pytest.mark.parametrize("size", [10, 15, 17, 32])
def test_extent_cannot_trim_instructions_or_cross_next_typed_entry(ordinary, size):
    with pytest.raises(BodyExtentError):
        check(ordinary, size=size)


def test_data_and_unselected_function_claims_remain_separate(ordinary):
    c, b, image = ordinary
    assert validate_data_binding(c, "pointer", b, image) == (struct.pack("<I", TEXT_VA + 16), 1)
    c.sections[0]["data"] = c.sections[0]["data"][:31] + b"\xcb" + c.sections[0]["data"][32:]
    assert check(ordinary, "first", 6)[0]
    assert not check(ordinary)[0]
    c.sections[1]["data"] = struct.pack("<II", 8, 0)
    with pytest.raises(ValueError, match="Rebuilt data differs"):
        check(ordinary)


@pytest.mark.parametrize("mutation", ["view", "bytes", "entry", "relocation", "flags", "foreign",
                                      "record", "name mapping"])
def test_bounded_views_never_reuse_stale_or_foreign_native_ownership(ordinary, mutation):
    c, b, image = ordinary
    view = c.function("second")
    assert code_window(c, view, TEXT_VA + 16, 16, b, image)["data"] == view["data"]
    if mutation == "view":
        view["data"] = bytes(16)
    elif mutation == "bytes":
        c.sections[0]["data"] = bytes(48)
    elif mutation == "entry":
        c.named["third"]["value"] = 33
    elif mutation == "relocation":
        c.sections[0]["relocations"][0] = (18, 5, 6)
    elif mutation == "flags":
        c.sections[0]["flags"] |= 0x1000
    elif mutation == "record":
        c.symbols[1] = dict(c.symbols[1])
    elif mutation == "name mapping":
        c.named["second"] = c.named["alias"]
    else:
        c = COFF.from_bytes(ordinary_bytes())
    with pytest.raises(ValueError):
        code_window(c, view, TEXT_VA + 16, 16, b, image)


def test_bounded_views_do_not_hide_complete_region_owner_identity(ordinary):
    c, b, image = ordinary
    region = {"symbol": "first", "label": "complete", "entries": [
        {"symbol": name, "offset": offset} for name, offset in
        (("first", 0), ("second", 16), ("alias", 16), ("third", 32))],
        "sample": {"va": hex(TEXT_VA), "size": 48}}
    ctx = CodeRegions(c, [region], "sample", b, image)
    with pytest.raises(ValueError, match="Duplicate COFF code contribution ownership"):
        compare_function(c, entry("second", TEXT_VA + 16, 16), "sample", b, image, code_regions=ctx)
    assert validate_data_binding(c, "pointer", b, image, code_regions=ctx)[1] == 1
    raw = bytearray(c.sections[0]["data"])
    raw[47] = 0xcc
    c.sections[0]["data"] = bytes(raw)
    with pytest.raises(ValueError, match="complete code region differs"):
        validate_data_binding(c, "pointer", b, image, code_regions=ctx)


@pytest.fixture
def ordinary_inline(tmp_path):
    code = b"\xc3" + b"\x90" * 15
    code += b"\xff\x24\x85" + bytes(4) + b"\xc3\x31\xc0\xc3\x90" + bytes(8) + b"\x90" * 12
    code += b"\xc3" + b"\x90" * 15
    c = COFF.from_bytes(coff_bytes([{"data": code, "relocations": [(19, 5, 6), (28, 3, 6), (32, 4, 6)]}], [
        {"name": "first", "section": 1, "type": 0x20},
        {"name": "second", "section": 1, "type": 0x20, "value": 16},
        {"name": "third", "section": 1, "type": 0x20, "value": 48},
        {"name": "case0", "section": 1, "storage": 6, "value": 23},
        {"name": "case1", "section": 1, "storage": 6, "value": 24},
        {"name": "table", "section": 1, "storage": 6, "value": 28},
    ]))
    expected = bytearray(code)
    for offset, target in ((19, 28), (28, 23), (32, 24)):
        struct.pack_into("<I", expected, offset, TEXT_VA + target)
    b = {name: {"va": hex(TEXT_VA + offset)} for name, offset in
         (("first", 0), ("second", 16), ("third", 48), ("case0", 23), ("case1", 24), ("table", 28))}
    b["table"].update(data_size=8, inline_data=True)
    b["second"]["size"] = 11
    return c, b, reference(tmp_path, expected)


def test_inline_tables_use_the_bounded_native_owner_and_rebased_code_fixups(ordinary_inline):
    c, b, image = ordinary_inline
    table = c.inline_table("table", 8)
    assert (table["owner_start"], table["owner_end"]) == (16, 48)
    assert check(ordinary_inline, size=11)[0]
    assert validate_data_binding(c, "table", b, image)[1] == 2
    assert check(ordinary_inline, "first", 1)[0]
    assert check(ordinary_inline, "third", 1)[0]
    window = code_window(c, c.function("second"), TEXT_VA + 16, 11, b, image)
    assert window["relocations"] == [(3, 5, 6)]
    assert window["inline_data_bytes"] == 8 and window["excluded_nop_padding"] == 13


@pytest.mark.parametrize("mutation", ["earlier case", "table address", "case address", "gap", "tail",
                                      "partial", "straddle", "alias"])
def test_inline_tables_cannot_escape_or_weaken_the_selected_owner(ordinary_inline, mutation):
    c, b, _ = ordinary_inline
    if mutation == "earlier case":
        c.named["case0"]["value"] = 1
    elif mutation in ("table address", "case address"):
        b["table" if mutation == "table address" else "case0"]["va"] = hex(TEXT_VA)
    elif mutation in ("gap", "tail"):
        raw = bytearray(c.sections[0]["data"])
        raw[27 if mutation == "gap" else 47] = 0xcc
        c.sections[0]["data"] = bytes(raw)
    elif mutation == "partial":
        b["table"]["data_size"] = 4
    elif mutation == "straddle":
        c.sections[0]["relocations"].append((46, 0, 6))
    else:
        c.symbols[6] = dict(c.named["second"], name="alias")
        c.named["alias"] = c.symbols[6]
    with pytest.raises(ValueError):
        check(ordinary_inline, size=11)


@pytest.fixture
def ordinary_project(tmp_path, ordinary):
    _, bindings, image = ordinary
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "Ordinary.obj").write_bytes(ordinary_bytes())
    (tmp_path / "candidate.obj").write_bytes(ordinary_bytes())
    functions = [entry(name, TEXT_VA + offset, size) for name, offset, size in
                 (("first", 0, 6), ("second", 16, 16), ("third", 32, 7))]
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps([{"name": "Ordinary", "functions": functions,
                                     "bindings": {"sample": bindings}}]))
    config = tmp_path / "binary-comp.json"
    config.write_text(json.dumps({"targets": {"sample": {"kind": "i386-coff",
        "original_exe": str(image.path), "original_sha256": hashlib.sha256(Path(image.path).read_bytes()).hexdigest(),
        "object_dir": "objects"}}, "coff": {"modules": "modules.json"}}))
    return config, manifest, tmp_path / "report"


def report_args(project):
    return ["report", "--config", str(project[0]), "--target", "sample", "--no-build",
            "--output", str(project[2])]


def score_args(project):
    return ["compare", "second", "--config", str(project[0]), "--target", "sample",
            "--no-build", "--score", "--object", str(project[0].parent / "candidate.obj")]


def filtered_args(project):
    return ["compare", "second", *report_args(project)[1:]]


def test_report_metrics_filters_and_candidate_score_use_the_same_native_bounds(ordinary_project, capsys):
    p = ordinary_project
    assert main(report_args(p)) == 0
    result = json.loads((p[2] / "report.json").read_text())
    assert [(row["symbol"], row["body_size"], row["checked_fixups"]) for row in result["code"]] == [
        ("first", 6, 0), ("second", 16, 2), ("third", 7, 1)]
    assert all(row["relocated_bytes"]["exact"] for row in result["code"])
    assert all(row["decoded_bytes"]["rebuilt"] == row["body_size"] for row in result["code"])
    assert [(row["native_function_offset"], row["native_function_end"], row["native_section_size"])
            for row in result["code"]] == [(0, 16, 48), (16, 32, 48), (32, 48, 48)]
    assert main(filtered_args(p)) == 0
    filtered = json.loads((p[2] / "report.json").read_text())
    assert len(filtered["code"]) == 1 and filtered["code"][0] == result["code"][1]
    capsys.readouterr()
    assert main(score_args(p)) == 0
    assert capsys.readouterr().out == "0\n"
    c = COFF.from_bytes(ordinary_bytes())
    code = bytearray(c.sections[0]["data"])
    code[27] ^= 1  # unrelocated same-section CALL displacement
    c.sections[0]["data"] = bytes(code)
    (p[0].parent / "candidate.obj").write_bytes(coff_bytes(c.sections, c.symbols.values()))
    assert main(score_args(p)) == 0
    assert capsys.readouterr().out == "1\n"


def test_rejected_extent_diagnostics_describe_only_the_bounded_function(ordinary_project):
    p = ordinary_project
    modules = json.loads(p[1].read_text())
    modules[0]["functions"][1]["sample"]["size"] = 15
    p[1].write_text(json.dumps(modules))
    assert main(filtered_args(p)) == 1
    result = json.loads((p[2] / "report.json").read_text())
    row = result["code"][0]
    assert row["diagnostic_only"] and row["coff_window_size"] == 16
    assert row["comparison_window"] == "complete_coff_function_window"
    assert row["native_function_offset"] == 16 and "error" in row
    assert "complete ordinary COFF function window (16 bytes)" in (p[2] / row["review"]).read_text()
    assert "complete ordinary function window=16 bytes" in (p[2] / "report.txt").read_text()


@pytest.mark.parametrize("mutation", ["filtered binding", "straddling fixup", "COMDAT"])
def test_invalid_filtered_inputs_invalidate_stale_success(ordinary_project, capsys, mutation):
    p = ordinary_project
    args = filtered_args(p)
    assert main(args) == 0
    if mutation == "filtered binding":
        modules = json.loads(p[1].read_text())
        modules[0]["functions"][0]["sample"]["va"] = hex(TEXT_VA + 1)
        p[1].write_text(json.dumps(modules))
    else:
        c = COFF.from_bytes(ordinary_bytes())
        if mutation == "straddling fixup":
            c.sections[0]["relocations"].append((13, 5, 6))
        else:
            c.sections[0]["flags"] |= 0x1000
        raw = coff_bytes(c.sections, c.symbols.values())
        (p[0].parent / "objects/Ordinary.obj").write_bytes(raw)
        (p[0].parent / "candidate.obj").write_bytes(raw)
    assert main(args) == (2 if mutation == "filtered binding" else 1)
    if mutation == "filtered binding":
        assert not (p[2] / "report.json").exists() and not (p[2] / "report.txt").exists()
    else:
        result = json.loads((p[2] / "report.json").read_text())
        row = result["code"][0]
        expected = ("Relocation crosses a typed function boundary" if mutation == "straddling fixup"
                    else "Nonzero function entries require ordinary executable code")
        assert expected in row["error"] and "relocated_bytes" not in row and "review" not in row
        assert not list(p[2].rglob("00401010-function.txt"))
        assert expected in (p[2] / "report.txt").read_text()
    assert main(score_args(p)) == 2
