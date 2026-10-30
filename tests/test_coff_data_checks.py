"""Strict synthetic duplicate emissions that do not add storage coverage."""
import json
import struct

import pytest

from binary_comp.analyzers.coff_report import report, score_config
from binary_comp.cli import main
from binary_comp.core.coff import COFF, compare_function, validate_data_binding
from conftest import DATA_VA, TEXT_VA
from test_coff_contract import DATA_FLAGS, coff_bytes, entry, reference
from test_coff_executable_data import arguments, project


def make_case(tmp_path):
    payload = struct.pack("<II", 17, 3)
    metadata = struct.pack("<IHHIHBBH", 8, 1, 0, 0, 0, 2, 0, 0)
    coff = COFF.from_bytes(coff_bytes([
        {"data": b"\xa1" + bytes(4) + b"\xc3", "relocations": [(1, 3, 6)]},
        {"name": ".data", "flags": DATA_FLAGS, "data": struct.pack("<I", 7)},
        *[{"name": ".rdata", "flags": DATA_FLAGS | 0x1000,
           "data": payload, "relocations": [(4, 1, 6)]} for _ in range(2)],
    ], [{"name": "caller", "section": 1, "type": 0x20},
        {"name": "value", "section": 2, "type": 4},
        {"name": "owned", "section": 3}, {"name": "copy", "section": 4},
        {"name": ".rdata", "section": 3, "storage": 3, "aux": metadata},
        {"name": ".rdata", "section": 4, "storage": 3, "aux": metadata}]))
    bindings = {"value": {"va": hex(DATA_VA), "data_size": 4},
                "owned": {"va": hex(DATA_VA + 4), "data_size": 8},
                "copy": {"va": hex(DATA_VA + 4), "data_size": 8, "data_check_only": True}}
    image = reference(tmp_path, b"\xa1" + struct.pack("<I", DATA_VA + 4) + b"\xc3",
                      struct.pack("<III", 7, 17, DATA_VA + 3))
    return coff, entry("caller", TEXT_VA, 6), bindings, image


def test_duplicate_is_checked_through_real_fixups_without_adding_counts(tmp_path):
    c, f, b, image = case = make_case(tmp_path)
    assert compare_function(c, f, "sample", b, image)[0]
    assert validate_data_binding(c, "copy", b, image) == (image.read(DATA_VA + 4, 8), 1)
    _, _, modules, targets = project(case, tmp_path)
    assert report(modules, targets, ["sample"], tmp_path / "checked") == 0
    checked = json.loads((tmp_path / "checked/report.json").read_text())
    assert len(checked["data"]) == 2
    assert sum(r["size"] for r in checked["data"]) == 12
    assert sum(r["checked_fixups"] for r in checked["data"]) == 1
    assert checked["data_checks"] == [{"version": "sample", "module": "Sample", "symbol": "copy",
                                       "exact": True, "size": 8, "checked_fixups": 1}]
    del b["copy"]["data_check_only"]
    assert report(modules, targets, ["sample"], tmp_path / "counted") == 0
    counted = json.loads((tmp_path / "counted/report.json").read_text())
    assert "data_checks" not in counted and checked["code"] == counted["code"]
    assert counted["data"][:2] == checked["data"]
    assert sum(r["size"] for r in counted["data"]) == 20
    assert sum(r["checked_fixups"] for r in counted["data"]) == 2


@pytest.mark.parametrize("mutation", ["byte", "addend", "kind", "extent"])
def test_bad_check_remains_fatal_to_core_report_and_candidate_score(tmp_path, mutation):
    c, f, b, image = case = make_case(tmp_path)
    raw = bytearray(c.data)
    header = 20 + 3 * 40
    if mutation == "extent":
        b["copy"]["data_size"] = 7
    elif mutation == "kind":
        struct.pack_into("<H", raw, struct.unpack_from("<I", raw, header + 24)[0] + 8, 7)
    else:
        raw[struct.unpack_from("<I", raw, header + 20)[0] + (4 if mutation == "addend" else 0)] ^= 1
    c = COFF.from_bytes(raw)
    with pytest.raises(ValueError):
        compare_function(c, f, "sample", b, image)
    p = project((c, f, b, image), tmp_path)
    with pytest.raises(ValueError):
        score_config(p[0], "sample", "caller", tmp_path / "Sample.obj", module_names=["Sample"])
    assert report(p[2], p[3], ["sample"], tmp_path / "report") > 0
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert result["data_checks"][0]["exact"] is False
    assert len(result["data"]) == 2 and all(r["exact"] for r in result["data"])


@pytest.mark.parametrize("mutation", ["false", "integer", "missing_extent", "inline_data",
                                     "executable_data", "section_layout", "section_layout_only"])
def test_marker_schema_is_strict_in_core_and_cli_and_clears_stale_success(tmp_path, mutation):
    c, _, b, image = case = make_case(tmp_path)
    if mutation in ("false", "integer"):
        b["copy"]["data_check_only"] = False if mutation == "false" else 1
    elif mutation == "missing_extent":
        del b["copy"]["data_size"]
    else:
        b["copy"][mutation] = True
    with pytest.raises(ValueError):
        validate_data_binding(c, "copy", b, image)
    p = project(case, tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    for name in ("report.json", "report.txt"):
        (output / name).write_text("stale success")
    assert main(arguments(p)) != 0
    assert not (output / "report.json").exists() and not (output / "report.txt").exists()


@pytest.mark.parametrize("mutation", ["ordinary", "bss", "code", "common", "undefined", "weak",
                                     "provider", "missing_aux", "sibling"])
def test_only_owned_complete_initialized_comdats_can_be_check_only(tmp_path, mutation):
    c, _, b, image = make_case(tmp_path)
    symbol, section = c.definition("copy")
    if mutation == "ordinary":
        section["flags"] &= ~0x1000
    elif mutation == "bss":
        section["flags"] = (section["flags"] & ~0x40) | 0x80
    elif mutation == "code":
        symbol["type"] = 0x20
        section["flags"] |= 0x20
    elif mutation in ("common", "undefined"):
        symbol.update(section=0, value=8 if mutation == "common" else 0)
    elif mutation == "weak":
        symbol["storage"] = 105
    elif mutation == "missing_aux":
        del c.symbols[6]
    elif mutation == "sibling":
        c.symbols[99] = {**symbol, "name": "sibling", "value": 4}
    with pytest.raises(ValueError):
        validate_data_binding(c, "copy", b, image,
                              common_definitions={"other": c} if mutation == "provider" else None)


def test_unreferenced_check_only_module_has_zero_coverage_and_still_fails(tmp_path):
    metadata = struct.pack("<IHHIHBBH", 4, 0, 0, 0, 0, 2, 0, 0)
    c = COFF.from_bytes(coff_bytes([
        {"name": ".rdata", "flags": DATA_FLAGS | 0x1000, "data": b"tag\0"}],
        [{"name": "copy", "section": 1},
         {"name": ".rdata", "section": 1, "storage": 3, "aux": metadata}]))
    image = reference(tmp_path, b"\xc3", b"tag\0")
    b = {"copy": {"va": hex(DATA_VA), "data_hex": "74616700", "data_check_only": True}}
    p = project((c, entry("caller", TEXT_VA, 1), b, image), tmp_path)
    p[2][0]["functions"] = []
    p[1].write_text(json.dumps(p[2]))
    assert report(p[2], p[3], ["sample"], tmp_path / "report") == 0
    good = json.loads((tmp_path / "report/report.json").read_text())
    assert good["code"] == [] and good["data"] == [] and len(good["data_checks"]) == 1
    raw = bytearray(c.data)
    raw[struct.unpack_from("<I", raw, 20 + 20)[0]] ^= 1
    (tmp_path / "Sample.obj").write_bytes(raw)
    assert main(arguments(p)) != 0
    bad = json.loads((tmp_path / "report/report.json").read_text())
    assert bad["data_checks"][0]["exact"] is False
