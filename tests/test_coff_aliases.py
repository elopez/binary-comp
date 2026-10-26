"""Co-located native entry aliases, using synthetic COFF and PE records only."""
from copy import deepcopy
import json
import struct

import pytest

from binary_comp.analyzers.coff_report import load_selection, report
from binary_comp.cli import main
from binary_comp.config import ConfigError
from binary_comp.core.coff import BodyExtentError, COFF, compare_function, validate_data_binding
from conftest import DATA_VA, TEXT_VA
from test_coff_cli import arguments, project
from test_coff_contract import DATA_FLAGS, coff_bytes, entry, inline_switch, reference
from test_coff_score import score_arguments


def aliased_bytes(names, *, offsets=(0, 0), code=None, value=7, self_references=False):
    code = bytes.fromhex("a1 00 00 00 00 c3") if code is None else code
    relocations = [(1, 2, 6)]
    if self_references:
        code = bytes.fromhex("a1 00 00 00 00 68 00 00 00 00 e8 00 00 00 00 c3")
        relocations += [(6, names.index("alias"), 6), (11, names.index("alias"), 0x14)]
    return coff_bytes([
        {"data": code, "relocations": relocations},
        {"name": ".data", "data": struct.pack("<I", value), "flags": DATA_FLAGS},
    ], [{"name": name, "section": 1, "value": offset, "type": 0x24}
        for name, offset in zip(names, offsets)] + [{"name": "number", "section": 2}])


@pytest.mark.parametrize("names", [("answer", "alias"), ("alias", "answer")])
def test_parser_preserves_both_zero_offset_entries_and_their_indices(names):
    coff = COFF.from_bytes(aliased_bytes(names))
    assert [coff.symbols[i]["name"] for i in range(3)] == [*names, "number"]
    assert coff.function("answer") is coff.function("alias") is coff.sections[0]
    assert coff.function("alias")["relocations"] == [(1, 2, 6)]


@pytest.mark.parametrize("offsets", [(0, 1), (1, 0), (1, 1), (1, 2)])
@pytest.mark.parametrize("selected", ["answer", "alias"])
def test_parser_rejects_distinct_entries_and_nonzero_alias_groups(offsets, selected):
    coff = COFF.from_bytes(aliased_bytes(("answer", "alias"), offsets=offsets))
    with pytest.raises(ValueError, match="section offset zero"):
        coff.function(selected)


@pytest.mark.parametrize("offset", [0, 1])
def test_static_function_auxiliary_metadata_does_not_hide_an_entry(offset):
    coff = COFF.from_bytes(coff_bytes([{"data": b"\xc3\x90"}], [
        {"name": "answer", "section": 1, "type": 0x20},
        {"name": "local_alias", "section": 1, "type": 0x24,
         "storage": 3, "aux": bytes(18), "value": offset},
    ]))
    assert coff.symbols[1]["name"] == "local_alias" and 2 not in coff.symbols
    for name in ("answer", "local_alias"):
        if offset:
            with pytest.raises(ValueError, match="section offset zero"):
                coff.function(name)
        else:
            assert coff.function(name) is coff.sections[0]


@pytest.fixture(params=[("answer", "alias"), ("alias", "answer")])
def aliased_function(tmp_path, request):
    coff = COFF.from_bytes(aliased_bytes(request.param, self_references=True))
    expected = bytearray(coff.sections[0]["data"])
    struct.pack_into("<I", expected, 1, DATA_VA)
    struct.pack_into("<I", expected, 6, TEXT_VA)
    struct.pack_into("<i", expected, 11, -15)
    bindings = {name: {"va": hex(TEXT_VA)} for name in request.param}
    bindings["number"] = {"va": hex(DATA_VA), "data_size": 4}
    return coff, bindings, reference(tmp_path, expected, struct.pack("<I", 7))


@pytest.mark.parametrize("selected", ["answer", "alias"])
def test_either_alias_compares_the_complete_body_with_self_and_data_fixups(aliased_function, selected):
    coff, bindings, image = aliased_function
    ok, actual, expected, fixups = compare_function(
        coff, entry(selected, TEXT_VA, 16), "sample", bindings, image)
    assert ok and actual == expected and fixups == 3
    assert coff.function(selected)["data"] != expected


@pytest.mark.parametrize("mutation", ["instruction", "data", "extent", "fixup", "missing alias"])
def test_aliases_keep_instruction_data_extent_and_fixup_checks(aliased_function, mutation):
    coff, bindings, image = deepcopy(aliased_function)
    section = coff.function("answer")
    if mutation == "instruction":
        section["data"] = b"\xa3" + section["data"][1:]
        assert not compare_function(coff, entry("alias", TEXT_VA, 16), "sample", bindings, image)[0]
        return
    if mutation == "data":
        coff.sections[1]["data"] = struct.pack("<I", 8)
    elif mutation == "extent":
        section["data"] += b"\xc3"
    elif mutation == "fixup":
        section["relocations"].append((14, 0, 6))
    else:
        del bindings["alias"]
    error = BodyExtentError if mutation == "extent" else ValueError
    with pytest.raises(error):
        compare_function(coff, entry("answer", TEXT_VA, 16), "sample", bindings, image)


@pytest.mark.parametrize("bad_name", ["answer", "alias"])
@pytest.mark.parametrize("address", [TEXT_VA + 16, TEXT_VA + (1 << 32)])
def test_all_explicit_alias_addresses_must_match_the_compared_origin(aliased_function, bad_name, address):
    coff, bindings, image = deepcopy(aliased_function)
    bindings[bad_name]["va"] = hex(address)
    with pytest.raises(ValueError, match="alias address differs|outside i386 address domain"):
        compare_function(coff, entry("answer", TEXT_VA, 16), "sample", bindings, image)


def test_aliases_agreeing_with_each_other_still_need_the_compared_origin(aliased_function):
    coff, bindings, image = deepcopy(aliased_function)
    for name in ("answer", "alias"):
        bindings[name]["va"] = hex(TEXT_VA + 16)
    with pytest.raises(ValueError, match="alias address differs"):
        compare_function(coff, entry("answer", TEXT_VA, 16), "sample", bindings, image)


def test_data_fixups_cannot_give_colocated_aliases_different_addresses(tmp_path):
    coff = COFF.from_bytes(coff_bytes([
        {"data": b"\xc3"},
        {"name": ".data", "data": bytes(4), "flags": DATA_FLAGS, "relocations": [(0, 0, 6)]},
    ], [{"name": "answer", "section": 1, "type": 0x20},
        {"name": "alias", "section": 1, "type": 0x20},
        {"name": "pointer", "section": 2}]))
    bindings = {"answer": {"va": hex(TEXT_VA)}, "alias": {"va": hex(TEXT_VA + 16)},
                "pointer": {"va": hex(DATA_VA), "data_size": 4}}
    image = reference(tmp_path, b"\xc3", struct.pack("<I", TEXT_VA))
    with pytest.raises(ValueError, match="alias address differs"):
        validate_data_binding(coff, "pointer", bindings, image)


@pytest.mark.parametrize("selected", ["_switch", "alias"])
def test_inline_tables_keep_the_single_typed_owner_guard(inline_switch, selected):
    original, _, bindings, image = inline_switch
    alias = {"name": "alias", "section": 1, "value": 0, "storage": 2, "type": 0x20}
    coff = COFF.from_bytes(coff_bytes(original.sections, [*original.symbols.values(), alias]))
    bindings["alias"] = {"va": hex(TEXT_VA), "size": 32}
    assert coff.function(selected) is coff.sections[0]
    with pytest.raises(ValueError, match="Inline data requires one /Gy function per section"):
        compare_function(coff, entry(selected, TEXT_VA, 32), "sample", bindings, image)


def configure_alias(project, selected, names, **kwargs):
    modules = json.loads(project[1].read_text())
    modules[0]["functions"][0]["symbol"] = selected
    other = "alias" if selected == "answer" else "answer"
    modules[0]["bindings"]["retail"][other] = {"va": hex(TEXT_VA)}
    project[1].write_text(json.dumps(modules))
    data = aliased_bytes(names, **kwargs)
    (project[0].parent / "candidate.obj").write_bytes(data)
    (project[0].parent / "retail/Sample.obj").write_bytes(data)


@pytest.mark.parametrize("selected", ["answer", "alias"])
@pytest.mark.parametrize("names", [("answer", "alias"), ("alias", "answer")])
def test_cli_score_and_report_accept_either_alias_in_either_order(project, capsys, selected, names):
    configure_alias(project, selected, names)
    assert main(score_arguments(project)) == 0
    assert capsys.readouterr().out == "0\n"
    assert main(arguments(project, target="retail")) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert len(result["code"]) == 1
    row = result["code"][0]
    assert row["symbol"] == selected and row["relocated_bytes"]["exact"]
    assert not row["raw_bytes"]["exact"] and row["checked_fixups"] == 1
    assert row["body_size"] == row["decoded_bytes"]["rebuilt"] == 6
    assert len(result["data"]) == 1 and result["data"][0]["exact"]


@pytest.mark.parametrize("mutation", ["entry offset", "alias address", "selected address", "data", "fixup"])
def test_cli_score_and_report_reject_invalid_alias_candidates(project, capsys, mutation):
    configure_alias(project, "answer", ("answer", "alias"),
                    offsets=(0, 1) if mutation == "entry offset" else (0, 0),
                    value=8 if mutation == "data" else 7)
    if mutation in ("alias address", "selected address"):
        modules = json.loads(project[1].read_text())
        name = "alias" if mutation == "alias address" else "answer"
        modules[0]["bindings"]["retail"][name] = {"va": hex(TEXT_VA + 16)}
        project[1].write_text(json.dumps(modules))
    elif mutation == "fixup":
        for path in [project[0].parent / "candidate.obj", project[0].parent / "retail/Sample.obj"]:
            data = bytearray(path.read_bytes())
            reloc_at = struct.unpack_from("<I", data, 44)[0]
            struct.pack_into("<H", data, reloc_at + 8, 0x99)
            path.write_bytes(data)
    assert main(score_arguments(project)) == 2
    captured = capsys.readouterr()
    assert not captured.out and "error:" in captured.err
    assert main(arguments(project, target="retail")) != 0


def test_report_api_does_not_overwrite_an_explicit_selected_alias_address(project):
    configure_alias(project, "answer", ("answer", "alias"))
    modules, targets, _ = load_selection(project[0], "retail")
    modules[0]["bindings"]["retail"]["answer"] = {"va": hex(TEXT_VA + 16)}
    with pytest.raises(ConfigError, match="Conflicting configured code address"):
        report(modules, targets, ["retail"], project[2])


@pytest.mark.parametrize("mutation", ["instruction", "extent"])
def test_cli_alias_mismatches_remain_positive_scores_and_failed_reports(project, capsys, mutation):
    code = bytes.fromhex("a1 00 00 00 00 cb" if mutation == "instruction" else "a1 00 00 00 00 c3 c3")
    configure_alias(project, "alias", ("alias", "answer"), code=code)
    assert main(score_arguments(project)) == 0
    assert int(capsys.readouterr().out) > 0
    assert main(arguments(project, target="retail")) == 1
    row = json.loads((project[2] / "report.json").read_text())["code"][0]
    assert not row["relocated_bytes"]["exact"]
    if mutation == "extent":
        assert row["diagnostic_only"] and "Non-NOP" in row["error"]
