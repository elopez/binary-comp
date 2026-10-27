"""Explicit complete data in executable contributions, with synthetic inputs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import pytest

from binary_comp.analyzers.coff_report import report, validate_counted_code
from binary_comp.cli import main
from binary_comp.core.coff import (COFF, CodeRegions, apply_fixups, compare_code_region, compare_function,
                                   validate_common_definitions, validate_data_binding)
from binary_comp.core.pe import FileBackedPEImage
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_contract import CODE_FLAGS, DATA_FLAGS, coff_bytes, entry


def make_case(tmp_path, *, extra_symbols=(), referenced=True, table_storage=3,
              table_flags=CODE_FLAGS, table_aux=None):
    caller = b"\xa1" + bytes(4) + b"\xc3" if referenced else b"\xc3"
    table = struct.pack("<IIII", 0, 17, 23, 29)
    symbols = [{"name": "caller", "section": 1, "type": 0x20},
               {"name": "header", "section": 2},
               {"name": "table", "section": 3, "storage": table_storage},
               {"name": "value", "section": 4},
               {"name": ".text$t", "section": 3, "storage": 3,
                "aux": table_aux if table_aux is not None else struct.pack("<IHHIHBBH", 16, 1, 0, 0, 0, 0, 0, 0)},
               *extra_symbols]
    raw = coff_bytes([
        {"data": caller, "relocations": [(1, 1, 6)] if referenced else []},
        {"name": ".rdata", "flags": DATA_FLAGS, "data": struct.pack("<II", 7, 0),
         "relocations": [(4, 2, 6)]},
        {"name": ".text$t", "flags": table_flags, "data": table, "relocations": [(0, 3, 6)]},
        {"name": ".rdata", "flags": DATA_FLAGS, "data": struct.pack("<I", 31)},
    ], symbols)
    code = bytearray(caller + bytes(0x40 - len(caller)) + table)
    if referenced:
        struct.pack_into("<I", code, 1, DATA_VA)
    struct.pack_into("<I", code, 0x40, DATA_VA + 8)
    image_path = tmp_path / "original.exe"
    write_tiny_pe(image_path, code, {0: struct.pack("<III", 7, TEXT_VA + 0x40, 31)})
    bindings = {"header": {"va": hex(DATA_VA), "data_size": 8},
                "table": {"va": hex(TEXT_VA + 0x40), "data_size": 16, "executable_data": True},
                "value": {"va": hex(DATA_VA + 8), "data_size": 4}}
    return COFF.from_bytes(raw), entry("caller", TEXT_VA, len(caller)), bindings, FileBackedPEImage(image_path)


@pytest.fixture
def case(tmp_path):
    return make_case(tmp_path)


def check(case):
    c, f, b, image = case
    return compare_function(c, f, "sample", b, image)


def project(case, tmp_path):
    c, f, b, image = case
    (tmp_path / "Sample.obj").write_bytes(c.data)
    modules = [{"name": "Sample", "functions": [f], "bindings": {"sample": b}}]
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps(modules))
    targets = {"sample": {"kind": "i386-coff", "object_dir": str(tmp_path),
                          "original_exe": image.path,
                          "original_sha256": hashlib.sha256(Path(image.path).read_bytes()).hexdigest()}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"coff": {"modules": "modules.json"}, "targets": targets}))
    return config, manifest, modules, targets


def arguments(p, *, score=False, filtered=False):
    if score:
        return ["compare", "caller", "--score", "--object", str(p[0].parent / "Sample.obj"),
                "--config", str(p[0]), "--target", "sample", "--module", "Sample", "--no-build"]
    result = (["compare", "caller"] if filtered else ["report"]) + ["--config", str(p[0]), "--target", "sample", "--no-build",
              "--output", str(p[0].parent / "report")]
    return result


def test_complete_array_and_pointing_header_reuse_native_data_recursion(case):
    c, _, b, image = case
    actual, fixups = validate_data_binding(c, "table", b, image)
    assert len(actual) == 16 and fixups == 1
    assert validate_data_binding(c, "header", b, image)[1] == 1
    assert check(case)[0] and check(case)[3] == 1
    b["table"] = {"va": b["table"]["va"], "data_hex": actual.hex(), "executable_data": True}
    assert check(case)[0]


def test_native_auxiliary_payload_is_retained_without_changing_symbol_indices(case):
    c = case[0]
    assert c.auxiliary_records == {4: struct.pack("<IHHIHBBH", 16, 1, 0, 0, 0, 0, 0, 0)}
    assert 4 in c.symbols and 5 not in c.symbols
    assert c.resolve_symbol(2)["name"] == "table"
    assert "auxiliary_records" not in c.symbols[4]


@pytest.mark.parametrize("field,value", [("length", 15), ("relocations", 0), ("lines", 1),
                                         ("number", 1), ("selection", 5), ("reserved", 1), ("high", 1)])
def test_section_auxiliary_metadata_must_agree_with_the_actual_contribution(case, field, value):
    c, _, b, image = case
    fields = dict(zip(("length", "relocations", "lines", "checksum", "number", "selection", "reserved", "high"),
                      struct.unpack("<IHHIHBBH", c.auxiliary_records[4])))
    fields[field] = value
    c.auxiliary_records[4] = struct.pack("<IHHIHBBH", *fields.values())
    with pytest.raises(ValueError, match="another native definition"):
        validate_data_binding(c, "table", b, image)


@pytest.mark.parametrize("aux", [bytes(18), struct.pack("<IHHIHBBH", 16, 1, 0, 0, 0, 0, 0, 0)])
def test_aux_count_and_section_name_cannot_hide_a_fabricated_runtime_alias(tmp_path, aux):
    fixture = make_case(tmp_path, extra_symbols=[
        {"name": ".text$t", "section": 3, "storage": 3, "aux": aux}])
    with pytest.raises(ValueError, match="another native definition"):
        check(fixture)


def test_nonassociative_comdat_metadata_is_checked_without_owning_another_section(tmp_path):
    aux = struct.pack("<IHHIHBBH", 16, 1, 0, 0, 0, 2, 0, 0)
    fixture = make_case(tmp_path, table_flags=CODE_FLAGS | 0x1000, table_aux=aux)
    assert check(fixture)[0]
    fixture[0].auxiliary_records[4] = struct.pack("<IHHIHBBH", 16, 1, 0, 0, 1, 5, 0, 0)
    with pytest.raises(ValueError, match="another native definition"):
        check(fixture)


def test_existing_label_only_section_metadata_and_label_targets_remain_code(tmp_path):
    c = COFF.from_bytes(coff_bytes([{"data": b"\xc3"}], [
        {"name": "handler", "section": 1, "storage": 6},
        {"name": ".text", "section": 1, "storage": 3,
         "aux": struct.pack("<IHHIHBBH", 1, 0, 0, 0, 0, 0, 0, 0)}]))
    image_path = tmp_path / "original.exe"
    write_tiny_pe(image_path, b"\xc3")
    image = FileBackedPEImage(image_path)
    b = {"handler": {"va": hex(TEXT_VA)}, ".text": {"va": hex(TEXT_VA)}}
    for index in (0, 1):
        actual, count = apply_fixups(c, bytes(4), [(0, index, 6)], DATA_VA, b, image)
        assert actual == struct.pack("<I", TEXT_VA) and count == 1


def test_section_metadata_cannot_route_around_the_actual_runtime_data_owner(case):
    c, _, b, image = case
    c.definition("header")[1]["relocations"] = [(4, 4, 6)]
    b[".text$t"] = {"va": b["table"]["va"]}
    with pytest.raises(ValueError, match="target differs"):
        validate_data_binding(c, "header", b, image)
    del b["table"]["executable_data"]
    del b["table"]["data_size"]
    with pytest.raises(ValueError, match="complete explicit validation"):
        validate_data_binding(c, "header", b, image)


@pytest.mark.parametrize("remove", [("executable_data",), ("data_size",), ("executable_data", "data_size")])
def test_default_rejection_and_bare_address_cannot_bypass_pointed_to_data(case, remove):
    c, _, b, image = case
    for key in remove:
        del b["table"][key]
    with pytest.raises(ValueError):
        validate_data_binding(c, "header", b, image)
    with pytest.raises(ValueError):
        apply_fixups(c, bytes(4), [(0, 2, 6)], DATA_VA, b, image)
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("value", [False, 1, 0, "true", None, [], {}])
def test_opt_in_requires_exact_true_in_core_and_cli(case, tmp_path, value, capsys):
    case[2]["table"]["executable_data"] = value
    with pytest.raises(ValueError, match="must be true"):
        check(case)
    p = project(case, tmp_path)
    assert main(arguments(p)) != 0
    assert "must be true" in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["partial", "extra", "anchor", "typed", "storage", "aux",
                                     "nonexecute", "ordinary", "inline", "indices", "alignment", "code role",
                                     "ambiguous", "empty", "bool size", "bad hex"])
def test_whole_sole_untyped_executable_role_is_mandatory(case, mutation):
    c, _, b, image = case
    definition, section = c.definition("table")
    if mutation in ("partial", "extra"):
        b["table"]["data_size"] += -4 if mutation == "partial" else 4
    elif mutation == "anchor":
        definition["value"] = 1
    elif mutation == "typed":
        definition["type"] = 0x20
    elif mutation == "storage":
        definition["storage"] = 6
    elif mutation == "aux":
        definition["aux_count"] = 1
    elif mutation == "nonexecute":
        section["flags"] &= ~0x20000000
    elif mutation == "ordinary":
        section["flags"] = DATA_FLAGS
    elif mutation in ("inline", "indices", "alignment", "code role"):
        b["table"][{"inline": "inline_data", "indices": "inline_index_for",
                    "alignment": "inline_alignment", "code role": "size"}[mutation]] = True
    elif mutation == "ambiguous":
        b["table"]["data_hex"] = "00"
    elif mutation == "empty":
        b["table"]["data_size"] = 0
    elif mutation == "bool size":
        b["table"]["data_size"] = True
    elif mutation == "bad hex":
        del b["table"]["data_size"]
        b["table"]["data_hex"] = "zz"
    with pytest.raises(ValueError):
        validate_data_binding(c, "table", b, image, set())


@pytest.mark.parametrize("symbol", [
    {"name": "alias", "section": 3, "storage": 3},
    {"name": "part", "section": 3, "storage": 3, "value": 8},
    {"name": "hidden", "section": 3, "storage": 3, "type": 0x24, "aux": bytes(18)},
    {"name": "label", "section": 3, "storage": 6},
])
def test_raw_native_aliases_subowners_and_typed_aux_entries_are_rejected(tmp_path, symbol):
    fixture = make_case(tmp_path, extra_symbols=[symbol])
    with pytest.raises(ValueError):
        check(fixture)


@pytest.mark.parametrize("mutation", ["wrap", "nonexecute", "raw tail", "section end", "no metadata"])
def test_original_window_is_executable_file_backed_and_inside_i386(case, mutation):
    c, _, b, image = case
    if mutation == "wrap":
        b["table"]["va"] = "0xfffffff8"
    elif mutation == "no metadata":
        image = type("ReadOnly", (), {"read": image.read})()
    else:
        old = image.sections[0]
        changes = {"nonexecute": {"flags": DATA_FLAGS}, "raw tail": {"rawsize": 0x48},
                   "section end": {"end": TEXT_VA + 0x48}}[mutation]
        image.sections[0] = replace(old, **changes)
    with pytest.raises(ValueError):
        validate_data_binding(c, "table", b, image)


@pytest.mark.parametrize("mutation", ["tail", "target", "missing", "outside", "overlap", "kind"])
def test_every_payload_byte_and_relocation_is_checked_through_header(case, mutation):
    c, _, b, image = case
    section = c.definition("table")[1]
    if mutation == "tail":
        section["data"] = section["data"][:-1] + b"\x01"
    elif mutation == "target":
        b["value"]["va"] = hex(DATA_VA + 9)
    elif mutation == "missing":
        del b["value"]
    elif mutation == "outside":
        section["relocations"] = [(14, 3, 6)]
    elif mutation == "overlap":
        section["relocations"].append((0, 3, 6))
    elif mutation == "kind":
        section["relocations"] = [(0, 3, 9)]
    with pytest.raises(ValueError):
        validate_data_binding(c, "header", b, image)


def test_opt_in_cannot_reclassify_common_or_its_provider(case):
    c, _, b, image = case
    provider = deepcopy(c)
    definition = c.named["table"]
    definition.update(section=0, storage=2, value=16)
    with pytest.raises(ValueError, match="COMMON"):
        validate_data_binding(c, "table", b, image, common_definitions={"table": provider})
    # A recognized provider cannot borrow its consumer's executable-data role.
    with pytest.raises(ValueError, match="Provider-owned"):
        validate_data_binding(provider, "table", b, image, set(), common_definitions={"table": provider})
    with pytest.raises(ValueError):
        validate_common_definitions(c, b, image, {"table": provider})


def test_ordinary_provider_recursion_works_but_cannot_borrow_consumer_static_roles(case):
    c, f, b, image = case
    c.named["value"].update(section=0, storage=2, value=4)
    provider = COFF.from_bytes(coff_bytes([
        {"data": struct.pack("<I", 31), "flags": DATA_FLAGS}],
        [{"name": "value", "section": 1}]))
    assert compare_function(c, f, "sample", b, image, common_definitions={"value": provider})[0]
    foreign = deepcopy(case[0])
    foreign.named["value"].update(section=4, value=0)
    with pytest.raises(ValueError, match="Provider-owned"):
        compare_function(c, f, "sample", b, image, common_definitions={"value": foreign})


@pytest.mark.parametrize("score,filtered", [(False, False), (False, True), (True, True)])
def test_cli_report_and_score_validate_complete_dependency_and_separate_data(case, tmp_path, score, filtered, capsys):
    p = project(case, tmp_path)
    assert main(arguments(p, score=score, filtered=filtered)) == 0
    if score:
        assert capsys.readouterr().out == "0\n"
    else:
        result = json.loads((tmp_path / "report/report.json").read_text())
        assert len(result["code"]) == 1 and result["code"][0]["kind"] == "function"
        assert len(result["data"]) == 3 and all(row["exact"] for row in result["data"])
        assert next(row for row in result["data"] if row["symbol"] == "table")["storage"] == "executable"
    raw = bytearray((tmp_path / "Sample.obj").read_bytes())
    c = case[0]
    # A non-pointer tail byte changes; no relocation can conceal it.
    offset = c.data.index(c.definition("table")[1]["data"])
    raw[offset + 15] ^= 1
    (tmp_path / "Sample.obj").write_bytes(raw)
    assert main(arguments(p, score=score, filtered=filtered)) != 0


@pytest.mark.parametrize("kind", ["function", "label", "region", "data", "unselected function"])
def test_original_executable_data_cannot_overlap_other_complete_owners(case, tmp_path, kind):
    _, _, b, _ = case
    modules = project(case, tmp_path)[2]
    other = {"symbol": "other", "label": "other", "sample": {"va": b["table"]["va"], "size": 4}}
    if kind == "data":
        modules.append({"name": "Other", "bindings": {"sample": {"other": {"va": b["table"]["va"], "data_size": 4}}}})
    elif kind == "unselected function":
        modules[0]["_ordinary_code_owners"] = [other]
    else:
        modules.append({"name": "Other", {"function": "functions", "label": "code_labels", "region": "code_regions"}[kind]: [other]})
    with pytest.raises(ValueError, match="overlap"):
        validate_counted_code(modules, ["sample"])


def test_same_native_contribution_cannot_get_two_disjoint_original_owners(case, tmp_path):
    p = project(case, tmp_path)
    other = deepcopy(p[2][0])
    other.update(name="Other", functions=[])
    (tmp_path / "Other.obj").symlink_to("Sample.obj")
    other["bindings"]["sample"] = {"table": {"va": hex(TEXT_VA + 0x80), "data_size": 16,
                                               "executable_data": True}}
    with pytest.raises(ValueError, match="Duplicate native"):
        report(p[2] + [other], p[3], ["sample"], tmp_path / "report")


def test_external_identity_check_still_covers_executable_data(tmp_path):
    fixture = make_case(tmp_path, table_storage=2)
    p = project(fixture, tmp_path)
    other = deepcopy(p[2][0])
    other.update(name="Other", functions=[])
    (tmp_path / "Other.obj").write_bytes(fixture[0].data)
    other["bindings"]["sample"]["table"]["va"] = hex(TEXT_VA + 0x80)
    with pytest.raises(ValueError, match="Conflicting external binding"):
        report(p[2] + [other], p[3], ["sample"], tmp_path / "report")


@pytest.mark.parametrize("score", [False, True])
def test_unreferenced_configured_array_must_still_pass_report_and_score(tmp_path, score):
    fixture = make_case(tmp_path, referenced=False)
    del fixture[2]["header"]
    p = project(fixture, tmp_path)
    assert main(arguments(p, score=score, filtered=True)) == 0
    raw = bytearray(fixture[0].data)
    offset = raw.index(fixture[0].definition("table")[1]["data"])
    raw[offset + 15] ^= 1
    (tmp_path / "Sample.obj").write_bytes(raw)
    assert main(arguments(p, score=score, filtered=True)) != 0


@pytest.mark.parametrize("score", [False, True])
def test_filtered_cli_cannot_hide_an_overlapping_unselected_code_owner(case, tmp_path, score):
    p = project(case, tmp_path)
    p[2][0]["functions"].append({"label": "unselected", "symbol": "unselected",
                                  "sample": {"va": case[2]["table"]["va"], "size": 4}})
    p[1].write_text(json.dumps(p[2]))
    assert main(arguments(p, score=score, filtered=True)) != 0


def test_explicit_data_coexists_with_strict_code_region_dependencies(tmp_path):
    from test_coff_regions import make_shared

    original_coff, region, bindings, old_image = make_shared(tmp_path)
    sections = [dict(section) for section in original_coff.sections]
    sections.append({"data": b"list", "flags": CODE_FLAGS, "relocations": []})
    symbols = list(original_coff.symbols.values()) + [{"name": "lookup", "section": 4, "storage": 3}]
    c = COFF.from_bytes(coff_bytes(sections, symbols))
    raw_image = bytearray(Path(old_image.path).read_bytes())
    raw_image[0x300:0x304] = b"list"
    Path(old_image.path).write_bytes(raw_image)
    image = FileBackedPEImage(old_image.path)
    bindings["lookup"] = {"va": hex(TEXT_VA + 0x100), "data_size": 4, "executable_data": True}
    context = CodeRegions(c, [region], "retail", bindings, image)
    assert compare_code_region(c, region, "retail", bindings, image, code_regions=context)[0]
    assert validate_data_binding(c, "lookup", bindings, image, code_regions=context) == (b"list", 0)
