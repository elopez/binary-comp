"""Complete shared code contributions, using only synthetic COFF/PE records."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from binary_comp.cli import main
from binary_comp.analyzers.coff_report import report, validate_counted_code
from binary_comp.core.coff import (COFF, CodeRegions, compare_code_region, compare_function,
                                   validate_data_binding, validate_common_definitions)
from binary_comp.core.pe import FileBackedPEImage
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_contract import coff_bytes, entry, DATA_FLAGS, inline_switch


def make_shared(tmp_path, prefix=b"", *, prefix_reference=None, prefix_relocations=()):
    # Both native entries reach the shared load/return tail. References include
    # an internal call, a local branch label, an entry address and real data.
    code = bytearray(b"\x90" * 32)
    code[0:10] = b"\xe8" + bytes(4) + b"\xe9" + bytes(4)
    code[10:18] = b"\x68" + bytes(4) + b"\x58\xeb\x02"
    code[20:26] = b"\xa1" + bytes(4) + b"\xc3"
    start = len(prefix)
    symbols = [{"name": "first", "section": 1, "value": start, "type": 0x20},
               {"name": "second", "section": 1, "value": start + 10, "type": 0x24},
               {"name": "number", "section": 2},
               {"name": "tail", "section": 1, "value": start + 20, "storage": 6},
               {"name": "caller", "section": 3, "type": 0x20},
               {"name": "pointer", "section": 2, "value": 4}]
    raw = coff_bytes([
        {"data": prefix + bytes(code), "relocations": [*prefix_relocations,
            (start + 1, 1, 0x14), (start + 6, 3, 0x14),
            (start + 11, 0, 6), (start + 21, 2, 6)]},
        {"name": ".data", "flags": DATA_FLAGS, "data": struct.pack("<II", 7, 0),
         "relocations": [(4, 1, 6)]},
        {"data": b"\xe8" + bytes(4) + b"\xc3", "relocations": [(1, 1, 0x14)]},
    ], symbols)
    coff = COFF.from_bytes(raw)
    original = bytearray((prefix if prefix_reference is None else prefix_reference)
                         + bytes(code) + b"\x90" * (0x80 - start - len(code)))
    struct.pack_into("<I", original, start + 1, 5)
    struct.pack_into("<I", original, start + 6, 10)
    struct.pack_into("<I", original, start + 11, TEXT_VA + start)
    struct.pack_into("<I", original, start + 21, DATA_VA)
    original += b"\xe8" + struct.pack("<i", start + 10 - 0x85) + b"\xc3"
    image_path = tmp_path / "original.exe"
    write_tiny_pe(image_path, original, {0: struct.pack("<II", 7, TEXT_VA + start + 10)})
    image = FileBackedPEImage(image_path)
    region = {"label": "shared contribution", "symbol": "first",
              "entries": [{"symbol": "first", "offset": start},
                          {"symbol": "second", "offset": start + 10}],
              "retail": {"va": hex(TEXT_VA), "size": start + 32}}
    if start:
        region["anchor_offset"] = start
    bindings = {"first": {"va": hex(TEXT_VA + start)}, "second": {"va": hex(TEXT_VA + start + 10)},
                "tail": {"va": hex(TEXT_VA + start + 20)},
                "number": {"va": hex(DATA_VA), "data_size": 4},
                "pointer": {"va": hex(DATA_VA + 4), "data_size": 4}}
    return coff, region, bindings, image


@pytest.fixture
def shared(tmp_path):
    return make_shared(tmp_path)


@pytest.fixture
def prefixed(tmp_path):
    return make_shared(tmp_path, b"\x90\x90\x90")


def context(shared, **kwargs):
    c, r, b, image = shared
    return CodeRegions(c, [r], "retail", b, image, **kwargs)


def check(shared, **kwargs):
    c, r, b, image = shared
    return compare_code_region(c, r, "retail", b, image, **kwargs)


def test_nonzero_anchor_compares_the_whole_section_and_preserves_function_guards(prefixed):
    c, r, b, image = prefixed
    assert c.code_region("first", r["entries"], anchor_offset=3) is c.sections[0]
    with pytest.raises(ValueError, match="anchor_offset"):
        c.code_region("first", r["entries"])
    exact, actual, original, fixups = check(prefixed)
    assert exact and actual == original and len(actual) == 35 and fixups == 4
    assert original[:3] == b"\x90\x90\x90"
    ctx = context(prefixed)
    assert validate_data_binding(c, "pointer", b, image, code_regions=ctx)[1] == 1
    assert compare_function(c, entry("caller", TEXT_VA + 0x80, 6), "sample", b, image,
                            code_regions=ctx)[0]
    for name in ("first", "second"):
        with pytest.raises(ValueError, match="section offset zero"):
            c.function(name)
    with pytest.raises(ValueError, match="not separate code entries"):
        c.function("tail")
    with pytest.raises(ValueError, match="section offset zero"):
        validate_data_binding(c, "pointer", b, image)


def test_a_later_typed_anchor_does_not_change_the_region_origin(prefixed):
    _, r, *_ = prefixed
    r.update(symbol="second", anchor_offset=13)
    assert check(prefixed)[0]


@pytest.mark.parametrize("offset", [None, False, True, -1, 35, 36, 3.0, "3", [], {}])
def test_anchor_offset_requires_an_in_range_integer(prefixed, offset):
    c, r, *_ = prefixed
    r["anchor_offset"] = offset
    with pytest.raises(ValueError, match="anchor_offset"):
        c.code_region("first", r["entries"], anchor_offset=offset)
    with pytest.raises(ValueError, match="anchor_offset"):
        check(prefixed)


@pytest.mark.parametrize("offset", [0, 2, 4, 13])
def test_declared_anchor_offset_must_equal_its_native_record(prefixed, offset):
    prefixed[1]["anchor_offset"] = offset
    with pytest.raises(ValueError, match="anchor_offset"):
        check(prefixed)


@pytest.mark.parametrize("mutation", ["missing declaration", "missing anchor", "missing sibling",
                                      "moved entry", "origin", "anchor binding", "sibling binding"])
def test_nonzero_anchor_cannot_shift_or_omit_native_ownership(prefixed, mutation):
    _, r, b, _ = prefixed
    if mutation == "missing declaration":
        del r["anchor_offset"]
    elif mutation == "missing anchor":
        r["entries"].pop(0)
    elif mutation == "missing sibling":
        r["entries"].pop()
    elif mutation == "moved entry":
        r["entries"][0]["offset"] = 0
    elif mutation == "origin":
        r["retail"]["va"] = hex(TEXT_VA + 3)
    elif mutation == "anchor binding":
        b["first"]["va"] = hex(TEXT_VA)
    else:
        b["second"]["va"] = hex(TEXT_VA + 10)
    with pytest.raises(ValueError):
        check(prefixed)


@pytest.mark.parametrize("offset", range(3))
def test_nonzero_anchor_prefix_bytes_are_not_trimmed(prefixed, offset):
    c, *_ = prefixed
    raw = bytearray(c.sections[0]["data"])
    raw[offset] ^= 1
    c.sections[0]["data"] = bytes(raw)
    assert not check(prefixed)[0]
    with pytest.raises(ValueError, match="complete code region differs"):
        compare_function(c, entry("caller", TEXT_VA + 0x80, 6), "sample", prefixed[2], prefixed[3],
                         code_regions=context(prefixed))


def test_nonzero_anchor_prefix_relocations_and_data_are_checked(tmp_path):
    shared = make_shared(tmp_path, b"\xa1" + bytes(4),
                         prefix_reference=b"\xa1" + struct.pack("<I", DATA_VA),
                         prefix_relocations=[(1, 2, 6)])
    c, _, b, image = shared
    exact, actual, original, fixups = check(shared)
    assert exact and actual == original and len(actual) == 37 and fixups == 5
    # Only the relocation in the prefix changes; the entry bodies stay intact.
    c.sections[0]["relocations"][0] = (1, 0, 6)
    assert not check(shared)[0]
    with pytest.raises(ValueError, match="complete code region differs"):
        compare_function(c, entry("caller", TEXT_VA + 0x80, 6), "sample", b, image,
                         code_regions=context(shared))
    c.sections[0]["relocations"][0] = (1, 2, 6)
    c.sections[1]["data"] = struct.pack("<II", 8, 0)
    with pytest.raises(ValueError, match="Rebuilt data differs"):
        check(shared)


@pytest.mark.parametrize("mutation", ["anchor offset", "native offset", "entry binding", "prefix"])
def test_nonzero_anchor_context_rechecks_contract_and_bytes(prefixed, mutation):
    c, r, b, image = prefixed
    ctx = context(prefixed)
    assert check(prefixed, code_regions=ctx)[0]
    if mutation == "anchor offset":
        r["anchor_offset"] = 0
    elif mutation == "native offset":
        c.named["first"]["value"] = 4
    elif mutation == "entry binding":
        b["first"]["va"] = hex(TEXT_VA)
    else:
        c.sections[0]["data"] = b"\xcc" + c.sections[0]["data"][1:]
        assert not check(prefixed, code_regions=ctx)[0]
        return
    with pytest.raises(ValueError, match="context|anchor_offset"):
        compare_code_region(c, r, "retail", b, image, code_regions=ctx)


def test_nonzero_anchor_prefix_cannot_overlap_data_or_an_ordinary_owner(prefixed):
    c, _, b, image = prefixed
    with pytest.raises(ValueError, match="Ordinary code overlaps"):
        compare_function(c, entry("caller", TEXT_VA, 2), "sample", b, image,
                         code_regions=context(prefixed))
    b["number"]["va"] = hex(TEXT_VA + 1)
    with pytest.raises(ValueError, match="overlaps configured complete data"):
        check(prefixed)


def test_complete_region_fixups_data_and_helper_inventory(shared):
    c, r, b, image = shared
    exact, actual, original, fixups = check(shared)
    assert exact and actual == original and len(actual) == 32 and fixups == 4
    assert c.sections[0]["data"] != original
    ctx = context(shared)
    assert validate_data_binding(c, "pointer", b, image, code_regions=ctx)[1] == 1
    assert compare_function(c, entry("caller", TEXT_VA + 0x80, 6), "sample", b, image,
                            code_regions=ctx)[0]
    with pytest.raises(ValueError, match="section offset zero"):
        c.function("first")
    with pytest.raises(ValueError, match="section offset zero"):
        validate_data_binding(c, "pointer", b, image)


@pytest.mark.parametrize("offset", [0, 5, 20, 25, 31])
def test_every_instruction_tail_and_padding_byte_participates(shared, offset):
    c, *_ = shared
    raw = bytearray(c.sections[0]["data"])
    raw[offset] ^= 1
    c.sections[0]["data"] = bytes(raw)
    assert not check(shared)[0]


@pytest.mark.parametrize("size", [7, 31, 33])
def test_no_stub_subregion_or_nop_trimming(shared, size):
    shared[1]["retail"]["size"] = size
    with pytest.raises(ValueError, match="complete COFF contribution"):
        check(shared)


@pytest.mark.parametrize("tail", [b"", b"\x90", b"\xc3"])
def test_complete_section_size_is_exact(shared, tail):
    c, *_ = shared
    c.sections[0]["data"] = (c.sections[0]["data"] + tail if tail
                               else c.sections[0]["data"][:-1])
    with pytest.raises(ValueError, match="complete COFF contribution"):
        check(shared)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "moved", "anchor", "one offset"])
def test_exhaustive_exact_entry_map(shared, mutation):
    c, r, *_ = shared
    if mutation == "missing":
        r["entries"].pop()
    elif mutation == "extra":
        r["entries"].append({"symbol": "invented", "offset": 12})
    elif mutation == "duplicate":
        r["entries"].append(dict(r["entries"][0]))
    elif mutation == "moved":
        r["entries"][1]["offset"] += 1
    elif mutation == "anchor":
        r["symbol"] = "second"
    else:
        c.named["second"]["value"] = 0
        r["entries"][1]["offset"] = 0
    with pytest.raises(ValueError):
        check(shared)


@pytest.mark.parametrize("start", [0, 3])
def test_static_auxiliary_entry_is_not_hidden(start):
    c = COFF.from_bytes(coff_bytes([{"data": b"\x90" * start + b"\xc3\x90\xc3"}], [
        {"name": "first", "section": 1, "value": start, "type": 0x20},
        {"name": "second", "section": 1, "value": start + 2, "type": 0x24,
         "storage": 3, "aux": bytes(18)},
    ]))
    entries = [{"symbol": "first", "offset": start}, {"symbol": "second", "offset": start + 2}]
    assert c.code_region("first", entries, anchor_offset=start) is c.sections[0]
    with pytest.raises(ValueError):
        c.code_region("first", entries[:1], anchor_offset=start)
    c.named["second"]["storage"] = 6
    with pytest.raises(ValueError, match="Unsupported native"):
        c.code_region("first", entries, anchor_offset=start)


@pytest.mark.parametrize("name", ["first", "second"])
@pytest.mark.parametrize("mutation", ["missing", "address", "data role"])
def test_each_entry_requires_explicit_consistent_code_binding(shared, name, mutation):
    b = shared[2]
    if mutation == "missing":
        del b[name]
    elif mutation == "address":
        b[name]["va"] = hex(TEXT_VA + 1)
    else:
        b[name]["data_size"] = 1
    with pytest.raises(ValueError, match="exact explicit address"):
        check(shared)


@pytest.mark.parametrize("index", [0, 1, 3])
def test_data_pointers_require_complete_region_even_for_zero_anchor_and_local_labels(shared, index):
    c, _, b, image = shared
    c.sections[1]["relocations"] = [(4, index, 6)]
    with pytest.raises(ValueError, match="section offset zero"):
        validate_data_binding(c, "pointer", b, image)
    c.sections[0]["data"] = bytes([c.sections[0]["data"][0] ^ 1]) + c.sections[0]["data"][1:]
    with pytest.raises(ValueError, match="complete code region differs"):
        validate_data_binding(c, "pointer", b, image, code_regions=context(shared))


def test_local_label_cannot_be_rebound_inside_region(shared):
    shared[2]["tail"]["va"] = hex(TEXT_VA + 21)
    with pytest.raises(ValueError, match="native offset"):
        check(shared)


@pytest.mark.parametrize("mutation", ["overlap", "outside", "unsupported", "missing", "addend"])
def test_fixups_fail_closed(shared, mutation):
    c, _, b, _ = shared
    if mutation == "overlap":
        c.sections[0]["relocations"].append((2, 1, 6))
    elif mutation == "outside":
        c.sections[0]["relocations"].append((30, 1, 6))
    elif mutation == "unsupported":
        c.sections[0]["relocations"][0] = (1, 1, 0x99)
    elif mutation == "missing":
        del b["number"]
    else:
        raw = bytearray(c.sections[0]["data"])
        raw[11] = 1
        c.sections[0]["data"] = bytes(raw)
        assert not check(shared)[0]
        return
    with pytest.raises((ValueError, KeyError)):
        check(shared)


def test_configured_unreferenced_data_still_must_match(shared):
    c, _, _, _ = shared
    c.sections[1]["data"] = c.sections[1]["data"][:4] + b"\x01\0\0\0"
    with pytest.raises(ValueError, match="Rebuilt data differs"):
        check(shared)


@pytest.mark.parametrize("mutation", ["nonexecute", "wrap", "data overlap", "inline role"])
def test_original_backing_and_region_roles(shared, mutation):
    _, r, b, image = shared
    if mutation == "nonexecute":
        r["retail"]["va"] = hex(DATA_VA)
        b["first"]["va"], b["second"]["va"] = hex(DATA_VA), hex(DATA_VA + 10)
    elif mutation == "wrap":
        r["retail"]["va"] = "0xfffffff0"
    elif mutation == "data overlap":
        b["number"]["va"] = hex(TEXT_VA + 31)
    else:
        b["tail"].update(inline_data=True, data_size=4)
    with pytest.raises(ValueError):
        check(shared)


@pytest.mark.parametrize("mutation", ["bindings", "reference", "version", "owner", "entry record"])
def test_context_cannot_authorize_other_inputs_or_stale_records(shared, mutation, tmp_path):
    c, r, b, image = shared
    ctx = context(shared)
    if mutation == "bindings":
        b["second"]["va"] = hex(TEXT_VA + 11)
    elif mutation == "reference":
        image = FileBackedPEImage(image.path)
    elif mutation == "owner":
        c = deepcopy(c)
    elif mutation == "entry record":
        r["entries"][1]["offset"] += 1
    with pytest.raises(ValueError, match="context"):
        compare_code_region(c, r, "other" if mutation == "version" else "retail", b, image,
                            code_regions=ctx)


def test_context_does_not_cache_bytes_or_contribution_ownership(shared):
    c, _, b, image = shared
    ctx = context(shared)
    assert check(shared, code_regions=ctx)[0]
    c.sections[0]["data"] = c.sections[0]["data"][:-1] + b"\xcc"
    with pytest.raises(ValueError, match="complete code region differs"):
        validate_data_binding(c, "pointer", b, image, code_regions=ctx)


def test_common_provider_cannot_inherit_same_section_index_region_permission(shared):
    c, r, b, image = shared
    # The provider has its own typed entries in section 1, just like the owner.
    provider = COFF.from_bytes(coff_bytes([
        {"data": b"\xc3\x90\xc3"},
        {"name": ".data", "flags": DATA_FLAGS, "data": bytes(4), "relocations": [(0, 1, 6)]},
    ], [{"name": "first", "section": 1, "type": 0x20},
        {"name": "second", "section": 1, "value": 2, "type": 0x20},
        {"name": "pointer", "section": 2}]))
    c.named["pointer"].update(section=0, value=4)
    providers = {"pointer": provider}
    ctx = context(shared, common_definitions=providers)
    with pytest.raises(ValueError, match="Provider-owned code regions are unsupported"):
        validate_common_definitions(c, b, image, providers, code_regions=ctx)


def test_sole_nonzero_typed_entry_and_untyped_label_never_bypass_guard(tmp_path):
    c = COFF.from_bytes(coff_bytes([
        {"data": b"\x90\xc3"},
        {"name": ".data", "flags": DATA_FLAGS, "data": bytes(4), "relocations": [(0, 1, 6)]},
    ], [{"name": "only", "section": 1, "value": 1, "type": 0x20},
        {"name": "local", "section": 1, "value": 1, "storage": 6},
        {"name": "pointer", "section": 2}]))
    image_path = tmp_path / "sole.exe"
    write_tiny_pe(image_path)
    b = {"only": {"va": hex(TEXT_VA + 1)}, "local": {"va": hex(TEXT_VA + 1)},
         "pointer": {"va": hex(DATA_VA), "data_size": 4}}
    for index in [0, 1]:
        c.sections[1]["relocations"] = [(0, index, 6)]
        with pytest.raises(ValueError, match="section offset zero"):
            validate_data_binding(c, "pointer", b, FileBackedPEImage(image_path))


def make_region_project(shared, tmp_path):
    c, r, b, image = shared
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "Sample.obj").write_bytes(c.data)
    caller = {"label": "caller", "symbol": "caller",
              "retail": {"va": hex(TEXT_VA + 0x80), "size": 6}}
    modules = [{"name": "Sample", "functions": [caller], "code_regions": [r],
                "bindings": {"retail": b}}]
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps(modules))
    config = tmp_path / "binary-comp.json"
    config.write_text(json.dumps({"coff": {"modules": "modules.json"}, "targets": {
        "retail": {"kind": "i386-coff", "original_exe": image.path,
                   "original_sha256": hashlib.sha256(Path(image.path).read_bytes()).hexdigest(),
                   "object_dir": "objects"}}}))
    return config, manifest, tmp_path / "report", modules


@pytest.fixture
def region_project(shared, tmp_path):
    return make_region_project(shared, tmp_path)


@pytest.fixture
def prefixed_project(prefixed, tmp_path):
    return make_region_project(prefixed, tmp_path)


def args(project, *extra):
    return ["report", "--config", str(project[0]), "--target", "retail", "--no-build",
            "--output", str(project[2]), *extra]


def test_native_report_retains_nonzero_anchor_and_whole_region_origin(prefixed_project):
    assert main(args(prefixed_project)) == 0
    result = json.loads((prefixed_project[2] / "report.json").read_text())
    row = result["code"][1]
    assert row["kind"] == "region" and row["original_va"] == hex(TEXT_VA)
    assert row["anchor_offset"] == 3 and row["body_size"] == 35
    assert row["decoded_bytes"] == {"original": 35, "rebuilt": 35}
    assert not row["raw_bytes"]["exact"] and row["relocated_bytes"]["exact"]
    assert row["checked_fixups"] == 4 and row["excluded_nop_padding"] == 0
    assert row["entries"] == [{"symbol": "first", "offset": 3, "original_va": hex(TEXT_VA + 3)},
                               {"symbol": "second", "offset": 13, "original_va": hex(TEXT_VA + 13)}]
    assert len(result["data"]) == 2 and all(row["exact"] for row in result["data"])


@pytest.mark.parametrize("offset", [None, False, True, -1, 35, 3.0, "3", [], {}])
def test_manifest_rejects_invalid_anchor_offsets_and_stale_success(prefixed_project, offset):
    assert main(args(prefixed_project)) == 0
    _, manifest, output, modules = prefixed_project
    modules[0]["code_regions"][0]["anchor_offset"] = offset
    manifest.write_text(json.dumps(modules))
    assert main(args(prefixed_project)) == 2
    assert not (output / "report.json").exists() and not (output / "report.txt").exists()


@pytest.mark.parametrize("mutation", ["missing declaration", "wrong declaration", "entry binding", "origin"])
def test_manifest_cannot_reinterpret_a_nonzero_anchor_as_region_origin(prefixed_project, mutation):
    _, manifest, output, modules = prefixed_project
    region = modules[0]["code_regions"][0]
    if mutation == "missing declaration":
        del region["anchor_offset"]
    elif mutation == "wrong declaration":
        region["anchor_offset"] = 4
    elif mutation == "entry binding":
        modules[0]["bindings"]["retail"]["first"]["va"] = hex(TEXT_VA)
    else:
        region["retail"]["va"] = hex(TEXT_VA + 3)
    manifest.write_text(json.dumps(modules))
    assert main(args(prefixed_project)) == 2
    assert not (output / "report.json").exists()


def test_anchor_offset_is_not_an_ordinary_function_opt_out(region_project):
    _, manifest, _, modules = region_project
    modules[0]["functions"][0]["anchor_offset"] = 1
    manifest.write_text(json.dumps(modules))
    assert main(args(region_project)) == 2


@pytest.mark.parametrize("score", [False, True])
def test_filtered_ordinary_comparison_rechecks_nonzero_entry_prefix(prefixed_project, capsys, score):
    config, _, output, _ = prefixed_project
    obj = config.parent / "objects/Sample.obj"
    command = ["compare", "caller", "--config", str(config), "--target", "retail", "--no-build"]
    command += ["--score", "--object", str(obj)] if score else ["--output", str(output)]
    assert main(command) == 0
    if score:
        assert capsys.readouterr().out == "0\n"
    else:
        result = json.loads((output / "report.json").read_text())
        assert [row["kind"] for row in result["code"]] == ["function"]
    c = COFF(obj)
    raw = bytearray(obj.read_bytes())
    raw[raw.index(c.sections[0]["data"])] = 0xcc
    obj.write_bytes(raw)
    assert main(command) != 0
    if score:
        assert "complete code region differs" in capsys.readouterr().err
    else:
        result = json.loads((output / "report.json").read_text())
        assert "complete code region differs" in result["code"][0]["error"]


def test_filtered_nonzero_entry_dependency_rechecks_prefix_fixups(tmp_path):
    shared = make_shared(tmp_path, b"\xa1" + bytes(4),
                         prefix_reference=b"\xa1" + struct.pack("<I", DATA_VA),
                         prefix_relocations=[(1, 2, 6)])
    project = make_region_project(shared, tmp_path)
    command = ["compare", "caller", "--config", str(project[0]), "--target", "retail",
               "--no-build", "--output", str(project[2])]
    assert main(command) == 0
    c = shared[0]
    c.sections[0]["relocations"][0] = (1, 0, 6)
    (tmp_path / "objects/Sample.obj").write_bytes(coff_bytes(c.sections, c.symbols.values()))
    assert main(command) != 0
    result = json.loads((project[2] / "report.json").read_text())
    assert "complete code region differs" in result["code"][0]["error"]


def test_nonzero_anchor_region_is_not_a_score_target(prefixed_project, capsys):
    assert main(["compare", "first", "--config", str(prefixed_project[0]), "--target", "retail",
                 "--module", "Sample", "--no-build", "--score", "--object",
                 str(prefixed_project[0].parent / "objects/Sample.obj")]) == 2
    assert "exactly one native function" in capsys.readouterr().err


def test_native_report_has_separate_region_metrics_and_entry_metadata(region_project, capsys):
    assert main(args(region_project)) == 0
    result = json.loads((region_project[2] / "report.json").read_text())
    assert [x["kind"] for x in result["code"]] == ["function", "region"]
    region = result["code"][1]
    assert region["body_size"] == region["decoded_bytes"]["rebuilt"] == 32
    assert region["excluded_nop_padding"] == 0 and region["checked_fixups"] == 4
    assert region["entries"] == [{"symbol": "first", "offset": 0, "original_va": hex(TEXT_VA)},
                                 {"symbol": "second", "offset": 10, "original_va": hex(TEXT_VA + 10)}]
    text = capsys.readouterr().out
    assert "function: 0/1 raw-byte equal; 1/1 relocated-byte equal" in text
    assert "region: 0/1 raw-byte equal; 1/1 relocated-byte equal" in text
    assert len(result["data"]) == 2 and all(x["exact"] for x in result["data"])


def test_filtered_function_checks_region_bytes_as_uncounted_dependency(region_project, capsys):
    config, _, output, _ = region_project
    command = ["compare", "caller", "--config", str(config), "--target", "retail",
               "--no-build", "--output", str(output)]
    assert main(command) == 0
    result = json.loads((output / "report.json").read_text())
    assert [x["kind"] for x in result["code"]] == ["function"]
    obj = config.parent / "objects/Sample.obj"
    c = COFF(obj)
    # Change a shared-tail instruction in the object payload, without changing
    # metadata, placement or the ordinary caller's bytes.
    raw = obj.read_bytes()
    at = raw.index(c.sections[0]["data"])
    mutated = bytearray(raw)
    mutated[at + 25] ^= 1
    obj.write_bytes(mutated)
    assert main(command) != 0
    result = json.loads((output / "report.json").read_text())
    assert "complete code region differs" in result["code"][0]["error"]


def test_filtered_function_cannot_hide_missing_sibling_binding(region_project):
    config, manifest, output, modules = region_project
    del modules[0]["bindings"]["retail"]["first"]
    manifest.write_text(json.dumps(modules))
    assert main(["compare", "caller", "--config", str(config), "--target", "retail",
                 "--no-build", "--output", str(output)]) != 0
    assert not (output / "report.json").exists()


def test_region_selection_has_no_function_score_api(region_project, capsys):
    assert main(["compare", "first", "--config", str(region_project[0]), "--target", "retail",
                 "--module", "Sample", "--no-build", "--score", "--object",
                 str(region_project[0].parent / "objects/Sample.obj")]) == 2
    assert "exactly one native function" in capsys.readouterr().err


@pytest.mark.parametrize("role", ["functions", "code_labels", "code_regions", "data"])
def test_original_regions_cannot_overlap_any_counted_role(region_project, role):
    modules = region_project[3]
    if role == "data":
        modules.append({"name": "Other", "bindings": {"retail": {
            "data": {"va": hex(TEXT_VA + 31), "data_size": 4}}}})
    else:
        other = deepcopy(modules[0]["code_regions"][0])
        other["symbol"] = "other"
        other["retail"] = {"va": hex(TEXT_VA + 16), "size": 4}
        modules.append({"name": "Other", role: [other]})
    with pytest.raises(ValueError, match="overlap|Overlapping"):
        validate_counted_code(modules, ["retail"])


def test_native_section_cannot_have_two_regions_at_disjoint_original_addresses(shared):
    c, r, b, image = shared
    alias = {"name": "alias", "section": 1, "type": 0x20, "storage": 2,
             "value": 0, "aux_count": 0}
    c.symbols[100] = alias
    c.named["alias"] = alias
    r["entries"].append({"symbol": "alias", "offset": 0})
    other = deepcopy(r)
    other["symbol"] = "alias"
    # Equal entries/origin still must not count the one native contribution twice.
    b["alias"] = {"va": hex(TEXT_VA)}
    with pytest.raises(ValueError, match="contribution ownership"):
        CodeRegions(c, [r, other], "retail", b, image)


def test_context_rejects_distinct_contributions_at_overlapping_original_spans(shared):
    c, r, b, image = shared
    c.sections.append(deepcopy(c.sections[0]))
    for name, offset in (("other_first", 0), ("other_second", 10)):
        symbol = {"name": name, "section": 4, "type": 0x20, "storage": 2,
                  "value": offset, "aux_count": 0}
        c.symbols[len(c.symbols)] = symbol
        c.named[name] = symbol
        b[name] = {"va": hex(TEXT_VA + offset)}
    other = deepcopy(r)
    other["symbol"] = "other_first"
    other["entries"] = [{"symbol": "other_first", "offset": 0},
                        {"symbol": "other_second", "offset": 10}]
    with pytest.raises(ValueError, match="Overlapping original"):
        CodeRegions(c, [r, other], "retail", b, image)


def test_direct_ordinary_code_window_cannot_overlap_region(shared):
    c, _, b, image = shared
    with pytest.raises(ValueError, match="Ordinary code overlaps"):
        compare_function(c, entry("caller", TEXT_VA + 30, 6), "sample", b, image,
                         code_regions=context(shared))


def test_native_region_contribution_cannot_also_be_a_function_owner(region_project):
    config, manifest, _, modules = region_project
    modules[0]["functions"] = [{"symbol": "second", "label": "claimed independent body",
                                "retail": {"va": hex(TEXT_VA + 0x80), "size": 6}}]
    # Keep the explicit true entry binding, so the ordinary role cannot rebind it.
    manifest.write_text(json.dumps(modules))
    assert main(args(region_project)) != 0


def test_zero_offset_aliases_with_untyped_local_target_remain_valid(tmp_path):
    c = COFF.from_bytes(coff_bytes([
        {"data": b"\xc3\x90\xc3"},
        {"name": ".data", "flags": DATA_FLAGS, "data": bytes(4), "relocations": [(0, 2, 6)]},
    ], [{"name": "one", "section": 1, "type": 0x20},
        {"name": "alias", "section": 1, "type": 0x20},
        {"name": "local", "section": 1, "value": 2, "storage": 6},
        {"name": "pointer", "section": 2}]))
    path = tmp_path / "aliases.exe"
    write_tiny_pe(path, data_overrides={0: struct.pack("<I", TEXT_VA + 2)})
    bindings = {"one": {"va": hex(TEXT_VA)}, "alias": {"va": hex(TEXT_VA)},
                "local": {"va": hex(TEXT_VA + 2)},
                "pointer": {"va": hex(DATA_VA), "data_size": 4}}
    assert validate_data_binding(c, "pointer", bindings, FileBackedPEImage(path))[1] == 1


def test_filtered_function_does_not_hide_region_function_role_conflict(region_project):
    config, manifest, output, modules = region_project
    modules[0]["functions"].append({"label": "invalid extra owner", "symbol": "first",
                                    "retail": {"va": hex(TEXT_VA), "size": 7}})
    modules[0]["bindings"]["retail"]["first"]["data_size"] = 7
    manifest.write_text(json.dumps(modules))
    assert main(["compare", "caller", "--config", str(config), "--target", "retail",
                 "--no-build", "--output", str(output)]) != 0
    assert not (output / "report.json").exists()


def test_context_freezes_common_provider_object_identities(shared):
    c, r, b, image = shared
    providers = {"pointer": deepcopy(c)}
    ctx = context(shared, common_definitions=providers)
    providers["pointer"] = deepcopy(c)
    with pytest.raises(ValueError, match="context"):
        compare_code_region(c, r, "retail", b, image, common_definitions=providers,
                            code_regions=ctx)


def test_weak_cycle_remains_invalid_inside_a_region(shared):
    c, *_ = shared
    c.symbols[100] = {"name": "weak", "value": 0, "section": 0, "type": 0,
                      "storage": 105, "aux_count": 1, "weak_fallback": 100, "weak_search": 3}
    c.named["weak"] = c.symbols[100]
    c.sections[0]["relocations"][0] = (1, 100, 0x14)
    with pytest.raises(ValueError, match="Cyclic COFF weak"):
        check(shared)


def test_ordinary_inline_tables_keep_region_dependency_context(inline_switch, tmp_path):
    original, _, b, image = inline_switch
    region_code = b"\xc3\x90\xc3\x90"
    c = COFF.from_bytes(coff_bytes([*original.sections, {"data": region_code}], [
        *original.symbols.values(),
        {"name": "first", "section": 2, "type": 0x20},
        {"name": "second", "section": 2, "value": 2, "type": 0x20},
    ]))
    path = tmp_path / "inline-and-region.exe"
    code = image.read(TEXT_VA, 44) + b"\x90" * (0x80 - 44) + region_code
    write_tiny_pe(path, code)
    reference = FileBackedPEImage(path)
    b.update(first={"va": hex(TEXT_VA + 0x80)}, second={"va": hex(TEXT_VA + 0x82)})
    r = {"label": "shared", "symbol": "first",
         "entries": [{"symbol": "first", "offset": 0}, {"symbol": "second", "offset": 2}],
         "sample": {"va": hex(TEXT_VA + 0x80), "size": 4}}
    ctx = CodeRegions(c, [r], "sample", b, reference)
    assert compare_function(c, entry("_switch", TEXT_VA, 32), "sample", b, reference,
                            code_regions=ctx)[0]


def test_context_rejects_moving_native_entries_to_another_contribution(shared):
    c, r, b, image = shared
    ctx = context(shared)
    c.sections.append(deepcopy(c.sections[0]))
    for s in c.symbols.values():
        if s["section"] == 1:
            s["section"] = 4
    with pytest.raises(ValueError, match="native contribution"):
        check(shared, code_regions=ctx)


def test_context_rechecks_mutated_original_bytes(shared):
    c, r, b, image = shared
    ctx = context(shared)
    assert check(shared, code_regions=ctx)[0]
    section = image.section_for_va(TEXT_VA)
    raw = bytearray(image.data)
    raw[section.rawptr + 31] ^= 1
    image.data = bytes(raw)
    assert not check(shared, code_regions=ctx)[0]


def test_inline_owner_transition_cannot_erase_a_data_role(inline_switch, tmp_path):
    original, _, b, image = inline_switch
    region_code = b"\xc3\x90\xc3\x90"
    c = COFF.from_bytes(coff_bytes([*original.sections, {"data": region_code}], [
        *original.symbols.values(),
        {"name": "first", "section": 2, "type": 0x20},
        {"name": "second", "section": 2, "value": 2, "type": 0x20},
    ]))
    path = tmp_path / "invalid-owner.exe"
    write_tiny_pe(path, image.read(TEXT_VA, 44) + b"\x90" * (0x80 - 44) + region_code)
    reference = FileBackedPEImage(path)
    b.update(first={"va": hex(TEXT_VA + 0x80)}, second={"va": hex(TEXT_VA + 0x82)})
    b["_switch"]["data_size"] = 32
    r = {"label": "shared", "symbol": "first",
         "entries": [{"symbol": "first", "offset": 0}, {"symbol": "second", "offset": 2}],
         "sample": {"va": hex(TEXT_VA + 0x80), "size": 4}}
    with pytest.raises(ValueError, match="owner address or role"):
        compare_function(c, entry("_switch", TEXT_VA, 32), "sample", b, reference,
                         code_regions=CodeRegions(c, [r], "sample", b, reference))


def test_filtered_score_cannot_hide_native_label_ownership(region_project, capsys):
    config, manifest, output, modules = region_project
    obj = config.parent / "objects/Sample.obj"
    c = COFF(obj)
    obj.write_bytes(coff_bytes(c.sections, [*c.symbols.values(), {
        "name": "unused", "section": 1, "value": 18, "storage": 6}]))
    modules[0]["code_labels"] = [{"label": "invalid label owner", "symbol": "unused",
                                  "retail": {"va": hex(TEXT_VA + 0x100), "size": 1}}]
    manifest.write_text(json.dumps(modules))
    assert main(["compare", "caller", "--config", str(config), "--target", "retail",
                 "--module", "Sample", "--no-build", "--score", "--object", str(obj)]) == 2
    assert "contribution ownership" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["_code_region_dependencies", "_ordinary_code_owners"])
def test_manifest_cannot_override_reserved_selection_metadata(region_project, key, capsys):
    _, manifest, _, modules = region_project
    modules[0][key] = []
    manifest.write_text(json.dumps(modules))
    assert main(args(region_project)) == 2
    assert "reserved selection metadata" in capsys.readouterr().err


def test_retained_dependencies_cannot_hide_duplicate_counted_region_rows(region_project):
    modules = region_project[3]
    region = modules[0]["code_regions"][0]
    modules[0]["_code_region_dependencies"] = [region]
    modules[0]["code_regions"].append(deepcopy(region))
    with pytest.raises(ValueError, match="Duplicate counted"):
        validate_counted_code(modules, ["retail"])


def test_region_rejects_original_virtual_only_tail(shared):
    _, r, b, image = shared
    # Keep executable virtual coverage, but remove the last file-backed byte.
    section = image.section_for_va(TEXT_VA)
    from dataclasses import replace
    image.sections[0] = replace(section, rawsize=31)
    with pytest.raises(ValueError, match="file-backed"):
        check(shared)
