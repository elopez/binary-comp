"""Opt-in complete ordinary contribution layouts, with synthetic native inputs."""
from copy import deepcopy
from dataclasses import replace
import json
import struct

import pytest

from binary_comp.analyzers.coff_report import report, score_config
from binary_comp.cli import main
from binary_comp.config import ConfigError
from binary_comp.core.coff import (COFF, CodeRegions, apply_fixups, compare_code_region,
                                   compare_function, validate_common_definitions, validate_data_binding)
from conftest import DATA_VA, TEXT_VA
from test_coff_contract import BSS_FLAGS, DATA_FLAGS, coff_bytes, entry, reference
from test_coff_executable_data import arguments, project


def aux(size, relocations=0):
    return struct.pack("<IHHIHBBH", size, relocations, 0, 0, 0, 0, 0, 0)


def make_case(tmp_path, *, bss=False, prefix_fixup=False, extra_symbols=(), alignment_flags=0):
    caller = b"\xa1" + bytes(4) + b"\xc3"
    data = bytes(12) if bss else b"head" + struct.pack("<II", 17, 3) + b"abcdefgh"
    offsets = (0, 4, 8) if bss else (4, 8, 12)
    relocations = [] if bss else [(8, 1, 6)] + ([(0, 1, 6)] if prefix_fixup else [])
    symbols = [{"name": "caller", "section": 1, "type": 0x24},
               {"name": "scalar", "section": 2, "type": 4},
               {"name": "pointer", "section": 2, "type": 0x14, "storage": 3},
               {"name": "array", "section": 2, "type": 0x34, "storage": 3},
               {"name": ".data", "section": 2, "storage": 3, "aux": aux(len(data), len(relocations))},
               *extra_symbols]
    for symbol, offset in zip(symbols[1:4], offsets):
        symbol["value"] = offset
    if prefix_fixup:
        data = bytes(4) + data[4:]
    c = COFF.from_bytes(coff_bytes([
        {"data": caller, "relocations": [(1, 2, 6)]},
        {"name": ".data", "flags": (BSS_FLAGS if bss else DATA_FLAGS) | alignment_flags,
         "data": data, "size": len(data), "relocations": relocations},
    ], symbols))
    expected = bytearray(data)
    if not bss:
        struct.pack_into("<I", expected, 8, DATA_VA + 7)
    if prefix_fixup:
        struct.pack_into("<I", expected, 0, DATA_VA + 4)
    image = reference(tmp_path, b"\xa1" + struct.pack("<I", DATA_VA + offsets[1]) + b"\xc3", bytes(expected) + (bytes(32 - len(expected)) if bss else b""))
    b = {name: {"va": hex(DATA_VA + offset), "data_size": c.definition_extent(name)}
         for name, offset in zip(("scalar", "pointer", "array"), offsets)}
    b["pointer"]["section_layout"] = True
    return c, entry("caller", TEXT_VA, 6), b, image


@pytest.fixture
def case(tmp_path):
    return make_case(tmp_path)


def check(case, **kwargs):
    c, f, b, image = case
    return compare_function(c, f, "sample", b, image, **kwargs)


def test_interior_anchor_typed_scalar_array_pointer_and_unchanged_totals(case, tmp_path):
    c, _, b, image = case
    assert check(case)[0]
    assert [validate_data_binding(c, name, b, image)[1] for name in b] == [0, 1, 0]
    b["array"] = {"va": b["array"]["va"], "data_hex": b"abcdefgh".hex()}
    assert check(case)[0]
    _, _, modules, targets = project(case, tmp_path)
    output = tmp_path / "report"
    assert report(modules, targets, ["sample"], output) == 0
    result = json.loads((output / "report.json").read_text())
    assert len(result["data"]) == 3
    assert sum(row["size"] for row in result["data"]) == 16
    assert sum(row["checked_fixups"] for row in result["data"]) == 1
    assert len(result["code"]) == 1 and result["code"][0]["checked_fixups"] == 1


def test_explicit_alignment_uses_contribution_origin_not_interior_anchor(tmp_path):
    case = make_case(tmp_path, bss=True, alignment_flags=0x00400000)
    c, _, b, image = case
    assert DATA_VA % 8 == 0 and int(b["pointer"]["va"], 16) % 8 == 4
    assert check(case)[0]
    assert validate_data_binding(c, "scalar", b, image) == (bytes(4), 0)


@pytest.mark.parametrize("api", ["data", "function"])
def test_misaligned_origin_rejects_before_original_reads(tmp_path, api):
    c, f, b, image = make_case(tmp_path, bss=True, alignment_flags=0x00400000)
    for binding in b.values():
        binding["va"] = hex(int(binding["va"], 16) + 4)

    class NoReads:
        def section_for_va(self, address):
            return image.section_for_va(address)

        def read(self, address, size):
            raise AssertionError("Alignment must reject before original bytes are read")

    with pytest.raises(ValueError, match="original origin violates native alignment"):
        if api == "data":
            validate_data_binding(c, "scalar", b, NoReads())
        else:
            compare_function(c, f, "sample", b, NoReads())
    del b["pointer"]["section_layout"]
    assert validate_data_binding(c, "scalar", b, image) == (bytes(4), 0)


@pytest.mark.parametrize("alignment_flags", [0, 0x00F00000])
def test_unspecified_and_unsupported_layout_alignment(tmp_path, alignment_flags):
    c, _, b, image = make_case(tmp_path, bss=True, alignment_flags=alignment_flags)
    for binding in b.values():
        binding["va"] = hex(int(binding["va"], 16) + 4)
    if alignment_flags:
        with pytest.raises(ValueError, match="Unsupported native section layout alignment"):
            validate_data_binding(c, "scalar", b, image)
    else:
        assert validate_data_binding(c, "scalar", b, image) == (bytes(4), 0)


@pytest.mark.parametrize("value", [False, 0, 1, "true", None, [], {}])
def test_true_only_syntax_in_core_cli_and_score(case, tmp_path, value):
    case[2]["pointer"]["section_layout"] = value
    with pytest.raises(ValueError, match="must be true"):
        apply_fixups(case[0], b"", [], TEXT_VA, case[2], case[3], set())
    p = project(case, tmp_path)
    assert main(arguments(p)) != 0
    with pytest.raises(ConfigError, match="must be true"):
        score_config(p[0], "sample", "caller", tmp_path / "Sample.obj", module_names=["Sample"])


@pytest.mark.parametrize("name", ["scalar", "pointer", "array"])
@pytest.mark.parametrize("mutation", ["missing", "bare", "short", "long", "size", "inline_data",
                                     "inline_index_for", "inline_alignment", "executable_data", "ambiguous"])
def test_every_member_requires_complete_ordinary_role(case, name, mutation):
    b = case[2]
    if mutation == "missing":
        del b[name]
    elif mutation == "bare":
        del b[name]["data_size"]
    elif mutation in ("short", "long"):
        b[name]["data_size"] += -1 if mutation == "short" else 1
    elif mutation == "ambiguous":
        b[name]["data_hex"] = "00"
    else:
        b[name][mutation] = True
    # Removing the selected anchor removes the constraint by design.
    if name == "pointer" and mutation == "missing":
        b["scalar"]["section_layout"] = True
    with pytest.raises((ValueError, KeyError)):
        check(case)


@pytest.mark.parametrize("mutation", ["shift", "rotate", "duplicate_anchor", "negative", "overflow"])
def test_all_offsets_need_one_checked_original_origin(tmp_path, mutation):
    c, f, b, image = case = make_case(tmp_path, bss=True)
    if mutation == "shift":
        b["scalar"]["va"] = hex(DATA_VA + 1)
    elif mutation == "rotate":
        b["scalar"]["va"], b["array"]["va"] = b["array"]["va"], b["scalar"]["va"]
    elif mutation == "duplicate_anchor":
        b["scalar"]["section_layout"] = True
    else:
        base = -1 if mutation == "negative" else 0xFFFFFFF8
        for name in b:
            b[name]["va"] = hex(base + c.named[name]["value"])
    with pytest.raises(ValueError):
        validate_data_binding(c, "pointer", b, image, set())


def test_default_independent_zero_placements_and_uniform_shift_limit(tmp_path):
    c, f, b, image = make_case(tmp_path, bss=True)
    original = deepcopy(b)
    del b["pointer"]["section_layout"]
    b["scalar"]["va"], b["array"]["va"] = b["array"]["va"], b["scalar"]["va"]
    assert validate_data_binding(c, "scalar", b, image) == (bytes(4), 0)
    b = original
    # Whole repeated zeros do not establish a unique incoming address.
    for binding in b.values():
        binding["va"] = hex(int(binding["va"], 16) + 12)
    assert validate_data_binding(c, "scalar", b, image) == (bytes(4), 0)
    assert not compare_function(c, f, "sample", b, image)[0]


@pytest.mark.parametrize("mutation", ["prefix_byte", "prefix_addend", "tail_byte", "kind", "index", "overlap",
                                     "outside", "cross_member", "cross_prefix", "target"])
def test_complete_bytes_and_all_fixups_remain_authoritative(tmp_path, mutation):
    c, _, b, image = case = make_case(tmp_path, prefix_fixup=True)
    section = c.sections[1]
    data = bytearray(section["data"])
    if mutation in ("prefix_byte", "prefix_addend", "tail_byte", "target"):
        if mutation == "target":
            b["scalar"]["va"] = hex(DATA_VA + 5)
        else:
            data[{"prefix_byte": 0, "prefix_addend": 1, "tail_byte": 19}[mutation]] ^= 1
            section["data"] = bytes(data)
    else:
        offset, index, kind = section["relocations"][0]
        section["relocations"][0] = {
            "kind": (offset, index, 7), "index": (offset, 99, kind),
            "overlap": (1, index, kind), "outside": (18, index, kind),
            "cross_member": (6, index, kind), "cross_prefix": (2, index, kind),
        }[mutation]
    with pytest.raises(ValueError):
        validate_data_binding(c, "array", b, image)


@pytest.mark.parametrize("field", range(8))
def test_native_auxiliary_fields_are_checked(case, field):
    c = case[0]
    fields = list(struct.unpack("<IHHIHBBH", c.auxiliary_records[4]))
    if field == 3:  # The native checksum is intentionally not an ownership contract.
        fields[field] = 123
        c.auxiliary_records[4] = struct.pack("<IHHIHBBH", *fields)
        assert check(case)[0]
        return
    fields[field] += 1
    c.auxiliary_records[4] = struct.pack("<IHHIHBBH", *fields)
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("symbol", [
    {"name": "alias", "section": 2, "value": 4},
    {"name": "", "section": 2, "value": 3},
    {"name": "end", "section": 2, "value": 20},
    {"name": "label", "section": 2, "value": 1, "storage": 6},
    {"name": "auxdata", "section": 2, "value": 1, "aux": bytes(18)},
    {"name": "function", "section": 2, "value": 1, "type": 0x24},
    {"name": ".data", "section": 2, "storage": 3, "aux": aux(20, 1)},
])
def test_exhaustive_inventory_rejects_extra_runtime_and_duplicate_metadata(tmp_path, symbol):
    with pytest.raises(ValueError):
        check(make_case(tmp_path, extra_symbols=[symbol]))


@pytest.mark.parametrize("mutation", ["missing_aux", "header_length", "header_flags", "header_lines", "empty",
                                     "code", "execute", "comdat", "info", "remove", "discard", "no_data",
                                     "common", "weak", "absolute", "aux", "type", "storage"])
def test_native_metadata_and_anchor_roles_cannot_reclassify_storage(case, mutation):
    c, _, b, _ = case
    section, symbol = c.sections[1], c.named["pointer"]
    if mutation == "missing_aux":
        c.symbols.pop(4)
    elif mutation.startswith("header_"):
        raw = bytearray(c.data)
        offset, fmt, value = {"header_length": (76, "<I", 19), "header_flags": (96, "<I", DATA_FLAGS | 1),
                              "header_lines": (94, "<H", 1)}[mutation]
        struct.pack_into(fmt, raw, offset, value)
        c.data = bytes(raw)
    elif mutation == "empty":
        section["data"] = b""
    elif mutation in ("code", "execute", "comdat", "info", "remove", "discard", "no_data"):
        section["flags"] = (DATA_FLAGS & ~0x40 if mutation == "no_data" else DATA_FLAGS | {
            "code": 0x20, "execute": 0x20000000, "comdat": 0x1000, "info": 0x200,
            "remove": 0x800, "discard": 0x02000000}[mutation])
    else:
        symbol.update({"common": {"section": 0, "value": 4}, "weak": {"storage": 105},
                       "absolute": {"section": -1}, "aux": {"aux_count": 1},
                       "type": {"type": 0x24}, "storage": {"storage": 6}}[mutation])
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("mutation", ["virtual", "cross_end", "execute", "code", "unreadable", "no_metadata"])
def test_bss_requires_readable_ordinary_complete_file_backing(tmp_path, mutation):
    c, f, b, image = make_case(tmp_path, bss=True)
    assert compare_function(c, f, "sample", b, image)[0]
    data = image.sections[1]
    if mutation == "virtual":
        image.sections[1] = replace(data, rawsize=8)
    elif mutation == "cross_end":
        image.sections[1] = replace(data, end=DATA_VA + 8)
    elif mutation in ("execute", "code", "unreadable"):
        flags = data.flags | ({"execute": 0x20000000, "code": 0x20}.get(mutation, 0))
        image.sections[1] = replace(data, flags=flags & ~0x40000000 if mutation == "unreadable" else flags)
    else:
        image.section_for_va = None
    with pytest.raises(ValueError):
        validate_data_binding(c, "scalar", b, image)


def graph_case(tmp_path, *, separate=False, reverse=False, cyclic=None):
    sections = [{"data": b"\xc3"}]
    symbols = [{"name": "caller", "section": 1, "type": 0x20}]
    b = {}
    for i, name in enumerate(("left", "right")):
        if separate or i == 0:
            sections.append({"name": f".d{i}", "flags": DATA_FLAGS, "data": bytes(4 if separate else 8),
                             "relocations": []})
        section = len(sections)
        symbols.append({"name": name, "section": section, "value": 0 if separate else i * 4})
        b[name] = {"va": hex(DATA_VA + i * 4), "data_size": 4}
    # Scalar left, right -> left is deliberately acyclic; declaration order is irrelevant.
    pointer_section = sections[-1]
    pointer_section["relocations"] = [(0 if separate else 4, 1, 6)]
    if cyclic:
        sections[1]["relocations"].append((0, 1 if cyclic == "self" else 2, 6))
    for i, section in enumerate(sections[1:], 2):
        symbols.append({"name": section["name"], "section": i, "storage": 3,
                        "aux": aux(len(section["data"]), len(section["relocations"]))})
    b["right"]["section_layout"] = True
    if separate:
        b["left"]["section_layout"] = True
    if reverse:
        b = dict(reversed(list(b.items())))
    expected = bytes(4) + struct.pack("<I", DATA_VA)
    if cyclic:
        expected = struct.pack("<I", DATA_VA if cyclic == "self" else DATA_VA + 4) + expected[4:]
    return COFF.from_bytes(coff_bytes(sections, symbols)), entry("caller", TEXT_VA, 1), b, reference(tmp_path, b"\xc3", expected)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("api", ["data", "explicit_data", "fixups", "explicit_fixups", "common", "function", "region"])
def test_one_outer_sweep_does_not_create_false_pointer_cycles(tmp_path, separate, reverse, api):
    c, f, b, image = graph_case(tmp_path, separate=separate, reverse=reverse)
    if api in ("data", "explicit_data"):
        assert validate_data_binding(c, "left", b, image, set() if api.startswith("explicit") else None)[0] == bytes(4)
    elif api in ("fixups", "explicit_fixups"):
        assert apply_fixups(c, b"", [], TEXT_VA, b, image, set() if api.startswith("explicit") else None) == (b"", 0)
    elif api == "common":
        validate_common_definitions(c, b, image)
    elif api == "function":
        assert compare_function(c, f, "sample", b, image)[0]
    else:
        # A complete multi-entry region is an independent root of the same layout context.
        code = b"\xc3\xc3"
        # Keep the existing data objects and native relocation indices unchanged.
        c.sections[0]["data"] = code
        c.named["caller"]["name"] = "first"
        c.named["first"] = c.named.pop("caller")
        c.symbols[99] = {"name": "second", "section": 1, "value": 1, "type": 0x20, "storage": 2, "aux_count": 0}
        c.named["second"] = c.symbols[99]
        image = reference(tmp_path, code, image.read(DATA_VA, 8))
        b.update(first={"va": hex(TEXT_VA)}, second={"va": hex(TEXT_VA + 1)})
        region = {**entry("first", TEXT_VA, 2), "entries": [{"symbol": "first", "offset": 0}, {"symbol": "second", "offset": 1}]}
        assert compare_code_region(c, region, "sample", b, image)[0]


@pytest.mark.parametrize("cyclic", ["self", "mutual"])
def test_actual_pointer_cycles_still_reject(tmp_path, cyclic):
    c, _, b, image = graph_case(tmp_path, cyclic=cyclic)
    with pytest.raises(ValueError, match="Cyclic data binding"):
        validate_data_binding(c, "left", b, image, set())


@pytest.mark.parametrize("api", ["data", "fixups", "function"])
def test_each_independent_call_revalidates_bytes_even_with_visiting(case, api):
    c, f, b, image = case
    def call():
        if api == "data":
            return validate_data_binding(c, "scalar", b, image, set())
        if api == "fixups":
            return apply_fixups(c, b"", [], TEXT_VA, b, image, set())
        return compare_function(c, f, "sample", b, image)
    call()
    c.sections[1]["data"] = b"bad!" + c.sections[1]["data"][4:]
    with pytest.raises(ValueError, match="complete section layout"):
        call()


def test_direct_function_and_region_windows_cover_unnamed_layout_prefix(case):
    c, f, b, image = case
    # This reference deliberately exposes ordinary backing at both windows;
    # layout overlap must fail before any byte comparison can hide the prefix.
    f["sample"]["va"] = hex(DATA_VA)
    with pytest.raises(ValueError, match="Code window overlaps"):
        check(case)
    c.sections[0]["data"] = b"\xc3\xc3"
    c.symbols[99] = {"name": "second", "section": 1, "value": 1, "type": 0x20, "storage": 2, "aux_count": 0}
    c.named["second"] = c.symbols[99]
    b.update(caller={"va": hex(DATA_VA)}, second={"va": hex(DATA_VA + 1)})
    region = {**entry("caller", DATA_VA, 2), "entries": [{"symbol": "caller", "offset": 0}, {"symbol": "second", "offset": 1}]}
    with pytest.raises(ValueError, match="Code window overlaps"):
        CodeRegions(c, [region], "sample", b, image)


@pytest.mark.parametrize("mutation", ["syntax", "origin", "alignment", "metadata", "prefix", "sibling", "overlap"])
@pytest.mark.parametrize("filtered", [False, True])
def test_cli_clears_stale_summaries_for_early_and_late_errors(case, tmp_path, mutation, filtered):
    c, _, b, _ = case
    if mutation == "syntax":
        b["pointer"]["section_layout"] = 1
    elif mutation == "origin":
        b["array"]["va"] = hex(DATA_VA + 13)
    elif mutation == "alignment":
        raw = bytearray(c.data)
        struct.pack_into("<I", raw, 96, DATA_FLAGS | 0x00400000)
        c.data = bytes(raw)
        for binding in b.values():
            binding["va"] = hex(int(binding["va"], 16) + 4)
    elif mutation == "metadata":
        raw = bytearray(c.data)
        # Reparse a truly malformed native auxiliary record through the CLI.
        struct.pack_into("<I", raw, struct.unpack_from("<I", raw, 8)[0] + 5 * 18, 19)
        c.data = bytes(raw)
    elif mutation == "prefix":
        raw = bytearray(c.data)
        raw[struct.unpack_from("<I", raw, 80)[0]] ^= 1
        c.data = bytes(raw)
    elif mutation == "sibling":
        del b["array"]
    else:
        b["other"] = {"va": hex(DATA_VA), "data_size": 1}
    p = project(case, tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    for name in ("report.json", "report.txt"):
        (output / name).write_text("stale success")
    assert main(arguments(p, filtered=filtered)) != 0
    assert not (output / "report.json").exists() and not (output / "report.txt").exists()


def test_candidate_score_uses_actual_native_layout_and_never_returns_zero_for_bad_data(case, tmp_path):
    p = project(case, tmp_path)
    candidate = tmp_path / "Candidate.obj"
    candidate.write_bytes(case[0].data)
    assert score_config(p[0], "sample", "caller", candidate, module_names=["Sample"]) == 0
    raw = bytearray(candidate.read_bytes())
    raw[struct.unpack_from("<I", raw, 80)[0]] ^= 1
    candidate.write_bytes(raw)
    with pytest.raises(ValueError, match="complete section layout"):
        score_config(p[0], "sample", "caller", candidate, module_names=["Sample"])


@pytest.mark.parametrize("kind", ["data", "function", "retained_function", "region", "layout"])
def test_cross_module_footprints_include_the_unnamed_prefix(case, tmp_path, kind):
    c, _, b, image = case
    _, _, modules, targets = project(case, tmp_path)
    if kind == "layout":
        # Counted definition intervals are disjoint; the full prefixes overlap.
        symbols = [{"name": "item", "section": 1, "value": 8},
                   {"name": ".other", "section": 1, "storage": 3, "aux": aux(12)}]
        raw = coff_bytes([{"name": ".other", "flags": DATA_FLAGS, "data": bytes(12)}], symbols)
        other = {"name": "Other", "bindings": {"sample": {
            "item": {"va": hex(DATA_VA + 2), "data_size": 4, "section_layout": True}}}}
        # The second layout origin is six bytes before the first contribution.
        # Reference metadata must contain that origin to reach overlap preflight.
        for binding in b.values():
            binding["va"] = hex(int(binding["va"], 16) + 8)
        other["bindings"]["sample"]["item"]["va"] = hex(DATA_VA + 8)
    else:
        raw = coff_bytes([{"data": b"\xc3\xc3"}], [
            {"name": "first", "section": 1, "type": 0x20},
            {"name": "second", "section": 1, "value": 1, "type": 0x20}])
        other = {"name": "Other", "bindings": {"sample": {}}}
        if kind == "data":
            other["bindings"]["sample"]["foreign"] = {"va": hex(DATA_VA), "data_size": 1}
        elif kind == "region":
            other["code_regions"] = [{**entry("first", DATA_VA, 2), "entries": [
                {"symbol": "first", "offset": 0}, {"symbol": "second", "offset": 1}]}]
        else:
            other["_ordinary_code_owners" if kind == "retained_function" else "functions"] = [entry("first", DATA_VA, 1)]
            if kind == "retained_function":
                other["bindings"]["sample"]["first"] = {"va": hex(DATA_VA)}
    (tmp_path / "Other.obj").write_bytes(raw)
    with pytest.raises(ValueError, match="overlap|Overlapping"):
        report(modules + [other], targets, ["sample"], tmp_path / "report")


def test_duplicate_resolved_native_contribution_cannot_get_a_second_origin(tmp_path):
    case = make_case(tmp_path, bss=True)
    _, _, modules, targets = project(case, tmp_path)
    other = deepcopy(modules[0])
    other["name"] = "Other"
    other["functions"] = []
    for binding in other["bindings"]["sample"].values():
        binding["va"] = hex(int(binding["va"], 16) + 12)
    (tmp_path / "Other.obj").symlink_to(tmp_path / "Sample.obj")
    with pytest.raises(ValueError, match="Duplicate native section layout"):
        report(modules + [other], targets, ["sample"], tmp_path / "report")


@pytest.mark.parametrize("filtered", [False, True])
def test_full_and_filtered_reports_keep_unselected_function_ownership(case, tmp_path, filtered):
    _, _, modules, targets = project(case, tmp_path)
    modules[0]["functions"].append(entry("unused", DATA_VA, 1))
    # A missing unselected native entry cannot hide this complete original owner.
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    p = (tmp_path / "config.json",)
    assert main(arguments(p, filtered=filtered)) != 0
    assert not (tmp_path / "report" / "report.json").exists()


@pytest.mark.parametrize("mutation", [None, "common_flag", "provider_name", "provider_section"])
def test_provider_namespace_cannot_borrow_consumer_layout(tmp_path, mutation):
    # Consumer layout and COMMON provider deliberately use the same section index.
    sections = [{"data": b"\xc3"}, {"name": ".data", "flags": DATA_FLAGS, "data": b"unit"}]
    symbols = [{"name": "caller", "section": 1, "type": 0x20},
               {"name": "owned", "section": 2, "storage": 3},
               {"name": "buffer", "value": 4},
               {"name": ".data", "section": 2, "storage": 3, "aux": aux(4)}]
    c = COFF.from_bytes(coff_bytes(sections, symbols))
    provider_symbols = [{"name": "buffer", "section": 2}]
    if mutation == "provider_name":
        provider_symbols.append({"name": "owned", "section": 2, "storage": 3})
    if mutation == "provider_section":
        provider_symbols += [{"name": "owned", "section": 1, "storage": 3},
                             {"name": ".data", "section": 1, "storage": 3, "aux": aux(4)}]
    provider = COFF.from_bytes(coff_bytes([
        {"name": ".data", "flags": DATA_FLAGS, "data": b"unit"},
        {"name": ".value", "flags": DATA_FLAGS, "data": b"word"}], provider_symbols))
    b = {"owned": {"va": hex(DATA_VA), "data_size": 4, "section_layout": True},
         "buffer": {"va": hex(DATA_VA + 4), "data_size": 4}}
    if mutation == "common_flag":
        b["buffer"]["section_layout"] = True
    image = reference(tmp_path, b"\xc3", b"unitword")
    if mutation is None:
        assert compare_function(c, entry("caller", TEXT_VA, 1), "sample", b, image,
                                common_definitions={"buffer": provider})[0]
    else:
        with pytest.raises(ValueError, match="Provider-owned|ordinary native definition"):
            compare_function(c, entry("caller", TEXT_VA, 1), "sample", b, image,
                             common_definitions={"buffer": provider})


@pytest.mark.parametrize("mutation", ["origin", "metadata", "prefix", "sibling", "syntax"])
def test_direct_report_also_removes_stale_summaries(case, tmp_path, mutation):
    c, _, b, _ = case
    _, _, modules, targets = project(case, tmp_path)
    if mutation == "origin":
        b["scalar"]["va"] = hex(DATA_VA + 5)
    elif mutation == "sibling":
        del b["array"]
    elif mutation == "syntax":
        b["pointer"]["section_layout"] = False
    else:
        raw = bytearray(c.data)
        offset = (struct.unpack_from("<I", raw, 8)[0] + 5 * 18 if mutation == "metadata"
                  else struct.unpack_from("<I", raw, 80)[0])
        raw[offset] ^= 1
        (tmp_path / "Sample.obj").write_bytes(raw)
    output = tmp_path / "report"
    output.mkdir()
    for name in ("report.json", "report.txt"):
        (output / name).write_text("stale success")
    with pytest.raises(ValueError):
        report(modules, targets, ["sample"], output)
    assert not (output / "report.json").exists() and not (output / "report.txt").exists()


def test_data_only_report_preserves_definition_counts(tmp_path):
    case = make_case(tmp_path, bss=True)
    _, _, modules, targets = project(case, tmp_path)
    modules[0]["functions"] = []
    assert report(modules, targets, ["sample"], tmp_path / "report") == 0
    result = json.loads((tmp_path / "report" / "report.json").read_text())
    assert result["code"] == []
    assert len(result["data"]) == 3 and sum(row["size"] for row in result["data"]) == 12
    assert all(row["checked_fixups"] == 0 for row in result["data"])


def test_score_checks_candidate_metadata_not_only_baseline_object(case, tmp_path):
    p = project(case, tmp_path)
    c = deepcopy(case[0])
    raw = bytearray(c.data)
    # Shift only the candidate's native pointer offset, leaving its image bytes intact.
    symbol_table = struct.unpack_from("<I", raw, 8)[0]
    struct.pack_into("<I", raw, symbol_table + 2 * 18 + 8, 9)
    candidate = tmp_path / "Candidate.obj"
    candidate.write_bytes(raw)
    with pytest.raises(ValueError, match="extent differs|disagree"):
        score_config(p[0], "sample", "caller", candidate, module_names=["Sample"])


@pytest.mark.parametrize("reverse", [False, True])
def test_actual_region_dependency_and_unrelated_layout_pointer_paths(tmp_path, reverse):
    caller = b"\xe8" + bytes(4) + b"\xc3"
    region_code = b"\xa1" + bytes(4) + b"\xc3\xc3"
    c = COFF.from_bytes(coff_bytes([
        {"data": caller, "relocations": [(1, 1, 0x14)]},
        {"data": region_code, "relocations": [(1, 4, 6)]},
        {"name": ".left", "flags": DATA_FLAGS, "data": bytes(4)},
        {"name": ".right", "flags": DATA_FLAGS, "data": bytes(4), "relocations": [(0, 3, 6)]},
    ], [{"name": "caller", "section": 1, "type": 0x20},
        {"name": "first", "section": 2, "type": 0x20},
        {"name": "second", "section": 2, "value": 6, "type": 0x20},
        {"name": "left", "section": 3}, {"name": "right", "section": 4},
        {"name": ".left", "section": 3, "storage": 3, "aux": aux(4)},
        {"name": ".right", "section": 4, "storage": 3, "aux": aux(4, 1)}]))
    original_code = b"\xe8" + struct.pack("<i", 16 - 5) + b"\xc3" + bytes(10)
    original_code += b"\xa1" + struct.pack("<I", DATA_VA + 4) + b"\xc3\xc3"
    image = reference(tmp_path, original_code, bytes(4) + struct.pack("<I", DATA_VA))
    b = {"left": {"va": hex(DATA_VA), "data_size": 4, "section_layout": True},
         "right": {"va": hex(DATA_VA + 4), "data_size": 4, "section_layout": True},
         "first": {"va": hex(TEXT_VA + 16)}, "second": {"va": hex(TEXT_VA + 22)}}
    if reverse:
        b = dict(reversed(list(b.items())))
    region = {**entry("first", TEXT_VA + 16, 7), "entries": [
        {"symbol": "first", "offset": 0}, {"symbol": "second", "offset": 6}]}
    context = CodeRegions(c, [region], "sample", b, image)
    assert compare_function(c, entry("caller", TEXT_VA, 6), "sample", b, image, code_regions=context)[0]
    # An explicit nonempty definition path cannot bypass a new outer check.
    assert apply_fixups(c, b"", [], TEXT_VA, b, image, {(id(c), "unrelated")}, code_regions=context) == (b"", 0)
    c.sections[2]["data"] = b"oops"
    with pytest.raises(ValueError, match="Rebuilt data differs"):
        apply_fixups(c, b"", [], TEXT_VA, b, image, {(id(c), "unrelated")}, code_regions=context)


def invalid_prefix_case(tmp_path):
    c, f, b, image = make_case(tmp_path)
    raw = bytearray(c.data)
    raw[struct.unpack_from("<I", raw, 80)[0]] ^= 1
    return COFF.from_bytes(raw), f, b, image


class CallbackReference:
    """Delegate original metadata/bytes while invoking a caller-owned callback."""

    def __init__(self, image, callback):
        self.image = image
        self.callback = callback
        self.reads = 0

    def section_for_va(self, *args):
        return self.image.section_for_va(*args)

    def read(self, *args):
        self.reads += 1
        if self.reads == 2:
            self.callback()
        return self.image.read(*args)


def public_layout_call(api, case):
    from binary_comp.core.coff import code_window, relocate
    c, f, b, image = case
    if api == "function":
        return compare_function(c, f, "sample", b, image)
    if api == "data":
        return validate_data_binding(c, "scalar", b, image, set())
    if api == "common":
        return validate_common_definitions(c, b, image)
    if api == "fixups":
        return apply_fixups(c, b"", [], TEXT_VA, b, image, set())
    if api == "window":
        return code_window(c, c.function("caller"), TEXT_VA, 6, b, image)
    return relocate(c, c.function("caller"), TEXT_VA, 6, b, image)


@pytest.mark.parametrize("same_inputs", [False, True])
@pytest.mark.parametrize("api", ["function", "data", "common", "fixups", "window", "relocate"])
def test_independent_public_callback_entry_gets_a_fresh_layout_sweep(tmp_path, same_inputs, api):
    inner_dir, outer_dir = tmp_path / "inner", tmp_path / "outer"
    inner_dir.mkdir()
    outer_dir.mkdir()
    inner = invalid_prefix_case(inner_dir)
    outer = inner if same_inputs else make_case(outer_dir)
    observed = []

    def callback():
        try:
            public_layout_call(api, (*inner[:3], proxy if same_inputs else inner[3]))
        except ValueError as error:
            observed.append(str(error))
        else:
            observed.append("accepted invalid prefix")

    proxy = CallbackReference(outer[3], callback)
    if same_inputs:
        with pytest.raises(ValueError, match="complete section layout"):
            compare_function(*outer[:2], "sample", outer[2], proxy)
    else:
        assert compare_function(*outer[:2], "sample", outer[2], proxy)[0]
    assert len(observed) == 1 and "complete section layout" in observed[0]


@pytest.mark.parametrize("same_inputs", [False, True])
def test_copied_async_context_cannot_reuse_an_expired_layout_operation(tmp_path, same_inputs):
    import asyncio

    async def run():
        inner_dir, outer_dir = tmp_path / "inner", tmp_path / "outer"
        inner_dir.mkdir()
        outer_dir.mkdir()
        inner = invalid_prefix_case(inner_dir)
        outer = inner if same_inputs else make_case(outer_dir)
        tasks = []

        async def later():
            with pytest.raises(ValueError, match="complete section layout"):
                compare_function(*inner[:2], "sample", inner[2], proxy if same_inputs else inner[3])

        proxy = CallbackReference(outer[3], lambda: tasks.append(asyncio.create_task(later())))
        if same_inputs:
            with pytest.raises(ValueError, match="complete section layout"):
                compare_function(*outer[:2], "sample", outer[2], proxy)
        else:
            assert compare_function(*outer[:2], "sample", outer[2], proxy)[0]
        assert len(tasks) == 1
        await tasks[0]

    asyncio.run(run())


@pytest.mark.parametrize("api", ["region", "region_compare", "region_target"])
@pytest.mark.parametrize("same_inputs", [False, True])
def test_public_region_callback_entries_cannot_inherit_native_dispatch(tmp_path, api, same_inputs):
    inner_dir, outer_dir = tmp_path / "inner", tmp_path / "outer"
    inner_dir.mkdir()
    outer_dir.mkdir()
    c, _, b, image = invalid_prefix_case(inner_dir)
    symbols = [{**symbol, "aux": c.auxiliary_records.get(index, b"")}
               for index, symbol in c.symbols.items()]
    symbols.append({"name": "second", "section": 1, "type": 0x20, "value": 6})
    c = COFF.from_bytes(coff_bytes([{**c.sections[0], "data": c.sections[0]["data"] + b"\xc3"},
                                  c.sections[1]], symbols))
    image = reference(inner_dir, image.read(TEXT_VA, 6) + b"\xc3", image.read(DATA_VA, 20))
    b.update(caller={"va": hex(TEXT_VA)}, second={"va": hex(TEXT_VA + 6)})
    region = {**entry("caller", TEXT_VA, 7), "entries": [
        {"symbol": "caller", "offset": 0}, {"symbol": "second", "offset": 6}]}
    context = CodeRegions(c, [region], "sample", b, image)
    observed = []

    def callback():
        try:
            if api == "region":
                compare_code_region(c, region, "sample", b, context.reference, code_regions=context)
            elif api == "region_compare":
                context.compare("caller", set())
            else:
                context.validate_target(c, c.named["caller"], TEXT_VA, set())
        except ValueError as error:
            observed.append(str(error))
        else:
            observed.append("accepted invalid prefix")

    if same_inputs:
        proxy = CallbackReference(image, callback)
        context.reference = proxy
        with pytest.raises(ValueError, match="complete section layout"):
            context.compare("caller")
    else:
        outer = make_case(outer_dir)
        proxy = CallbackReference(outer[3], callback)
        assert compare_function(*outer[:2], "sample", outer[2], proxy)[0]
    assert len(observed) == 1 and "complete section layout" in observed[0]


@pytest.mark.parametrize("with_provider", [False, True])
def test_derived_inline_region_context_keeps_acyclic_layouts_and_providers(tmp_path, with_provider):
    from test_coff_inline_alignment import switch_fixture

    original, function, b, image = switch_fixture(tmp_path)
    symbols = list(original.symbols.values()) + [
        {"name": "left", "section": 2}, {"name": "right", "section": 3},
        {"name": "first", "section": 4, "type": 0x20},
        {"name": "second", "section": 4, "type": 0x20, "value": 2},
        {"name": ".left", "section": 2, "storage": 3, "aux": aux(4)},
        {"name": ".right", "section": 3, "storage": 3, "aux": aux(4, 1)}]
    if with_provider:
        symbols.append({"name": "buffer", "value": 4})
    region_code = b"\xc3\x90\xc3\x90"
    c = COFF.from_bytes(coff_bytes([*original.sections,
        {"name": ".left", "flags": DATA_FLAGS, "data": bytes(4)},
        {"name": ".right", "flags": DATA_FLAGS, "data": bytes(4), "relocations": [(0, 4, 6)]},
        {"data": region_code}], symbols))
    code = image.read(TEXT_VA, 32) + bytes(96) + region_code
    image = reference(tmp_path, code, bytes(4) + struct.pack("<I", DATA_VA) + b"unit")
    b.update(left={"va": hex(DATA_VA), "data_size": 4, "section_layout": True},
             right={"va": hex(DATA_VA + 4), "data_size": 4, "section_layout": True},
             first={"va": hex(TEXT_VA + 128)}, second={"va": hex(TEXT_VA + 130)})
    # Force the existing checked ordinary inline owner/region context transition.
    del b["dispatch"]["size"]
    providers = None
    if with_provider:
        b["buffer"] = {"va": hex(DATA_VA + 8), "data_size": 4}
        providers = {"buffer": COFF.from_bytes(coff_bytes([
            {"name": ".data", "flags": DATA_FLAGS, "data": b"unit"}], [
            {"name": "buffer", "section": 1}]))}
    region = {**entry("first", TEXT_VA + 128, 4), "entries": [
        {"symbol": "first", "offset": 0}, {"symbol": "second", "offset": 2}]}
    context = CodeRegions(c, [region], "sample", b, image, common_definitions=providers)
    assert compare_function(c, function, "sample", b, image,
                            common_definitions=providers, code_regions=context)[0]
