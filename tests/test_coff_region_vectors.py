"""Explicit interleaved native vectors, using only authored synthetic records."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import pytest

from binary_comp.cli import main
from binary_comp.analyzers.coff_report import register_external_bindings
from binary_comp.core.coff import (COFF, CodeRegions, apply_fixups, compare_code_region, compare_function,
                                   validate_common_definitions, validate_data_binding)
from binary_comp.core.disasm import disassemble_raw
from binary_comp.core.pe import FileBackedPEImage
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_contract import coff_bytes, entry, DATA_FLAGS


def make_vectors(tmp_path, *, auxiliary=False):
    code = bytearray(b"\x90" * 160)
    code[3:13] = b"\xb8" + struct.pack("<I", 0xfffffffc) + b"\xe9" + bytes(4)
    code[16:24] = bytes(8)
    code[24:30] = b"\xb9" + struct.pack("<I", 4) + b"\xc3"
    code[64:68] = bytes(4)
    code[68:70] = b"\x42\xc3"
    code[143:146] = b"\xb0\x07\xc3"
    symbols = [{"name": "first", "section": 1, "value": 3, "type": 0x20},
               {"name": "alias", "section": 1, "value": 3, "type": 0x24, "storage": 3},
               {"name": "second", "section": 1, "value": 24, "type": 0x20, "storage": 3},
               {"name": "last", "section": 1, "value": 143, "type": 0x20, "storage": 3},
               {"name": "vector_a", "section": 1, "value": 16, "storage": 3},
               {"name": "vector_b", "section": 1, "value": 64, "storage": 3},
               {"name": "after_b", "section": 1, "value": 68, "storage": 6},
               {"name": "caller", "section": 2, "type": 0x20},
               {"name": "pointer", "section": 3}]
    if auxiliary:
        symbols[3]["aux"] = bytes(18)
    indices, index = {}, 0
    for symbol in symbols:
        indices[symbol["name"]] = index
        index += 1 + len(symbol.get("aux", b"")) // 18
    relocations = [(offset, indices[name], kind) for offset, name, kind in
                   [(4, "vector_a", 6), (9, "second", 0x14), (16, "last", 6),
                    (20, "second", 6), (25, "vector_b", 6), (64, "first", 6)]]
    coff = COFF.from_bytes(coff_bytes([
        {"data": bytes(code), "relocations": relocations},
        {"data": b"\xe8" + bytes(4) + b"\xc3", "relocations": [(1, 0, 0x14)]},
        {"name": ".data", "data": bytes(4), "flags": DATA_FLAGS, "relocations": [(0, 0, 6)]},
    ], symbols))
    original = bytearray(code + b"\x90" * 96)
    for offset, value in [(4, TEXT_VA + 12), (9, 11), (16, TEXT_VA + 143),
                          (20, TEXT_VA + 24), (25, TEXT_VA + 68), (64, TEXT_VA + 3)]:
        struct.pack_into("<I", original, offset, value)
    original += b"\xe8" + struct.pack("<i", 3 - 261) + b"\xc3"
    path = tmp_path / "original.exe"
    write_tiny_pe(path, original, {0: struct.pack("<I", TEXT_VA + 3)})
    image = FileBackedPEImage(path)
    region = {"label": "shared vector contribution", "symbol": "first", "anchor_offset": 3,
              "entries": [{"symbol": s["name"], "offset": s["value"]} for s in symbols[:4]],
              "inline_vectors": [{"symbol": "vector_a", "size": 8},
                                 {"symbol": "vector_b", "size": 4}],
              "retail": {"va": hex(TEXT_VA), "size": 160}}
    bindings = {s["name"]: {"va": hex(TEXT_VA + s.get("value", 0))} for s in symbols[:7]}
    bindings["pointer"] = {"va": hex(DATA_VA), "data_size": 4}
    return coff, region, bindings, image


@pytest.fixture
def vectors(tmp_path):
    return make_vectors(tmp_path)


def context(fixture):
    c, r, b, image = fixture
    return CodeRegions(c, [r], "retail", b, image)


def check(fixture, ctx=None):
    c, r, b, image = fixture
    return compare_code_region(c, r, "retail", b, image, code_regions=ctx)


def project(fixture, tmp_path):
    c, r, b, image = fixture
    objects = tmp_path / "objects"
    objects.mkdir()
    records = [{**s, "aux": c.auxiliary_records.get(i, b"")} for i, s in c.symbols.items()]
    (objects / "Sample.obj").write_bytes(coff_bytes(c.sections, records))
    caller = {"label": "caller", "symbol": "caller",
              "retail": {"va": hex(TEXT_VA + 256), "size": 6}}
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


def args(p, *extra):
    return ["report", "--config", str(p[0]), "--target", "retail", "--no-build",
            "--output", str(p[2]), *extra]


def mutate_byte(fixture, offset, value, *, both=False):
    c, _, _, image = fixture
    raw = bytearray(c.sections[0]["data"])
    raw[offset] = value
    c.sections[0]["data"] = bytes(raw)
    if both:
        data = bytearray(image.data)
        backing = image.section_for_va(TEXT_VA)
        data[backing.rawptr + offset] = value
        image.data = bytes(data)
        Path(image.path).write_bytes(image.data)


def test_complete_vectors_keep_every_byte_fixup_entry_and_padding(vectors):
    ctx = context(vectors)
    exact, actual, original, fixups = check(vectors, ctx)
    assert exact and actual == original and len(actual) == 160 and fixups == 6
    assert actual[146:] == b"\x90" * 14
    # The relocated pointer starts with an invalid POP /2 encoding. A complete
    # linear decode fails; the two explicitly owned vectors are real data.
    assert sum(i.size for i in disassemble_raw(actual, TEXT_VA)) < len(actual)
    decoded, spans = ctx.decode_vectors("first", original, actual)
    assert spans == [{"symbol": "vector_a", "offset": 16, "size": 8, "checked_fixups": 2},
                     {"symbol": "vector_b", "offset": 64, "size": 4, "checked_fixups": 1}]
    assert {k: sum(i.size for i in ins) for k, ins in decoded.items()} == {"original": 148, "rebuilt": 148}
    assert decoded["rebuilt"][-1].address == TEXT_VA + 159
    assert {i.address for i in decoded["rebuilt"]} >= {TEXT_VA + e["offset"] for e in vectors[1]["entries"]}
    assert validate_data_binding(vectors[0], "pointer", vectors[2], vectors[3], code_regions=ctx)[1] == 1


def test_report_splits_instruction_data_counts_without_new_rows(vectors, tmp_path):
    p = project(vectors, tmp_path)
    assert main(args(p)) == 0
    report = json.loads((p[2] / "report.json").read_text())
    assert [r["kind"] for r in report["code"]] == ["function", "region"]
    row = report["code"][1]
    assert row["body_size"] == 160 and row["instruction_bytes"] == 148 and row["embedded_data_bytes"] == 12
    assert row["decoded_bytes"] == {"original": 148, "rebuilt": 148}
    assert row["checked_fixups"] == 6 and row["excluded_nop_padding"] == 0
    assert row["relocated_bytes"]["exact"] and not row["raw_bytes"]["exact"]
    assert row["raw_bytes"]["original_size"] == row["relocated_bytes"]["original_size"] == 160
    assert len(row["entries"]) == 4 and row["mnemonic_similarity"] == 100
    assert len(report["data"]) == 1 and report["data"][0]["symbol"] == "pointer"
    review = (p[2] / row["review"]).read_text()
    assert "160 bytes; 148 instruction bytes; 12 embedded vector bytes" in review
    assert "Native vector vector_b" in review


def test_defaults_and_ordinary_function_contracts_are_not_relaxed(vectors, tmp_path):
    c, r, b, image = vectors
    with pytest.raises(ValueError, match="section offset zero"):
        c.function("first")
    function = entry("caller", TEXT_VA + 256, 6)
    function["inline_vectors"] = r["inline_vectors"]
    with pytest.raises(ValueError, match="only supported for code_regions"):
        compare_function(c, function, "sample", b, image)
    del r["inline_vectors"]
    assert check(vectors)[0]  # The default core comparison remains byte-only.
    p = project(vectors, tmp_path)
    assert main(args(p)) != 0
    row = json.loads((p[2] / "report.json").read_text())["code"][1]
    assert row["relocated_bytes"]["exact"] and "metric_error" in row
    assert row["mnemonic_similarity"] is None and "instruction_bytes" not in row


def test_typed_auxiliary_entries_and_aliases_remain_exhaustive(tmp_path):
    fixture = make_vectors(tmp_path, auxiliary=True)
    assert fixture[0].named["last"]["aux_count"] == 1 and check(fixture)[0]
    fixture[1]["entries"] = [e for e in fixture[1]["entries"] if e["symbol"] != "alias"]
    with pytest.raises(ValueError, match="typed records"):
        check(fixture)


@pytest.mark.parametrize("records", [None, False, True, {}, [], [False], [{}],
    [{"symbol": "", "size": 4}], [{"symbol": "vector_a", "size": True}],
    [{"symbol": "vector_a", "size": False}], [{"symbol": "vector_a", "size": 0}],
    [{"symbol": "vector_a", "size": -4}], [{"symbol": "vector_a", "size": 5}],
    [{"symbol": "vector_a", "size": 8.0}], [{"symbol": "vector_a", "size": "8"}],
    [{"symbol": "vector_a", "size": 8, "offset": 16}],
    [{"symbol": "vector_a", "size": 8}, {"symbol": "vector_a", "size": 8}]])
def test_schema_is_eager_in_core_and_filtered_cli_and_clears_stale_success(vectors, tmp_path, records):
    p = project(vectors, tmp_path)
    assert main(args(p)) == 0
    reviews = list(p[2].glob("*/*/*-region.txt"))
    assert len(reviews) == 1
    vectors[1]["inline_vectors"] = records
    with pytest.raises(ValueError, match="inline_vectors"):
        context(vectors)
    p[1].write_text(json.dumps(p[3]))
    assert main(["compare", "caller", "--config", str(p[0]), "--target", "retail",
                 "--no-build", "--output", str(p[2])]) == 2
    assert not (p[2] / "report.json").exists() and not (p[2] / "report.txt").exists()
    assert not reviews[0].exists()


@pytest.mark.parametrize("role", ["function", "label", "binding"])
def test_region_metadata_cannot_be_reinterpreted_as_another_role(vectors, tmp_path, role):
    p = project(vectors, tmp_path)
    if role == "binding":
        vectors[2]["unused"] = {"va": "0x500000", "inline_vectors": vectors[1]["inline_vectors"]}
        for operation in [lambda: context(vectors),
                          lambda: apply_fixups(vectors[0], b"\xc3", [], TEXT_VA, vectors[2], vectors[3]),
                          lambda: validate_common_definitions(vectors[0], vectors[2], vectors[3]),
                          lambda: validate_data_binding(vectors[0], "pointer", vectors[2], vectors[3])]:
            with pytest.raises(ValueError, match="only supported for code_regions"):
                operation()
    else:
        record = p[3][0]["functions"][0]
        record["inline_vectors"] = vectors[1]["inline_vectors"]
        if role == "label":
            p[3][0]["code_labels"] = [record]
            p[3][0]["functions"] = []
    p[1].write_text(json.dumps(p[3]))
    assert main(args(p)) == 2


@pytest.mark.parametrize("mutation", ["missing", "foreign", "typed", "local", "auxiliary", "common",
                                      "unaligned", "partial", "oversized", "shifted binding", "missing binding",
                                      "data role", "code role", "overlap", "typed entry inside"])
def test_vector_ownership_is_complete_native_and_address_only(vectors, mutation):
    c, r, b, _ = vectors
    symbol = c.named["vector_a"]
    if mutation == "missing":
        del c.named["vector_a"]
    elif mutation == "foreign":
        symbol["section"] = 2
    elif mutation == "typed":
        symbol["type"] = 0x20
    elif mutation == "local":
        symbol["storage"] = 6
    elif mutation == "auxiliary":
        symbol["aux_count"] = 1
    elif mutation == "common":
        symbol.update(section=0, value=8, storage=2)
    elif mutation == "unaligned":
        symbol["value"] = 17
        b["vector_a"]["va"] = hex(TEXT_VA + 17)
    elif mutation in ("partial", "oversized"):
        r["inline_vectors"][0]["size"] = 4 if mutation == "partial" else 12
    elif mutation == "shifted binding":
        b["vector_a"]["va"] = hex(TEXT_VA + 20)
    elif mutation == "missing binding":
        del b["vector_a"]
    elif mutation == "data role":
        b["vector_a"]["data_size"] = 8
    elif mutation == "code role":
        b["vector_a"]["size"] = 8
    elif mutation == "overlap":
        c.named["vector_b"]["value"] = 16
        b["vector_b"]["va"] = hex(TEXT_VA + 16)
        r["inline_vectors"][1]["size"] = 8
    else:
        c.named["second"]["value"] = 20
        next(e for e in r["entries"] if e["symbol"] == "second")["offset"] = 20
        b["second"]["va"] = hex(TEXT_VA + 20)
    with pytest.raises((ValueError, KeyError)):
        check(vectors)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unsupported", "cross start", "cross end",
                                      "addend", "foreign target", "local target", "untyped target",
                                      "section shortcut", "weak shortcut", "common target",
                                      "undefined typed shortcut", "invalid index"])
def test_vector_slots_cannot_hide_bytes_fixups_or_target_roles(vectors, mutation):
    c = vectors[0]
    section = c.sections[0]
    at = next(i for i, row in enumerate(section["relocations"]) if row[0] == 16)
    if mutation == "missing":
        section["relocations"].pop(at)
    elif mutation == "duplicate":
        section["relocations"].append(section["relocations"][at])
    elif mutation == "unsupported":
        section["relocations"][at] = (16, 3, 0x14)
    elif mutation == "cross start":
        section["relocations"].append((15, 3, 6))
    elif mutation == "cross end":
        section["relocations"][at + 1] = (22, 2, 6)
    elif mutation == "addend":
        mutate_byte(vectors, 16, 1)
    elif mutation == "foreign target":
        section["relocations"][at] = (16, 7, 6)
    elif mutation == "local target":
        section["relocations"][at] = (16, 6, 6)
    elif mutation == "invalid index":
        section["relocations"][at] = (16, 999, 6)
    else:
        native = {"name": "first", "value": 0, "section": 1, "type": 0,
                  "storage": 3, "aux_count": 0}
        if mutation == "section shortcut":
            native["aux_count"] = 1
        elif mutation == "weak shortcut":
            native.update(name="weak_entry", section=0, type=0x20, storage=105,
                          aux_count=1, weak_fallback=0, weak_search=3)
            c.named["weak_entry"] = native
        elif mutation == "common target":
            native.update(name="common", section=0, value=4, storage=2)
            c.named["common"] = native
        elif mutation == "undefined typed shortcut":
            native.update(section=0, type=0x20, storage=2)
        c.symbols[100] = native
        section["relocations"][at] = (16, 100, 6)
    with pytest.raises(ValueError, match="vector|symbol index"):
        check(vectors)


@pytest.mark.parametrize("offset", [0, 14, 31, 70, 146, 159])
def test_code_and_padding_bytes_remain_in_the_complete_gate(vectors, offset):
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    mutate_byte(vectors, offset, 0x91)
    assert not check(vectors, ctx)[0]


def test_original_vector_payload_is_never_excluded_from_equality(vectors):
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    image = vectors[3]
    raw = bytearray(image.data)
    raw[image.section_for_va(TEXT_VA).rawptr + 16] ^= 1
    image.data = bytes(raw)
    assert not check(vectors, ctx)[0]


@pytest.mark.parametrize("offset", [15, 63, 159])
def test_any_incomplete_complement_range_rejects_even_identical_full_bytes(vectors, offset):
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    mutate_byte(vectors, offset, 0x0f, both=True)
    with pytest.raises(ValueError, match="did not cover every"):
        check(vectors, ctx)
    with pytest.raises(ValueError, match="did not cover every"):
        compare_function(vectors[0], entry("caller", TEXT_VA + 256, 6), "sample", vectors[2], vectors[3],
                         code_regions=ctx)
    with pytest.raises(ValueError, match="did not cover every"):
        validate_data_binding(vectors[0], "pointer", vectors[2], vectors[3], code_regions=ctx)


def test_typed_entries_must_start_instructions_in_both_windows(vectors):
    c, r, b, _ = vectors
    # Move one complete typed entry to the immediate of MOV AL,7. All bytes,
    # vector targets and bindings still agree, but it is not an instruction.
    c.named["last"]["value"] = 144
    next(e for e in r["entries"] if e["symbol"] == "last")["offset"] = 144
    b["last"]["va"] = hex(TEXT_VA + 144)
    image = vectors[3]
    raw = bytearray(image.data)
    struct.pack_into("<I", raw, image.section_for_va(TEXT_VA).rawptr + 16, TEXT_VA + 144)
    image.data = bytes(raw)
    with pytest.raises(ValueError, match="instruction boundary"):
        check(vectors)


@pytest.mark.parametrize("mutation", ["vector value", "vector type", "entry value", "section", "relocation"])
def test_reused_context_revalidates_native_metadata(vectors, mutation):
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    c = vectors[0]
    if mutation == "vector value":
        c.named["vector_a"]["value"] += 4
    elif mutation == "vector type":
        c.named["vector_a"]["type"] = 0x20
    elif mutation == "entry value":
        c.named["last"]["value"] += 1
    elif mutation == "section":
        c.sections.append(deepcopy(c.sections[0]))
        c.named["vector_a"]["section"] = 4
    else:
        c.sections[0]["relocations"] = [r for r in c.sections[0]["relocations"] if r[0] != 16]
    with pytest.raises(ValueError):
        check(vectors, ctx)


def test_cli_filter_score_and_stale_reviews_cannot_bypass_complement_validation(vectors, tmp_path):
    p = project(vectors, tmp_path)
    assert main(args(p)) == 0
    row = json.loads((p[2] / "report.json").read_text())["code"][1]
    review = p[2] / row["review"]
    score = ["compare", "caller", "--config", str(p[0]), "--target", "retail", "--no-build", "--score",
             "--object", str(p[0].parent / "objects/Sample.obj")]
    assert main(score) == 0
    mutate_byte(vectors, 63, 0x0f, both=True)
    records = [{**s, "aux": vectors[0].auxiliary_records.get(i, b"")}
               for i, s in vectors[0].symbols.items()]
    (p[0].parent / "objects/Sample.obj").write_bytes(coff_bytes(vectors[0].sections, records))
    config = json.loads(p[0].read_text())
    config["targets"]["retail"]["original_sha256"] = hashlib.sha256(Path(vectors[3].path).read_bytes()).hexdigest()
    p[0].write_text(json.dumps(config))
    assert main(args(p)) != 0
    assert not review.exists()
    assert main(["compare", "caller", "--config", str(p[0]), "--target", "retail",
                 "--no-build", "--output", str(p[2])]) != 0
    assert main(score) == 2
    assert main(["compare", "first", "--config", str(p[0]), "--target", "retail", "--no-build", "--score",
                 "--object", str(p[0].parent / "objects/Sample.obj")]) == 2


@pytest.mark.parametrize("mutation", ["short extent", "long extent", "native nonexecute",
                                      "original nonexecute", "virtual tail", "truncated backing"])
def test_vector_opt_in_preserves_complete_executable_file_backed_owner(vectors, mutation):
    c, r, _, image = vectors
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    if mutation in ("short extent", "long extent"):
        r["retail"]["size"] += -1 if mutation == "short extent" else 1
    elif mutation == "native nonexecute":
        c.sections[0]["flags"] &= ~0x20000000
    elif mutation == "truncated backing":
        section = image.section_for_va(TEXT_VA)
        image.data = image.data[:section.rawptr + 159]
    else:
        section = image.section_for_va(TEXT_VA)
        image.sections[0] = replace(section, **({"flags": section.flags & ~0x20000000}
                                                if mutation == "original nonexecute" else {"rawsize": 159}))
    with pytest.raises(ValueError):
        check(vectors)


@pytest.mark.parametrize("window", ["original", "rebuilt"])
def test_each_window_must_decode_independently(vectors, window):
    ctx = context(vectors)
    if window == "rebuilt":
        mutate_byte(vectors, 63, 0x0f)
    else:
        image = vectors[3]
        raw = bytearray(image.data)
        raw[image.section_for_va(TEXT_VA).rawptr + 63] = 0x0f
        image.data = bytes(raw)
    with pytest.raises(ValueError, match=f"every {window} instruction byte"):
        ctx.compare("first")


@pytest.mark.parametrize("mutation", ["vector extent", "vector binding", "entry offset", "origin"])
def test_context_owned_records_are_revalidated_without_a_cached_decode(vectors, mutation):
    ctx = context(vectors)
    assert ctx.compare("first")[0]
    r = ctx.regions["first"]
    if mutation == "vector extent":
        r["inline_vectors"][0]["size"] = 4
    elif mutation == "vector binding":
        ctx.bindings["vector_a"]["va"] = hex(TEXT_VA + 20)
    elif mutation == "entry offset":
        r["entries"][-1]["offset"] = 144
    else:
        r["retail"]["va"] = hex(TEXT_VA + 4)
    with pytest.raises(ValueError):
        ctx.compare("first")


def test_declaration_order_is_not_an_instruction_skip_order(vectors):
    vectors[1]["inline_vectors"].reverse()
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    _, actual, original, _ = check(vectors, ctx)
    _, spans = ctx.decode_vectors("first", original, actual)
    assert [v["offset"] for v in spans] == [16, 64]
    # Omitting a declaration returns those bytes to the required instruction
    # stream; it cannot turn a decoding failure into an ignored gap.
    ctx.regions["first"]["inline_vectors"] = [{"symbol": "vector_b", "size": 4}]
    with pytest.raises(ValueError, match="every original instruction byte"):
        ctx.compare("first")


def test_vector_context_cannot_authorize_provider_entries_or_conflicting_registry(vectors):
    c, r, b, image = vectors
    registry = {}
    register_external_bindings(registry, c, b, "retail", "Sample")
    other = deepcopy(b)
    other["first"]["va"] = hex(TEXT_VA + 4)
    with pytest.raises(ValueError, match="Conflicting"):
        register_external_bindings(registry, c, other, "retail", "Other")
    provider = deepcopy(c)
    c.named["pointer"].update(section=0, value=4)
    providers = {"pointer": provider}
    ctx = CodeRegions(c, [r], "retail", b, image, common_definitions=providers)
    with pytest.raises(ValueError, match="Provider-owned code regions are unsupported"):
        validate_common_definitions(c, b, image, providers, code_regions=ctx)


def test_only_explicit_core_opt_in_adds_a_capstone_requirement(vectors, monkeypatch):
    from binary_comp.core import disasm

    def unavailable():
        raise RuntimeError("capstone is required")

    monkeypatch.setattr(disasm, "require_capstone_16", unavailable)
    with pytest.raises(RuntimeError, match="capstone is required"):
        check(vectors)
    del vectors[1]["inline_vectors"]
    assert check(vectors)[0]


def public_region_dependency(api, fixture, ctx, visiting):
    c, _, b, image = fixture
    if api == "data":
        return validate_data_binding(c, "pointer", b, ctx.reference, visiting, code_regions=ctx)
    if api == "target":
        return ctx.validate_target(c, c.named["first"], TEXT_VA + 3, visiting)
    section, origin = (c.sections[2], DATA_VA) if api == "data fixups" else (c.sections[1], TEXT_VA + 256)
    return apply_fixups(c, section["data"], section["relocations"], origin,
                        b, ctx.reference, visiting, code_regions=ctx)


@pytest.mark.parametrize("api", ["data", "target", "data fixups", "caller fixups"])
@pytest.mark.parametrize("path", ["forged region", "unrelated", "both"])
def test_caller_visiting_sets_cannot_skip_complete_region_decode(vectors, api, path):
    c = vectors[0]
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    mutate_byte(vectors, 63, 0x0f, both=True)
    identity = ("code region", id(c), id(c.sections[0]))
    visiting = {identity} if path == "forged region" else {(id(c), "unrelated")}
    if path == "both":
        visiting.add(identity)
    before = set(visiting)
    message = "active native validation" if identity in visiting else "every original instruction byte"
    with pytest.raises(ValueError, match=message):
        public_region_dependency(api, vectors, ctx, visiting)
    assert visiting == before


@pytest.mark.parametrize("mutation", ["vector type", "entry offset", "original payload"])
@pytest.mark.parametrize("api", ["data", "target", "data fixups"])
def test_forged_region_recursion_cannot_reuse_stale_native_or_original_inputs(vectors, api, mutation):
    c, _, _, image = vectors
    ctx = context(vectors)
    assert check(vectors, ctx)[0]
    if mutation == "vector type":
        c.named["vector_a"]["type"] = 0x20
    elif mutation == "entry offset":
        c.named["last"]["value"] += 1
    else:
        raw = bytearray(image.data)
        raw[image.section_for_va(TEXT_VA).rawptr + 16] ^= 1
        image.data = bytes(raw)
    visiting = {("code region", id(c), id(c.sections[0]))}
    with pytest.raises(ValueError):
        public_region_dependency(api, vectors, ctx, visiting)


class RegionCallbackReference:
    """Invoke a public callback during the first internal target validation."""

    def __init__(self, image):
        self.image = image
        self.callback = None
        self.whole_reads = 0

    def section_for_va(self, address):
        return self.image.section_for_va(address)

    def read(self, address, size):
        if self.callback is not None and (address, size) == (TEXT_VA, 160):
            self.whole_reads += 1
            if self.whole_reads == 2:
                callback, self.callback = self.callback, None
                callback()
        return self.image.read(address, size)


@pytest.mark.parametrize("api", ["data", "target", "data fixups", "caller fixups"])
def test_public_reentry_cannot_borrow_an_active_regions_recursion_permission(vectors, api):
    c, r, b, image = vectors
    reference = RegionCallbackReference(image)
    ctx = CodeRegions(c, [r], "retail", b, reference)
    observed = []

    def callback():
        # Equal malformed buffers must not be accepted just because an outer
        # native comparison is currently traversing this region's fixups.
        mutate_byte(vectors, 63, 0x0f, both=True)
        try:
            with pytest.raises(ValueError, match="active native validation"):
                public_region_dependency(api, vectors, ctx,
                    {("code region", id(c), id(c.sections[0]))})
            observed.append("rejected")
        finally:
            mutate_byte(vectors, 63, 0x90, both=True)

    reference.callback = callback
    assert ctx.compare("first")[0]
    assert observed == ["rejected"]
    assert validate_data_binding(c, "pointer", b, reference, code_regions=ctx)[1] == 1


@pytest.mark.parametrize("outer_fails", [False, True])
@pytest.mark.parametrize("api", ["data", "target", "data fixups"])
def test_copied_context_cannot_reuse_region_permission_after_operation_exit(vectors, outer_fails, api):
    from contextvars import copy_context

    c, r, b, image = vectors
    reference = RegionCallbackReference(image)
    ctx = CodeRegions(c, [r], "retail", b, reference)
    copied = []
    reference.callback = lambda: copied.append(copy_context())
    if outer_fails:
        mutate_byte(vectors, 63, 0x0f, both=True)
        with pytest.raises(ValueError, match="every original instruction byte"):
            ctx.compare("first")
    else:
        assert ctx.compare("first")[0]
        mutate_byte(vectors, 63, 0x0f, both=True)
    assert len(copied) == 1
    with pytest.raises(ValueError, match="active native validation"):
        copied[0].run(public_region_dependency, api, vectors, ctx,
                      {("code region", id(c), id(c.sections[0]))})
