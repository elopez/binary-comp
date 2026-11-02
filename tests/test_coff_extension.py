"""Native object extension checks, using exclusively synthetic COFF fixtures."""
from copy import deepcopy
import json
import struct

import pytest

from binary_comp.cli import main
from binary_comp.core.coff import COFF
from binary_comp.core.coff_extension import compare_coff_extension
from test_coff_contract import coff_bytes


def section_aux(size, relocations=0, *, selection=0, association=0, checksum=0x12345678):
    return struct.pack("<IHHIHB3x", size, relocations, 0, checksum, association, selection)


def fixture(*, extended=False, path=None):
    sections = [
        {"id": "code", "name": ".text", "data": bytes(4) + b"\xc3\x90\x90\x90",
         "flags": 0x60501020, "relocations": [(0, "local", 6)]},
        {"id": "data", "name": ".rdata", "data": b"tag\0", "flags": 0x40000040},
        {"id": "eh", "name": ".xdata", "data": bytes(4), "flags": 0x40001040,
         "relocations": [(0, "entry", 6)], "association": "code"},
        {"id": "debug", "name": ".debug$F", "data": b"meta", "flags": 0x42101048,
         "association": "code"},
    ]
    if extended:
        sections.insert(0, {"id": "new", "name": ".text", "data": b"\xc3", "flags": 0x60101020})
        sections.insert(3, {"id": "other", "name": ".text", "data": b"\x90\xc3", "flags": 0x60101020})
    numbers = {s["id"]: i for i, s in enumerate(sections, 1)}
    groups = {}
    for section in sections:
        ident = section["id"]
        selection = 5 if "association" in section else 1 if section["flags"] & 0x1000 else 0
        groups[ident] = [{"id": ident + "_section", "name": section["name"],
                          "section": numbers[ident], "storage": 3,
                          "aux": section_aux(len(section["data"]), len(section.get("relocations", [])),
                                             selection=selection,
                                             association=numbers.get(section.get("association"), 0))}]
    groups["code"] += [
        {"id": "entry", "name": "_entry", "type": 0x24, "section": numbers["code"]},
        {"id": "local", "name": "$private_72" if extended else "$private_1",
         "storage": 3, "value": 4, "section": numbers["code"]},
    ]
    groups["data"].append({"id": "datum", "name": "_datum", "section": numbers["data"]})
    groups["eh"].append({"id": "eh_data", "name": "_eh_data", "section": numbers["eh"]})
    if extended:
        for ident in ("new", "other"):
            groups[ident].append({"id": ident, "name": "_" + ident, "type": 0x20,
                                  "section": numbers[ident]})
    order = ["eh", "other", "debug", "data", "new", "code"] if extended else ["code", "data", "eh", "debug"]
    symbols = [symbol for ident in order for symbol in groups.get(ident, [])]
    symbols += [{"id": "weak", "name": "_weak_destroy", "storage": 105, "type": 0x20,
                 "aux": bytes(18)},
                {"id": "external", "name": "_destroy", "type": 0x20}]
    if not extended:
        symbols.append({"id": "new", "name": "_new", "type": 0x20})
    if path is not None:
        encoded = path.encode("ascii") + b"\0"
        symbols.append({"id": "file", "name": ".file", "section": -2, "storage": 103,
                        "aux": encoded.ljust((len(encoded) + 17) // 18 * 18, b"\0")})
    return sections, symbols


def pack(parts):
    sections, symbols = deepcopy(parts)
    indices, index = {}, 0
    for symbol in symbols:
        indices[symbol["id"]] = index
        index += 1 + len(symbol.get("aux", b"")) // 18
    for section in sections:
        section["relocations"] = [(offset, indices[target] if isinstance(target, str) else target, kind)
                                  for offset, target, kind in section.get("relocations", [])]
    for symbol in symbols:
        if symbol.get("storage") == 105 and symbol.get("fix_weak", True):
            symbol["aux"] = struct.pack("<II10x", indices[symbol.get("fallback", "external")],
                                        symbol.get("search", 2))
    return COFF.from_bytes(coff_bytes(sections, symbols))


def compare(parts=None, *, original=None, **options):
    return compare_coff_extension(pack(original or fixture()), pack(parts or fixture(extended=True)),
                                  ["_new", "_other"], **options)


def find(parts, ident, *, section=False):
    return next(item for item in parts[0 if section else 1] if item["id"] == ident)


def test_interleaved_additions_preserve_order_bytes_eh_associations_and_private_labels():
    result = compare()
    assert result["equivalent"]
    assert result["preserved_sections"] == 4
    assert result["preserved_bytes"] == 20
    assert result["preserved_relocations"] == 2
    assert result["promoted_undefined_symbols"] == 1
    assert result["section_map"] == [
        {"baseline": 1, "candidate": 2}, {"baseline": 2, "candidate": 3},
        {"baseline": 3, "candidate": 5}, {"baseline": 4, "candidate": 6}]
    assert {row["name"] for row in result["added_functions"]} == {"_new", "_other"}
    assert result["source_path_changes"] == []


def test_an_unchanged_undefined_record_can_remain_beside_its_new_definition():
    parts = fixture(extended=True)
    parts[1].append({"id": "new_declaration", "name": "_new", "type": 0x20})
    assert compare(parts)["promoted_undefined_symbols"] == 0


def test_incoming_relocations_and_weak_fallbacks_can_target_a_promoted_public_definition():
    original, parts = fixture(), fixture(extended=True)
    for records in (original, parts):
        find(records, "code", section=True)["relocations"] = [(0, "new", 20)]
        find(records, "weak")["fallback"] = "new"
    assert compare(parts, original=original)["promoted_undefined_symbols"] == 1


@pytest.mark.parametrize("storage", [3, 6])
def test_only_untyped_local_labels_can_change_spelling(storage):
    original, parts = fixture(), fixture(extended=True)
    for records in (original, parts):
        find(records, "local")["storage"] = storage
    assert compare(parts, original=original)["equivalent"]
    find(parts, "local")["type"] = 4
    with pytest.raises(ValueError):
        compare(parts, original=original)


def test_undefined_multiplicity_is_not_erased_by_one_new_definition():
    original = fixture()
    original[1].append({"id": "new_duplicate", "name": "_new", "type": 0x20})
    with pytest.raises(ValueError, match="Ambiguous added-function declaration"):
        compare(original=original)


def test_unreferenced_undefined_records_cannot_disappear():
    original = fixture()
    original[1].append({"id": "unreferenced", "name": "_unreferenced"})
    with pytest.raises(ValueError, match="Symbol identities"):
        compare(original=original)


def test_identity_without_additions_does_not_require_a_rebuild():
    obj = pack(fixture())
    result = compare_coff_extension(obj, obj)
    assert result["equivalent"] and result["added_functions"] == []


@pytest.mark.parametrize("ident", ["code", "data", "eh", "debug"])
def test_any_preserved_raw_byte_including_relocation_addends_must_remain_exact(ident):
    parts = fixture(extended=True)
    section = find(parts, ident, section=True)
    section["data"] = bytes([section["data"][0] ^ 1]) + section["data"][1:]
    with pytest.raises(ValueError, match="Raw section bytes"):
        compare(parts)


@pytest.mark.parametrize("change", ["offset", "kind", "target", "flags", "name"])
def test_relocation_tuples_runtime_targets_and_section_metadata_are_exact(change):
    parts = fixture(extended=True)
    section = find(parts, "code", section=True)
    if change == "offset":
        section["relocations"][0] = (1, "local", 6)
    elif change == "kind":
        section["relocations"][0] = (0, "local", 20)
    elif change == "target":
        section["relocations"][0] = (0, "external", 6)
    elif change == "flags":
        section["flags"] ^= 0x100000
    else:
        section["name"] = ".code"
        find(parts, "code_section")["name"] = ".code"
    with pytest.raises(ValueError):
        compare(parts)


@pytest.mark.parametrize("offset", [0, 4, 6, 8, 12, 14, 15])
def test_every_section_auxiliary_field_is_checked(offset):
    parts = fixture(extended=True)
    record = find(parts, "eh_section")
    aux = bytearray(record["aux"])
    aux[offset] ^= 1
    record["aux"] = bytes(aux)
    with pytest.raises(ValueError):
        compare(parts)


@pytest.mark.parametrize("field,value", [("name", "_changed"), ("type", 0x20),
                                         ("value", 1), ("storage", 3)])
def test_public_function_name_type_offset_and_storage_are_not_private_labels(field, value):
    parts = fixture(extended=True)
    find(parts, "entry")[field] = value
    with pytest.raises(ValueError):
        compare(parts)


def test_private_label_multiplicity_and_target_offset_are_preserved():
    for change in ("offset", "duplicate", "remove"):
        parts = fixture(extended=True)
        local = find(parts, "local")
        if change == "offset":
            local["value"] = 5
        elif change == "duplicate":
            parts[1].append({**local, "id": "duplicate", "name": "$private_99"})
        else:
            parts[1].remove(local)
            find(parts, "code", section=True)["relocations"] = [(0, "entry", 6)]
        with pytest.raises(ValueError):
            compare(parts)


@pytest.mark.parametrize("change", ["relocation", "extra_symbol", "data", "untyped", "offset", "flags", "aux"])
def test_added_function_exemption_cannot_hide_other_contributions(change):
    parts = fixture(extended=True)
    section = find(parts, "new", section=True)
    function = find(parts, "new")
    if change == "relocation":
        section["data"] = bytes(4)
        section["relocations"] = [(0, "external", 6)]
    elif change == "extra_symbol":
        parts[1].append({"id": "hidden", "name": "$hidden", "section": 1, "storage": 3})
    elif change == "data":
        section["name"] = ".data"
    elif change == "untyped":
        function["type"] = 0
    elif change == "offset":
        function["value"] = 1
    elif change == "flags":
        section["flags"] |= 0x80000000
    else:
        function["aux"] = bytes(18)
    with pytest.raises(ValueError):
        compare(parts)


def test_extra_data_sections_or_externals_are_never_allowed():
    parts = fixture(extended=True)
    parts[0].append({"id": "extra", "name": ".data", "flags": 0xC0000040, "data": b"x"})
    with pytest.raises(ValueError, match="section count"):
        compare(parts)
    parts = fixture(extended=True)
    parts[1].append({"id": "extra", "name": "_extra_external"})
    with pytest.raises(ValueError, match="Symbol identities"):
        compare(parts)


def test_section_order_is_never_inferred_from_matching_content():
    parts = fixture(extended=True)
    parts[0][2], parts[0][4] = parts[0][4], parts[0][2]
    for symbol in parts[1]:
        if symbol.get("section") in (3, 5):
            symbol["section"] = 8 - symbol["section"]
    with pytest.raises(ValueError, match="Section name or flags differ"):
        compare(parts)


def test_unchanged_undefined_records_and_common_requests_are_not_discarded():
    for mutation in ("undefined", "common"):
        original, candidate = fixture(), fixture(extended=True)
        if mutation == "undefined":
            candidate[1].append({"id": "new_undefined", "name": "_unexpected"})
        else:
            original[1].append({"id": "common", "name": "_buffer", "value": 4})
            candidate[1].append({"id": "common", "name": "_buffer", "value": 8})
        with pytest.raises(ValueError, match="Symbol identities"):
            compare(candidate, original=original)


@pytest.mark.parametrize("change", ["fallback", "search", "reserved", "cycle"])
def test_weak_fallbacks_preserve_targets_search_reserved_bytes_and_acyclic_topology(change):
    parts = fixture(extended=True)
    weak = find(parts, "weak")
    if change == "fallback":
        weak["fallback"] = "entry"
    elif change == "search":
        weak["search"] = 1
    elif change == "cycle":
        weak["fallback"] = "weak"
    else:
        weak["fix_weak"] = False
        weak["aux"] = struct.pack("<II", 0, 2) + b"x" + bytes(9)
    with pytest.raises(ValueError):
        compare(parts)


def test_association_cycles_and_links_to_added_functions_are_rejected():
    for target in (5, 1):
        parts = fixture(extended=True)
        record = find(parts, "eh_section")
        aux = bytearray(record["aux"])
        struct.pack_into("<H", aux, 12, target)
        record["aux"] = bytes(aux)
        with pytest.raises(ValueError):
            compare(parts)


def test_weak_chains_retain_their_named_topology_after_index_renumbering():
    original, parts = fixture(), fixture(extended=True)
    for records in (original, parts):
        records[1].append({"id": "weak2", "name": "_weak_destroy2", "storage": 105,
                           "type": 0x20, "fallback": "weak", "aux": bytes(18)})
    assert compare(parts, original=original)["equivalent"]
    find(parts, "weak2")["fallback"] = "entry"
    with pytest.raises(ValueError):
        compare(parts, original=original)


@pytest.mark.parametrize("change", ["size", "line_pointer", "reserved"])
def test_supported_function_auxiliaries_compare_size_and_reject_unsupported_metadata(change):
    original, parts = fixture(), fixture(extended=True)
    for records in (original, parts):
        find(records, "entry")["aux"] = struct.pack("<IIIIH", 0, 8, 0, 0, 0)
    assert compare(parts, original=original)["equivalent"]
    entry = find(parts, "entry")
    aux = bytearray(entry["aux"])
    aux[{"size": 4, "line_pointer": 8, "reserved": 16}[change]] ^= 1
    entry["aux"] = bytes(aux)
    with pytest.raises(ValueError):
        compare(parts, original=original)


@pytest.mark.parametrize("offset,width", [(18, 2), (20 + 8, 4), (20 + 12, 4)])
def test_header_flags_and_section_virtual_metadata_are_checked(offset, width):
    original = pack(fixture())
    raw = bytearray(original.data)
    struct.pack_into("<H" if width == 2 else "<I", raw, offset, 1)
    with pytest.raises(ValueError):
        compare_coff_extension(original, COFF.from_bytes(raw))


def test_source_path_changes_require_opt_in_and_report_metadata_only_witness():
    original = fixture(path="a/sample.cpp")
    parts = fixture(extended=True, path="a/longer/path/sample.cpp")
    with pytest.raises(ValueError, match="Symbol identities"):
        compare(parts, original=original)
    result = compare(parts, original=original, allow_source_path_change=True)
    assert result["source_path_changes"] == [{"before": "a/sample.cpp", "after": "a/longer/path/sample.cpp"}]


@pytest.mark.parametrize("ident", ["data_section", "eh_section", "debug_section"])
def test_nonselecting_checksums_require_opt_in_and_a_complete_preservation_witness(ident):
    parts = fixture(extended=True)
    record = find(parts, ident)
    aux = bytearray(record["aux"])
    struct.pack_into("<I", aux, 8, 0)
    record["aux"] = bytes(aux)
    with pytest.raises(ValueError):
        compare(parts)
    result = compare(parts, allow_nonselecting_checksum_change=True)
    change, = result["section_checksum_changes"]
    assert change["selection"] in (0, 5)
    assert (change["before"], change["after"]) == (0x12345678, 0)
    section = next(s for s in parts[0] if s["name"] == record["name"])
    section["data"] = b"FAIL"
    with pytest.raises(ValueError):
        compare(parts, allow_nonselecting_checksum_change=True)


@pytest.mark.parametrize("selection", [1, 2, 3, 4, 6])
def test_independently_selected_comdat_checksums_are_exact_even_with_opt_in(selection):
    original, parts = fixture(), fixture(extended=True)
    for records in (original, parts):
        record = find(records, "code_section")
        aux = bytearray(record["aux"])
        aux[14] = selection
        record["aux"] = bytes(aux)
    record = find(parts, "code_section")
    aux = bytearray(record["aux"])
    struct.pack_into("<I", aux, 8, 0)
    record["aux"] = bytes(aux)
    with pytest.raises(ValueError):
        compare(parts, original=original, allow_nonselecting_checksum_change=True)


@pytest.mark.parametrize("change", ["basename", "padding", "record_count", "type"])
def test_source_path_opt_in_does_not_hide_other_file_metadata_changes(change):
    original, parts = fixture(path="a/sample.cpp"), fixture(extended=True, path="b/sample.cpp")
    file = find(parts, "file")
    if change == "basename":
        file["aux"] = b"b/other.cpp\0".ljust(18, b"\0")
    elif change == "padding":
        file["aux"] = file["aux"][:-1] + b"x"
    elif change == "record_count":
        parts[1].append({**file, "id": "other_file"})
    else:
        file["type"] = 1
    with pytest.raises(ValueError):
        compare(parts, original=original, allow_source_path_change=True)


def test_unknown_auxiliaries_and_legacy_line_tables_fail_closed():
    parts = fixture(extended=True)
    parts[1].append({"id": "unknown", "name": "unknown", "storage": 101, "aux": bytes(18)})
    with pytest.raises(ValueError, match="Unsupported auxiliary"):
        compare(parts)
    original = pack(fixture())
    raw = bytearray(original.data)
    struct.pack_into("<I", raw, 20 + 28, 1)
    with pytest.raises(ValueError, match="line tables"):
        compare_coff_extension(original, COFF.from_bytes(raw))


def test_cli_emits_a_json_witness_and_nonzero_rejection(tmp_path, capsys):
    baseline, candidate = tmp_path / "baseline.obj", tmp_path / "candidate.obj"
    baseline.write_bytes(pack(fixture()).data)
    candidate.write_bytes(pack(fixture(extended=True)).data)
    args = ["coff-compare", str(baseline), str(candidate), "--added-function", "_new",
            "--added-function", "_other"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["equivalent"]
    assert main(args[:-2]) == 1
    assert not json.loads(capsys.readouterr().out)["equivalent"]
