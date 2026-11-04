"""Actual linked private placements, using only synthetic COFF/PE records."""

import hashlib
import json
import struct

import pytest

from binary_comp.analyzers.coff_data import compare_linked_data, format_linked_data
from binary_comp.analyzers.coff_report import load_selection
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_contract import BSS_FLAGS, DATA_FLAGS, coff_bytes
from test_exe_coff import PUBLIC_HEADER, function, map_line


def word(value):
    return struct.pack("<I", value & 0xFFFFFFFF)


def linked(raw, relocations, source, addresses):
    result = bytearray(raw)
    for offset, index, kind in relocations:
        addend = struct.unpack_from("<I", raw, offset)[0]
        value = addresses[index] + addend - (source + offset + 4 if kind == 20 else 0)
        struct.pack_into("<I", result, offset, value & 0xFFFFFFFF)
    return bytes(result)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "objects").mkdir()
    root = b"\xe8" + bytes(4) + b"\xa1" + bytes(4) + b"\xc3"
    handler = b"\xb8" + word(2) + b"\xe9" + bytes(4)
    cleanup = b"\x90\xc3"
    sections = [
        {"name": ".text$a", "data": root, "relocations": [(1, 1, 20), (6, 5, 6)]},
        {"name": ".text$b", "data": handler + b"\x90" * 6 + cleanup,
         "relocations": [(1, 3, 6), (6, 6, 20)]},
        {"name": ".data", "data": bytes(4) + b"meta" + bytes(4) + word(3) + b"tag\0",
         "flags": DATA_FLAGS, "relocations": [(0, 4, 6), (8, 2, 6)]},
    ]
    native = [
        {"name": "_root", "section": 1, "type": 0x20},
        {"name": "handler", "section": 2, "storage": 6},
        {"name": "cleanup", "section": 2, "storage": 6, "value": 16},
        {"name": "info", "section": 3, "storage": 3},
        {"name": "unwind", "section": 3, "storage": 3, "value": 8},
        {"name": "_public", "section": 3, "value": 16},
        {"name": "_runtime", "type": 0x20},
    ]
    addresses = [TEXT_VA, TEXT_VA + 0x40, TEXT_VA + 0x50,
                 DATA_VA, DATA_VA + 8, DATA_VA + 16, TEXT_VA + 0x120]
    module = {
        "name": "Sample", "functions": [function("_root", TEXT_VA, len(root))],
        "code_labels": [function("handler", addresses[1], len(handler)),
                        function("cleanup", addresses[2], len(cleanup))],
        "bindings": {"retail": {
            "info": {"va": hex(addresses[3]), "data_size": 8},
            "unwind": {"va": hex(addresses[4]), "data_size": 8},
            "_public": {"va": hex(addresses[5]), "data_size": 4},
            "_runtime": {"va": hex(addresses[6])},
        }},
    }
    state = {"path": tmp_path, "sections": sections, "native": native,
             "addresses": addresses, "module": module}
    save_object(state)
    save_modules(state)
    save_image(state, "original", addresses)
    save_image(state, "rebuilt", addresses)
    save_map(state, addresses)
    (tmp_path / "binary-comp.json").write_text(json.dumps({
        "targets": {"retail": {
            "kind": "i386-coff", "original_exe": "original.exe",
            "original_sha256": hashlib.sha256((tmp_path / "original.exe").read_bytes()).hexdigest(),
            "rebuilt_exe": "rebuilt.exe", "map": "rebuilt.map", "object_dir": "objects",
        }}, "coff": {"modules": "modules.json"}}))
    return state


def save_object(state):
    (state["path"] / "objects/Sample.obj").write_bytes(coff_bytes(state["sections"], state["native"]))


def save_modules(state):
    (state["path"] / "modules.json").write_text(json.dumps([state["module"]]))


def save_map(state, addresses, *, owner="Sample.obj", root=True):
    lines = PUBLIC_HEADER
    if root:
        lines += map_line("_root", addresses[0], owner)
    lines += map_line("_public", addresses[5], "Sample.obj", flag="")
    lines += map_line("_runtime", addresses[6], "Runtime.obj")
    (state["path"] / "rebuilt.map").write_text(lines)


def save_image(state, name, addresses):
    root, helper, data = state["sections"][:3]
    text = bytearray(b"\x90" * 0x200)
    text[addresses[0] - TEXT_VA:addresses[0] - TEXT_VA + len(root["data"])] = linked(
        root["data"], root["relocations"], addresses[0], addresses)
    text[addresses[1] - TEXT_VA:addresses[1] - TEXT_VA + len(helper["data"])] = linked(
        helper["data"], helper["relocations"], addresses[1], addresses)
    payload = linked(data["data"], data["relocations"], addresses[3], addresses)
    write_tiny_pe(state["path"] / f"{name}.exe", text, {addresses[3] - DATA_VA: payload})


def compare(state):
    modules, targets, _ = load_selection(state["path"] / "binary-comp.json", "retail")
    return compare_linked_data(targets["retail"], modules)


def rows(state):
    return {row.symbol: row for row in compare(state).rows}


def refresh_original_hash(state):
    path = state["path"] / "binary-comp.json"
    config = json.loads(path.read_text())
    config["targets"]["retail"]["original_sha256"] = hashlib.sha256((state["path"] / "original.exe").read_bytes()).hexdigest()
    path.write_text(json.dumps(config))


def test_actual_dir32_rel32_private_code_and_data_chain(project):
    summary = compare(project)
    result = {row.symbol: row for row in summary.rows}
    assert summary.matches
    for name in ("info", "unwind"):
        row = result[name]
        assert row.resolution == "relocation-graph"
        assert row.raw_equal is row.placement_equal is True
        assert row.rebuilt_address == row.original_address
        assert row.map_object is None
        proof = "\n".join(row.resolution_provenance)
        assert "PUBLIC MAP root Sample.obj:_root" in proof
        assert "REL32 index=1" in proof and "DIR32 index=3 addend=0x2" in proof
        assert "Source proof Sample.obj:cleanup" in proof
        assert "Source proof Sample.obj:info" in proof
    assert "validated actual COFF relocation graph" in format_linked_data(summary)
    assert "Source proof" in format_linked_data(summary, verbose=True)


def test_actual_placement_and_raw_original_bytes_remain_independent(project):
    shifted = [TEXT_VA + 0x10, TEXT_VA + 0x80, TEXT_VA + 0x90,
               DATA_VA + 8, DATA_VA + 16, DATA_VA + 24, TEXT_VA + 0x160]
    save_image(project, "rebuilt", shifted)
    save_map(project, shifted)
    result = rows(project)
    for name, address in (("info", shifted[3]), ("unwind", shifted[4])):
        row = result[name]
        assert row.resolution == "relocation-graph" and row.rebuilt_address == address
        assert row.placement_equal is False and row.raw_equal is False
        assert row.contract_status == "validated"
    assert result["_public"].raw_equal is True


@pytest.mark.parametrize("mutation", ["source-byte", "source-contract", "private-contract", "private-missing", "root-role"])
def test_complete_source_and_private_contract_failures_give_no_descendant_credit(project, mutation):
    if mutation == "source-byte":
        path = project["path"] / "rebuilt.exe"
        data = bytearray(path.read_bytes())
        data[0x200] = 0xE9
        path.write_bytes(data)
    elif mutation == "source-contract":
        project["sections"][0]["data"] = b"\xe9" + project["sections"][0]["data"][1:]
        save_object(project)
    elif mutation == "private-contract":
        project["sections"][1]["data"] = project["sections"][1]["data"][:-1] + b"\xcc"
        save_object(project)
    elif mutation == "private-missing":
        project["module"]["code_labels"].pop()
        save_modules(project)
    else:
        project["module"]["code_labels"].append(project["module"]["functions"].pop())
        save_modules(project)
    for name in ("info", "unwind"):
        row = rows(project)[name]
        assert row.status == "missing" and row.rebuilt_address is None
        assert row.raw_equal is None and row.compared_size == 0


@pytest.mark.parametrize("mutation", ["wrong-address", "wrong-role", "ambiguous"])
def test_known_public_target_identity_and_value_constrain_every_edge(project, mutation):
    addresses = list(project["addresses"])
    if mutation == "wrong-address":
        addresses[6] += 4
        save_map(project, addresses)
    else:
        path = project["path"] / "rebuilt.map"
        text = path.read_text()
        if mutation == "wrong-role":
            text = text.replace(map_line("_public", addresses[5], "Sample.obj", flag=""),
                                map_line("_public", addresses[5], "Sample.obj"))
        else:
            text += map_line("_runtime", addresses[6], "other/Runtime.obj")
        path.write_text(text)
    for name in ("info", "unwind"):
        row = rows(project)[name]
        assert row.rebuilt_address is None and row.raw_equal is None


@pytest.mark.parametrize("native_type,map_flag,resolved", [(0x20, "", False), (0, "", True), (0, "f", True)])
def test_undefined_function_type_requires_function_map_flag_but_untyped_externs_do_not(project, native_type, map_flag, resolved):
    project["native"][6]["type"] = native_type
    save_object(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text().replace(
        map_line("_runtime", project["addresses"][6], "Runtime.obj"),
        map_line("_runtime", project["addresses"][6], "Runtime.obj", flag=map_flag)))
    for name in ("info", "unwind"):
        row = rows(project)[name]
        assert row.raw_equal is (True if resolved else None)
        assert (row.rebuilt_address is not None) is resolved


@pytest.mark.parametrize("mutation", ["unanchored", "foreign-root", "static-root", "no-heading", "unsupported", "overlap", "outside"])
def test_unsupported_or_unanchored_graphs_stay_unresolved(project, mutation):
    if mutation == "unanchored":
        save_map(project, project["addresses"], root=False)
    elif mutation == "foreign-root":
        save_map(project, project["addresses"], owner="Selected.obj")
    elif mutation in ("static-root", "no-heading"):
        path = project["path"] / "rebuilt.map"
        path.write_text(path.read_text().replace(PUBLIC_HEADER, "Static symbols\n\n" if mutation == "static-root" else ""))
    else:
        root = project["sections"][0]
        root["relocations"].append({"unsupported": (6, 5, 7), "overlap": (2, 5, 6),
                                    "outside": (9, 5, 6)}[mutation])
        save_object(project)
    for name in ("info", "unwind"):
        row = rows(project)[name]
        assert row.status == "missing" and row.rebuilt_address is None


def test_nonrelocated_private_bytes_are_replayed_exactly(project):
    path = project["path"] / "rebuilt.exe"
    payload = bytearray(path.read_bytes())
    payload[0x404] ^= 1
    path.write_bytes(payload)
    for name in ("info", "unwind"):
        row = rows(project)[name]
        assert row.rebuilt_address is None and row.raw_equal is None
        assert "source replay differs" in row.detail


def test_missing_private_map_never_uses_same_named_foreign_record(project):
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("info", DATA_VA + 8, "Other.obj", flag=""))
    assert rows(project)["info"].rebuilt_address == DATA_VA


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("failure", ["public-edge", "private-conflict", "source-byte", "none"])
def test_shared_private_components_are_atomic_in_both_relocation_orders(project, reverse, failure):
    raw = b"\xb8" + bytes(4) + b"\xba" + bytes(4) + b"\xc3"
    relocations = [(1, 6, 6), (6, 3, 6)]
    if reverse:
        relocations.reverse()
    project["sections"].append({"name": ".text$c", "data": raw, "relocations": relocations})
    project["native"].append({"name": "_second", "section": 4, "type": 0x20})
    project["module"]["functions"].append(function("_second", TEXT_VA + 0xA0, len(raw)))
    save_object(project)
    save_modules(project)
    for role in ("original", "rebuilt"):
        addresses = list(project["addresses"])
        if role == "rebuilt":
            addresses[6] += 4 if failure == "public-edge" else 0
            addresses[3] += 24 if failure == "private-conflict" else 0
        payload = bytearray(linked(raw, relocations, TEXT_VA + 0xA0, addresses))
        if role == "rebuilt" and failure == "source-byte":
            payload[-1] = 0xCC
        path = project["path"] / f"{role}.exe"
        image = bytearray(path.read_bytes())
        image[0x2A0:0x2A0 + len(payload)] = payload
        path.write_bytes(image)
    refresh_original_hash(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("_second", TEXT_VA + 0xA0))
    for name in ("info", "unwind"):
        row = rows(project)[name]
        if failure == "none":
            assert row.raw_equal is True
            assert "PUBLIC MAP root Sample.obj:_second" in "\n".join(row.resolution_provenance)
        else:
            assert row.rebuilt_address is None and row.raw_equal is None
            if failure == "private-conflict":
                assert "Conflicting actual addresses" in row.detail


def test_unrelated_failed_root_sharing_only_public_targets_does_not_poison_valid_proofs(project):
    raw = b"\xb8" + bytes(4) + b"\xc3"
    relocations = [(1, 6, 6)]
    project["sections"].append({"name": ".text$c", "data": raw, "relocations": relocations})
    project["native"].append({"name": "_second", "section": 4, "type": 0x20})
    project["module"]["functions"].append(function("_second", TEXT_VA + 0xA0, len(raw)))
    save_object(project)
    save_modules(project)
    for role in ("original", "rebuilt"):
        addresses = list(project["addresses"])
        addresses[6] += 4 if role == "rebuilt" else 0
        payload = linked(raw, relocations, TEXT_VA + 0xA0, addresses)
        path = project["path"] / f"{role}.exe"
        image = bytearray(path.read_bytes())
        image[0x2A0:0x2A0 + len(payload)] = payload
        path.write_bytes(image)
    refresh_original_hash(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("_second", TEXT_VA + 0xA0))
    assert compare(project).matches


def test_complete_public_data_can_anchor_private_data_and_code(project):
    data = project["sections"][2]
    data["data"] = data["data"][:16] + bytes(4)
    data["relocations"].append((16, 3, 6))
    save_object(project)
    for role in ("original", "rebuilt"):
        save_image(project, role, project["addresses"])
    refresh_original_hash(project)
    save_map(project, project["addresses"], root=False)
    result = rows(project)
    assert result["info"].raw_equal is True
    proof = "\n".join(result["info"].resolution_provenance)
    assert "PUBLIC MAP root Sample.obj:_public" in proof
    assert "Source proof Sample.obj:cleanup" in proof


@pytest.mark.parametrize("shift", [0, 0x10])
def test_region_root_uses_map_anchor_offset_and_replays_the_complete_prefix(project, shift):
    root = project["sections"][0]
    root["data"] = b"\x90" * 3 + root["data"] + b"\xc3"
    root["relocations"] = [(offset + 3, index, kind) for offset, index, kind in root["relocations"]]
    project["native"][0]["value"] = 3
    project["native"].append({"name": "entry", "section": 1, "type": 0x20, "storage": 3, "value": 14})
    module = project["module"]
    module["functions"] = []
    module["code_regions"] = [{
        "label": "contribution", "symbol": "_root", "anchor_offset": 3,
        "entries": [{"symbol": "_root", "offset": 3}, {"symbol": "entry", "offset": 14}],
        "retail": {"va": hex(TEXT_VA), "size": len(root["data"])}}]
    module["bindings"]["retail"].update({"_root": {"va": hex(TEXT_VA + 3)},
                                           "entry": {"va": hex(TEXT_VA + 14)}})
    save_object(project)
    save_modules(project)
    save_image(project, "original", project["addresses"])
    refresh_original_hash(project)
    addresses = list(project["addresses"])
    addresses[0] += shift
    save_image(project, "rebuilt", addresses)
    save_map(project, addresses)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text().replace(map_line("_root", addresses[0]),
                                              map_line("_root", addresses[0] + 3)))
    row = rows(project)["info"]
    assert row.raw_equal is True
    assert f"complete region window at {TEXT_VA + shift:#010x}" in "\n".join(row.resolution_provenance)
    if shift:
        image = bytearray((project["path"] / "rebuilt.exe").read_bytes())
        image[0x200 + shift] = 0xCC
        (project["path"] / "rebuilt.exe").write_bytes(image)
        assert rows(project)["info"].rebuilt_address is None


def test_overlapping_pe_code_source_sections_cannot_authenticate_private_data(project):
    path = project["path"] / "rebuilt.exe"
    image = bytearray(path.read_bytes())
    struct.pack_into("<I", image, 0x1A0 + 12, 0x1000)
    path.write_bytes(image)
    assert rows(project)["info"].rebuilt_address is None


def test_overlapping_windows_visible_only_after_merging_shared_private_components_fail(project):
    other_raw = b"meta" + bytes(4)
    second_raw = b"\xb8" + bytes(4) + b"\xc3"
    project["sections"].extend([
        {"name": ".data$b", "data": other_raw, "flags": DATA_FLAGS, "relocations": [(4, 2, 6)]},
        {"name": ".text$c", "data": second_raw, "relocations": [(1, 7, 6)]},
    ])
    project["native"].extend([
        {"name": "other", "section": 4, "storage": 3},
        {"name": "_second", "section": 5, "type": 0x20},
    ])
    project["module"]["functions"].append(function("_second", TEXT_VA + 0xA0, len(second_raw)))
    project["module"]["bindings"]["retail"]["other"] = {"va": hex(DATA_VA + 24), "data_size": 8}
    save_object(project)
    save_modules(project)
    for role, other_va in (("original", DATA_VA + 24), ("rebuilt", DATA_VA + 4)):
        path = project["path"] / f"{role}.exe"
        image = bytearray(path.read_bytes())
        image[0x2A0:0x2A6] = b"\xb8" + word(other_va) + b"\xc3"
        if role == "original":
            image[0x418:0x420] = b"meta" + word(project["addresses"][2])
        path.write_bytes(image)
    refresh_original_hash(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("_second", TEXT_VA + 0xA0))
    for name in ("info", "unwind", "other"):
        row = rows(project)[name]
        assert row.rebuilt_address is None and row.raw_equal is None
        assert "Overlapping actual graph" in row.detail


@pytest.mark.parametrize("kind,addend", [(6, 3), (6, -3), (20, 3), (20, -3)])
def test_nonzero_signed_and_unsigned_addends_are_preserved(project, kind, addend):
    root = project["sections"][0]
    root["data"] = root["data"][:1] + word(addend) + root["data"][5:]
    root["relocations"][0] = (1, 1, kind)
    save_object(project)
    for role in ("original", "rebuilt"):
        save_image(project, role, project["addresses"])
    refresh_original_hash(project)
    assert compare(project).matches


@pytest.mark.parametrize("mutation", ["weak", "undefined-alias", "unknown-body"])
def test_no_weak_undefined_or_uncontracted_private_substitution(project, mutation):
    if mutation == "weak":
        # Existing native comparison may follow this fallback, but it is not
        # the actual private symbol named by the relocation's native index.
        project["native"].append({"name": "weak", "storage": 105,
                                  "aux": struct.pack("<II10x", 1, 3)})
        project["sections"][0]["relocations"][0] = (1, 7, 20)
    elif mutation == "undefined-alias":
        project["native"].append({"name": "handler"})
        project["sections"][0]["relocations"][0] = (1, 7, 20)
    else:
        project["module"]["code_labels"].pop(0)
        project["module"]["bindings"]["retail"]["handler"] = {"va": hex(project["addresses"][1])}
        save_modules(project)
    save_object(project)
    for name in ("info", "unwind"):
        row = rows(project)[name]
        assert row.rebuilt_address is None and row.raw_equal is None


@pytest.mark.parametrize("backing", ["file", "virtual", "mixed", "unmapped", "wrap"])
def test_bss_is_a_terminal_target_and_pe_backing_controls_raw_credit(project, backing):
    native = project["native"]
    native.append({"name": "zeros", "section": 4, "storage": 3})
    project["sections"].append({"name": ".bss", "size": 4, "flags": BSS_FLAGS})
    # Redirect the complete private data's final edge to a BSS target.
    project["sections"][2]["relocations"][1] = (8, 7, 6)
    addresses = project["addresses"] + [DATA_VA + 24]
    project["module"]["bindings"]["retail"]["zeros"] = {"va": hex(addresses[7]), "data_size": 4}
    save_object(project)
    save_modules(project)
    save_image(project, "original", addresses)
    refresh_original_hash(project)
    if backing in ("unmapped", "wrap"):
        addresses[7] = 0x500000 if backing == "unmapped" else 0xFFFFFFFE
    save_image(project, "rebuilt", addresses)
    path = project["path"] / "rebuilt.exe"
    payload = bytearray(path.read_bytes())
    if backing in ("virtual", "mixed"):
        struct.pack_into("<I", payload, 0x1A0 + 16, 24 if backing == "virtual" else 26)
    path.write_bytes(payload)
    row = rows(project)["zeros"]
    if backing in ("unmapped", "wrap"):
        assert row.rebuilt_address is None and row.raw_equal is None
    else:
        assert row.rebuilt_address == DATA_VA + 24
        assert row.resolution == "relocation-graph"
        assert row.raw_equal is (True if backing == "file" else None)
        assert row.rebuilt_backing == {"file": "file-backed", "virtual": "virtual-zero-fill",
                                       "mixed": "mixed-file-zero-fill"}[backing]
