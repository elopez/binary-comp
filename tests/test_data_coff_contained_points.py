"""Contained code points are edge targets, using synthetic COFF/PE windows."""

import pytest

from binary_comp.analyzers import coff_linked_graph
from binary_comp.analyzers.coff_data import data_window
from binary_comp.analyzers.coff_report import _code_bindings, ordinary_entries, region_context
from binary_comp.core.coff import COFF
from binary_comp.core.mapfile import parse_msvc_map_symbols
from binary_comp.core.pe import FileBackedPEImage
from conftest import TEXT_VA
from test_data_coff_graph import (
    compare, linked, map_line, project, refresh_original_hash, rows, save_image, save_map,
    save_modules, save_object, word,
)
from test_exe_coff import function


def save_project(state, *, rebuilt=None):
    save_object(state)
    save_modules(state)
    save_image(state, "original", state["addresses"])
    refresh_original_hash(state)
    save_image(state, "rebuilt", rebuilt or state["addresses"])
    save_map(state, rebuilt or state["addresses"])


@pytest.fixture
def points(project):
    # Typed entries bound the complete owner independently of its local label.
    for native in project["native"][1:3]:
        native.update(storage=3, type=0x20)
    module = project["module"]
    module["functions"].extend(module.pop("code_labels"))
    project["native"].append({"name": "inside", "section": 2, "storage": 6, "value": 5})
    project["addresses"].append(project["addresses"][1] + 5)
    module["bindings"]["retail"]["inside"] = {"va": hex(project["addresses"][7])}
    project["sections"][0]["relocations"][0] = (1, 7, 20)
    save_project(project)
    return project


@pytest.mark.parametrize("kind,addend", [(6, 0), (6, -3), (20, 0), (20, 3)])
@pytest.mark.parametrize("shift", [0, 0x20])
def test_real_edges_to_contained_points_replay_the_whole_owner(points, kind, addend, shift):
    root = points["sections"][0]
    root["relocations"][0] = (1, 7, kind)
    root["data"] = bytes([0xE8 if kind == 20 else 0xB8]) + word(addend) + root["data"][5:]
    actual = [address + shift for address in points["addresses"]]
    if shift:
        actual[3:6] = [address + 8 for address in points["addresses"][3:6]]
    save_project(points, rebuilt=actual)
    result = rows(points)
    assert set(result) == {"info", "unwind", "_public"}
    for name, offset in (("info", 3), ("unwind", 4)):
        row = result[name]
        assert row.rebuilt_address == actual[offset]
        assert row.resolution == "relocation-graph"
        assert row.placement_equal is (not shift)
        assert row.raw_equal is (not shift)
        proof = "\n".join(row.resolution_provenance)
        assert "index=7" in proof and "private contained point inside" in proof
        assert "Source proof Sample.obj:handler: complete native function" in proof
        assert "Source proof Sample.obj:inside" not in proof


@pytest.mark.parametrize("offset", [0, 1, 6, 10, 11, 16, 18])
def test_points_require_strict_interior_instruction_boundaries(points, offset):
    points["native"][7]["value"] = offset
    points["addresses"][7] = points["addresses"][1] + offset
    points["module"]["bindings"]["retail"]["inside"]["va"] = hex(points["addresses"][7])
    save_project(points)
    for name in ("info", "unwind"):
        assert rows(points)[name].rebuilt_address is None


def test_an_interior_nop_target_does_not_claim_a_new_entry_or_extent(points):
    section = points["sections"][1]
    section["data"] = section["data"][:5] + b"\x90" + section["data"][5:10] + b"\x90" * 5 + section["data"][16:]
    section["relocations"][1] = (7, 6, 20)
    points["module"]["functions"][1]["retail"]["size"] = 11
    save_project(points)
    assert compare(points).matches


def region_owner(state, *, anchor="handler"):
    # Keep strict native data validation acyclic when cleanup shares its owner.
    state["sections"][2]["relocations"][1] = (8, 6, 6)
    state["module"]["functions"] = state["module"]["functions"][:1]
    state["module"]["code_regions"] = [{
        "symbol": anchor, "label": "complete contribution",
        "anchor_offset": 16 if anchor == "cleanup" else 0,
        "entries": [{"symbol": "handler", "offset": 0}, {"symbol": "cleanup", "offset": 16}],
        "retail": {"va": hex(state["addresses"][1]), "size": len(state["sections"][1]["data"])},
    }]
    for name, index in (("handler", 1), ("cleanup", 2)):
        state["module"]["bindings"]["retail"][name] = {"va": hex(state["addresses"][index])}


@pytest.mark.parametrize("anchor", ["handler", "cleanup"])
def test_points_use_complete_region_owner_and_its_declared_anchor(points, anchor):
    region_owner(points, anchor=anchor)
    save_project(points)
    assert compare(points).matches
    proof = "\n".join(rows(points)["info"].resolution_provenance)
    assert f"Source proof Sample.obj:{anchor}: complete native region" in proof


@pytest.mark.parametrize("offset,accepted", [(5, True), (12, False)])
def test_region_inline_vectors_never_become_code_point_targets(points, offset, accepted):
    region_owner(points)
    points["module"]["code_regions"][0]["inline_vectors"] = [{"symbol": "vector", "size": 4}]
    points["native"].append({"name": "vector", "section": 2, "storage": 3, "value": 12})
    points["module"]["bindings"]["retail"]["vector"] = {"va": hex(points["addresses"][1] + 12)}
    section = points["sections"][1]
    section["data"] = section["data"][:12] + bytes(4) + section["data"][16:]
    section["relocations"].append((12, 1, 6))
    points["native"][7]["value"] = offset
    points["addresses"][7] = points["addresses"][1] + offset
    points["module"]["bindings"]["retail"]["inside"]["va"] = hex(points["addresses"][7])
    save_project(points)
    row = rows(points)["info"]
    assert (row.rebuilt_address is not None) is accepted
    if not accepted:
        assert "native and actual instruction boundary" in row.detail


@pytest.mark.parametrize("offset,accepted", [(5, True), (10, False), (12, False)])
def test_ordinary_inline_data_and_alignment_are_outside_code_points(points, offset, accepted):
    points["native"].extend([
        {"name": "table", "section": 2, "storage": 6, "value": 12},
        {"name": "case", "section": 2, "storage": 6, "value": 5},
    ])
    points["addresses"].extend([points["addresses"][1] + 12, points["addresses"][1] + 5])
    points["module"]["bindings"]["retail"].update({
        "table": {"va": hex(points["addresses"][8]), "data_size": 4, "inline_data": True},
        "case": {"va": hex(points["addresses"][9])},
    })
    section = points["sections"][1]
    section["data"] = section["data"][:12] + bytes(4) + section["data"][16:]
    section["relocations"].append((12, 9, 6))
    points["native"][7]["value"] = offset
    points["addresses"][7] = points["addresses"][1] + offset
    points["module"]["bindings"]["retail"]["inside"]["va"] = hex(points["addresses"][7])
    save_project(points)
    row = rows(points)["info"]
    assert row.contract_status == "validated"
    assert rows(points)["table"].contract_status == "validated"
    assert (row.rebuilt_address is not None) is accepted


@pytest.mark.parametrize("mutation", ["binding", "binding-role", "duplicate", "static", "external", "debug", "type", "aux", "unknown", "native-byte", "actual-byte"])
def test_invalid_point_identity_owner_or_replay_gives_no_descendant_credit(points, mutation):
    native = points["native"][7]
    if mutation == "binding":
        points["module"]["bindings"]["retail"]["inside"]["va"] = hex(points["addresses"][7] + 1)
    elif mutation == "binding-role":
        points["module"]["bindings"]["retail"]["inside"]["size"] = 1
    elif mutation == "duplicate":
        points["native"].append({"name": "inside"})
    elif mutation in ("static", "external", "debug"):
        native["storage"] = {"static": 3, "external": 2, "debug": 102}[mutation]
    elif mutation == "type":
        native["type"] = 1
    elif mutation == "aux":
        native["aux"] = bytes(18)
    elif mutation == "unknown":
        points["module"]["functions"].pop(1)
        points["module"]["bindings"]["retail"]["handler"] = {"va": hex(points["addresses"][1])}
    elif mutation == "native-byte":
        raw = points["sections"][1]["data"]
        points["sections"][1]["data"] = raw[:9] + b"\x01" + raw[10:]
    if mutation in ("native-byte", "actual-byte"):
        if mutation == "native-byte":
            save_object(points)
        else:
            path = points["path"] / "rebuilt.exe"
            image = bytearray(path.read_bytes())
            image[0x249] ^= 1
            path.write_bytes(image)
    else:
        save_project(points)
    for name in ("info", "unwind"):
        row = rows(points)[name]
        assert row.rebuilt_address is None and row.raw_equal is None


def test_point_map_name_cannot_anchor_an_owner(points):
    save_map(points, points["addresses"], root=False)
    path = points["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("inside", points["addresses"][7], flag=""))
    assert rows(points)["info"].rebuilt_address is None


@pytest.mark.parametrize("mutation", ["none", "value", "function", "ambiguous", "foreign", "unqualified"])
def test_point_map_records_constrain_only_exact_same_owner_edges(points, mutation):
    path = points["path"] / "rebuilt.map"
    address = points["addresses"][7] + (1 if mutation == "value" else 0)
    owner = "Other.obj" if mutation == "foreign" else "Sample.obj"
    line = map_line("inside", address, owner, flag="f" if mutation == "function" else "")
    if mutation == "foreign":
        line = map_line("inside", address + 1, owner)
    if mutation == "unqualified":
        line = map_line("inside", address + 1, "", flag="f")
    if mutation == "ambiguous":
        line += map_line("inside", address, "other/Sample.obj", flag="")
    path.write_text(path.read_text() + line)
    assert (rows(points)["info"].rebuilt_address is not None) is (mutation in ("none", "foreign", "unqualified"))


def test_unreferenced_point_map_rows_do_not_poison_other_edges(points):
    points["sections"][0]["relocations"][0] = (1, 1, 20)
    save_project(points)
    path = points["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("inside", points["addresses"][7] + 1))
    assert compare(points).matches


def test_overlapping_complete_code_owners_cannot_supply_a_point(points):
    points["native"].append({"name": "alias", "section": 2, "storage": 3, "type": 0x20})
    points["module"]["functions"].append(function("alias", points["addresses"][1], 10))
    save_project(points)
    # Exercise the graph's native ownership guard independently of the CLI's
    # earlier rejection of overlapping original function windows.
    module = points["module"]
    coff = COFF(points["path"] / "objects/Sample.obj")
    bindings = _code_bindings(module["bindings"]["retail"], ordinary_entries(module), "retail")
    reference = FileBackedPEImage(points["path"] / "original.exe")
    rebuilt = FileBackedPEImage(points["path"] / "rebuilt.exe")
    placements = coff_linked_graph.resolve_private_placements(
        coff, module, "retail", bindings, reference, rebuilt,
        parse_msvc_map_symbols(points["path"] / "rebuilt.map"),
        {"info": (8, 1), "unwind": (8, 1), "_public": (4, 0)}, providers={},
        regions=region_context(coff, module, "retail", bindings, reference, {}), read_window=data_window)
    assert not placements or all(proof.address is None for proof in placements.values())


def test_contained_point_requires_complete_decode_after_the_target(points):
    section = points["sections"][1]
    section["data"] = section["data"][:5] + b"\xc3\x90\x90\x90\x0f" + section["data"][10:]
    section["relocations"].pop()
    save_project(points)
    row = rows(points)["info"]
    assert row.rebuilt_address is None
    assert "complete code window" in row.detail


def test_actual_linked_instruction_boundaries_must_also_agree(points):
    # A legal DIR32 field at opcode bytes decodes differently after linking.
    # Native starts include +2; actual starts do not, despite exact full replay.
    section = points["sections"][1]
    section["data"] = bytes(4) + b"\x00\xc3\xb8" + bytes(4) + b"\xc3" + b"\x90" * 4 + section["data"][16:]
    section["relocations"] = [(0, 6, 6), (7, 3, 6)]
    points["module"]["functions"][1]["retail"]["size"] = 12
    points["native"][7]["value"] = 2
    points["addresses"][7] = points["addresses"][1] + 2
    points["module"]["bindings"]["retail"]["inside"]["va"] = hex(points["addresses"][7])
    actual = list(points["addresses"])
    actual[6] += 0x70
    save_project(points, rebuilt=actual)
    for name in ("info", "unwind"):
        row = rows(points)[name]
        assert row.rebuilt_address is None
        assert "native and actual instruction boundary" in row.detail


def test_missing_optional_decoder_fails_only_the_point_proof(points, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic decoder unavailable")
    monkeypatch.setattr(coff_linked_graph, "disassemble_raw", unavailable)
    row = rows(points)["info"]
    assert row.rebuilt_address is None and "decoder unavailable" in row.detail


@pytest.mark.parametrize("reverse", [False, True])
def test_failed_roots_sharing_a_contained_owner_are_atomic(points, reverse):
    # Reuse the existing PUBLIC root twice, with a second exact native owner.
    # Its public mismatch must poison the shared point owner even when visited
    # before that edge; the rejection walk must not depend on relocation order.
    raw = b"\xb8" + bytes(4) + b"\xba" + bytes(4) + b"\xc3"
    relocations = [(1, 6, 6), (6, 7, 6)]
    if reverse:
        relocations.reverse()
    points["sections"].append({"name": ".text$c", "data": raw, "relocations": relocations})
    points["native"].append({"name": "_second", "section": 4, "type": 0x20})
    points["module"]["functions"].append(function("_second", TEXT_VA + 0xA0, len(raw)))
    save_project(points)
    for role in ("original", "rebuilt"):
        addresses = list(points["addresses"])
        if role == "rebuilt":
            addresses[6] += 4
        path = points["path"] / f"{role}.exe"
        image = bytearray(path.read_bytes())
        image[0x2A0:0x2A0 + len(raw)] = linked(raw, relocations, TEXT_VA + 0xA0, addresses)
        path.write_bytes(image)
    refresh_original_hash(points)
    path = points["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("_second", TEXT_VA + 0xA0))
    for name in ("info", "unwind"):
        row = rows(points)[name]
        assert row.rebuilt_address is None
        assert "conflicts with MAP target" in row.detail


def test_partial_fragment_never_grants_contained_point_graph_trust(points):
    module = points["module"]
    owner = next(entry for entry in module["functions"] if entry["symbol"] == "handler")
    module["functions"].remove(owner)
    original = owner["retail"]
    module["code_fragments"] = [{"symbol": owner["symbol"], "label": "partial handler",
                                 "retail": {"va": original["va"], "owner_va": original["va"],
                                            "object_offset": 0, "size": original["size"]}}]
    save_project(points)
    result = rows(points)
    assert result["info"].rebuilt_address is None
    assert result["unwind"].rebuilt_address is None
    assert result["_public"].rebuilt_address is not None
