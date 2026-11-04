"""Explicit PUBLIC DLL terminals, with synthetic MAP/COFF provenance only."""

from dataclasses import replace
import struct

import pytest

from binary_comp.analyzers.exe import linked_map_identity
from binary_comp.core.mapfile import MapSymbol, parse_msvc_map_symbols
from conftest import DATA_VA, TEXT_VA
from test_data_coff_graph import (
    compare, project, refresh_original_hash, rows, save_image, save_modules,
    save_object, word,
)
from test_exe_coff import map_line


CONTRIBUTOR = "SUPPORT:SUPPORT.dll"


def prepare(state, *, name="_slot", native_type=0, flag="", kind=6, addend=0):
    state["native"][6].update(name=name, type=native_type)
    bindings = state["module"]["bindings"]["retail"]
    bindings.pop("_runtime")
    bindings[name] = {"va": hex(DATA_VA + 24)}
    helper = state["sections"][1]
    helper["data"] = helper["data"][:5] + b"\xa1" + word(addend) + helper["data"][10:]
    helper["relocations"][1] = (6, 6, kind)
    state["addresses"][6] = DATA_VA + 24
    save_object(state)
    save_modules(state)
    for role in ("original", "rebuilt"):
        save_image(state, role, state["addresses"])
    refresh_original_hash(state)
    path = state["path"] / "rebuilt.map"
    text = path.read_text().replace(map_line("_runtime", TEXT_VA + 0x120, "Runtime.obj"), "")
    path.write_text(text + map_line(name, DATA_VA + 24, CONTRIBUTOR, flag=flag, segment=2))


@pytest.mark.parametrize("name,native_type,flag,kind,addend", [
    ("_slot", 0, "", 6, 3), ("plain_name", 0, "f", 20, -2),
    ("_entry", 0x20, "f", 20, 0),
])
def test_explicit_dll_terminal_uses_actual_map_value_role_and_real_fixup(project, name, native_type, flag, kind, addend):
    prepare(project, name=name, native_type=native_type, flag=flag, kind=kind, addend=addend)
    assert compare(project).matches
    proof = "\n".join(rows(project)["info"].resolution_provenance)
    assert f"PUBLIC {name} at {DATA_VA + 24:#010x} from DLL {CONTRIBUTOR}" in proof
    assert f"index=6 addend={addend & 0xFFFFFFFF:#x}" in proof
    assert len(compare(project).rows) == 3


@pytest.mark.parametrize("update_operand", [False, True])
def test_dll_terminal_never_seeds_actual_placement_from_original_binding(project, update_operand):
    prepare(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text().replace(f"{DATA_VA + 24:08x}", f"{DATA_VA + 28:08x}"))
    if update_operand:
        actual = list(project["addresses"])
        actual[6] += 4
        save_image(project, "rebuilt", actual)
    assert (rows(project)["info"].rebuilt_address is not None) is update_operand


@pytest.mark.parametrize("mutation", ["static", "unknown-heading", "ownerless", "malformed", "wrong-name", "typed-role"])
def test_unqualified_private_or_inconsistent_dll_records_cannot_prove_terminals(project, mutation):
    prepare(project, native_type=0x20 if mutation == "typed-role" else 0)
    path = project["path"] / "rebuilt.map"
    line = map_line("_slot", DATA_VA + 24, CONTRIBUTOR, flag="", segment=2)
    replacement = {
        "static": "Static symbols\n" + line,
        "unknown-heading": "Unknown symbols\n" + line,
        "ownerless": map_line("_slot", DATA_VA + 24, "", flag="", segment=2),
        "malformed": line.replace(CONTRIBUTOR, "SUPPORT.dll"),
        "wrong-name": line.replace("_slot", "_other"),
        "typed-role": line,
    }[mutation]
    path.write_text(path.read_text().replace(line, replacement))
    assert rows(project)["info"].rebuilt_address is None


@pytest.mark.parametrize("extra", [
    map_line("_slot", DATA_VA + 24, "OTHER:OTHER.dll", flag="", segment=2),
    map_line("_slot", DATA_VA + 28, CONTRIBUTOR, flag="", segment=2),
    map_line("_slot", DATA_VA + 24, CONTRIBUTOR, flag="f", segment=2),
    map_line("_slot", DATA_VA + 24, "Sample.obj", flag="", segment=2),
    map_line("_slot", DATA_VA + 28, "", flag="", segment=2),
    map_line("_slot", DATA_VA + 28, "SUPPORT.dll", flag="", segment=2),
    map_line("_slot", DATA_VA + 24, "", flag="f", segment=2),
    map_line("_slot", DATA_VA + 24, "", flag="", segment=2),
])
def test_competing_public_object_or_dll_identities_remain_ambiguous(project, extra):
    prepare(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text() + extra)
    row = rows(project)["info"]
    assert row.rebuilt_address is None and "Ambiguous actual MAP target" in row.detail


def test_identical_dll_map_records_deduplicate(project):
    prepare(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("_slot", DATA_VA + 24, CONTRIBUTOR, flag="", segment=2))
    assert compare(project).matches


@pytest.mark.parametrize("mutation", ["private", "auxiliary", "duplicate", "weak", "unsupported-type"])
def test_only_one_strong_unaliased_supported_undefined_native_record_can_use_dll_terminal(project, mutation):
    prepare(project)
    if mutation == "private":
        project["native"][6]["storage"] = 3
    elif mutation == "auxiliary":
        project["native"][6]["aux"] = bytes(18)
    elif mutation == "duplicate":
        project["native"].append({"name": "_slot"})
    elif mutation == "weak":
        project["native"].append({"name": "weak", "storage": 105,
                                  "aux": struct.pack("<II10x", 6, 3)})
        project["sections"][1]["relocations"][1] = (6, 7, 6)
    else:
        project["native"][6]["type"] = 4
    save_object(project)
    assert rows(project)["info"].rebuilt_address is None


@pytest.mark.parametrize("common", [False, True])
def test_defined_native_and_common_records_cannot_borrow_public_dll_terminals(project, common):
    prepare(project)
    project["native"][6].update(section=0 if common else 3, value=4 if common else 20)
    project["module"]["bindings"]["retail"]["_slot"]["data_size"] = 4
    if not common:
        project["sections"][2]["data"] += bytes(4)
    save_object(project)
    save_modules(project)
    for role in ("original", "rebuilt"):
        save_image(project, role, project["addresses"])
    refresh_original_hash(project)
    row = rows(project)["info"]
    assert row.contract_status == "validated" and row.rebuilt_address is None


def test_dll_contributor_cannot_supply_an_owned_public_graph_root(project):
    prepare(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text().replace(map_line("_root", TEXT_VA),
                                              map_line("_root", TEXT_VA, CONTRIBUTOR)))
    assert rows(project)["info"].rebuilt_address is None


def test_typed_i386_import_pointer_has_a_bounded_scalar_slot_role(project):
    name = "__imp__ReadItem@8"
    prepare(project, name=name, native_type=0x20, addend=2)
    assert compare(project).matches
    proof = "\n".join(rows(project)["info"].resolution_provenance)
    assert f"PUBLIC {name} at {DATA_VA + 24:#010x} from DLL import-slot {CONTRIBUTOR}" in proof


@pytest.mark.parametrize("failure", ["REL32", "object", "unprefixed", "empty-suffix", "function-flag",
                                      "ambiguous", "alias", "virtual", "mixed"])
def test_import_slot_role_does_not_relax_other_native_map_or_backing_requirements(project, failure):
    name = "_slot" if failure == "unprefixed" else "__imp_" if failure == "empty-suffix" else "__imp__ReadItem@8"
    prepare(project, name=name, native_type=0x20,
            kind=20 if failure == "REL32" else 6, flag="f" if failure == "function-flag" else "")
    path = project["path"] / "rebuilt.map"
    if failure == "object":
        path.write_text(path.read_text().replace(CONTRIBUTOR, "External.obj"))
    elif failure == "ambiguous":
        path.write_text(path.read_text() + map_line(name, DATA_VA + 24, "OTHER:OTHER.dll", flag="", segment=2))
    elif failure == "alias":
        project["native"].append({"name": name, "type": 0x20})
        save_object(project)
    elif failure in ("virtual", "mixed"):
        image_path = project["path"] / "rebuilt.exe"
        image = bytearray(image_path.read_bytes())
        struct.pack_into("<I", image, 0x1A0 + 16, 24 if failure == "virtual" else 26)
        image_path.write_bytes(image)
    assert rows(project)["info"].rebuilt_address is None


@pytest.mark.parametrize("contributor,accepted", [
    (CONTRIBUTOR, True), ("support.lib:Helpers.DLL", True), ('"SUPPORT:SUPPORT.dll"', True),
    ("SUPPORT.dll", False), ("dir/SUPPORT:SUPPORT.dll", False),
    ("SUPPORT:SUPPORT.dll extra", False), ("SUPPORT:", False),
])
def test_map_parser_preserves_only_explicit_basename_dll_contributor_provenance(tmp_path, contributor, accepted):
    path = tmp_path / "input.map"
    path.write_text(map_line("_slot", DATA_VA, contributor, flag=""))
    record, = parse_msvc_map_symbols(path)
    assert record.object_file is None
    assert (record.dll_contributor is not None) is accepted
    if accepted:
        assert record.dll_contributor == contributor.strip('"')


def test_linked_identity_preserves_contributor_kind():
    obj = MapSymbol(2, DATA_VA, "_slot", False, CONTRIBUTOR)
    dll = replace(obj, object_file=None, dll_contributor=CONTRIBUTOR)
    assert linked_map_identity(obj) != linked_map_identity(dll)
