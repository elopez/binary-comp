"""Explicit absolute scalar terminals, using synthetic provider objects only."""

import hashlib
import json
import struct

import pytest

from binary_comp.config import ConfigError
from conftest import TEXT_VA
from test_coff_contract import coff_bytes
from test_data_coff_graph import (
    compare, project, refresh_original_hash, rows, save_image, save_modules,
    save_object, word,
)
from test_exe_coff import map_line


def configure(state, definitions):
    path = state["path"] / "binary-comp.json"
    config = json.loads(path.read_text())
    config["targets"]["retail"]["absolute_definitions"] = definitions
    path.write_text(json.dumps(config))


def provider(state, records=None, *, filename="Constants.obj", value=0):
    records = records if records is not None else [{"name": "_limit", "section": -1, "value": value}]
    path = state["path"] / "objects" / filename
    path.write_bytes(coff_bytes([], records))
    return path


def prepare(state, value=0, addend=0, kind=6):
    state["native"][6].update(name="_limit", type=0)
    bindings = state["module"]["bindings"]["retail"]
    bindings.pop("_runtime")
    bindings["_limit"] = {"va": hex(value)}
    helper = state["sections"][1]
    helper["data"] = helper["data"][:5] + b"\xba" + word(addend) + helper["data"][10:]
    helper["relocations"][1] = (6, 6, kind)
    state["addresses"][6] = value
    save_object(state)
    save_modules(state)
    for role in ("original", "rebuilt"):
        save_image(state, role, state["addresses"])
    refresh_original_hash(state)
    path = state["path"] / "rebuilt.map"
    path.write_text(path.read_text().replace(map_line("_runtime", TEXT_VA + 0x120, "Runtime.obj"), ""))
    configure(state, {"_limit": "Constants.obj"})
    return provider(state, value=value)


@pytest.mark.parametrize("value,addend,kind", [(0, 0, 6), (0, 7, 6), (0x1234, -3, 6), (0x1234, 5, 20)])
def test_exact_provider_value_and_native_addend_are_fixed_terminals(project, value, addend, kind):
    path = prepare(project, value, addend, kind)
    summary = compare(project)
    assert summary.matches and len(summary.rows) == 3
    proof = "\n".join(rows(project)["info"].resolution_provenance)
    assert f"ABS _limit at {value:#010x}" in proof
    assert f"addend={addend & 0xFFFFFFFF:#x}" in proof
    assert str(path) in proof and "storage=2 section=-1 type=0" in proof
    assert hashlib.sha256(path.read_bytes()).hexdigest() in proof


def test_actual_provider_value_is_independent_of_original_contract_binding(project):
    prepare(project, value=3)
    provider(project, value=9)
    actual = list(project["addresses"])
    actual[6] = 9
    save_image(project, "rebuilt", actual)
    assert compare(project).matches
    provider(project, value=3)
    assert rows(project)["info"].rebuilt_address is None


@pytest.mark.parametrize("mutation", [
    {"storage": 3}, {"section": 0}, {"section": 0, "value": 4}, {"section": -2}, {"type": 4},
    {"type": 0x20}, {"aux": bytes(18)}, {"name": "_other"},
    {"storage": 105, "aux": struct.pack("<II10x", 0, 3)},
])
def test_nonabsolute_private_typed_weak_auxiliary_or_wrong_name_provider_is_rejected(project, mutation):
    prepare(project)
    provider(project, [{"name": "_limit", "section": -1, **mutation}])
    row = rows(project)["info"]
    assert row.rebuilt_address is None and row.raw_equal is None
    assert "Invalid absolute provider" in row.detail


@pytest.mark.parametrize("failure", ["duplicate", "missing", "malformed", "unconfigured", "typed-source"])
def test_missing_ambiguous_or_unsupported_absolute_proofs_do_not_get_credit(project, failure):
    path = prepare(project)
    if failure == "duplicate":
        provider(project, [{"name": "_limit", "section": -1}] * 2)
    elif failure == "missing":
        path.unlink()
    elif failure == "malformed":
        path.write_bytes(b"invalid")
    elif failure == "unconfigured":
        configure(project, {})
    else:
        project["native"][6]["type"] = 0x20
        save_object(project)
    assert rows(project)["info"].rebuilt_address is None


@pytest.mark.parametrize("value,flag,resolved", [(0, "", True), (1, "", False), (0, "f", False)])
def test_any_conflicting_map_value_or_function_role_rejects_absolute_terminal(project, value, flag, resolved):
    prepare(project)
    path = project["path"] / "rebuilt.map"
    path.write_text(path.read_text() + map_line("_limit", value, "Other.obj", flag=flag))
    assert (rows(project)["info"].rebuilt_address is not None) is resolved


def test_conflicting_explicit_provider_objects_cannot_silently_override_one_another(project):
    prepare(project)
    provider(project, [{"name": "_other", "section": -1},
                       {"name": "_limit", "section": -1, "value": 1}], filename="Other.obj")
    configure(project, {"_limit": "Constants.obj", "_other": "Other.obj"})
    row = rows(project)["info"]
    assert row.rebuilt_address is None and "Conflicting configured absolute provider" in row.detail


@pytest.mark.parametrize("definitions", [[], {"": "Constants.obj"}, {"_limit": "dir/Constants.obj"},
                                          {"_limit": "Constants.exe"}, {"_limit": 0}])
def test_absolute_provider_config_requires_native_names_and_object_basenames(project, definitions):
    configure(project, definitions)
    with pytest.raises(ConfigError, match="absolute_definitions"):
        compare(project)
