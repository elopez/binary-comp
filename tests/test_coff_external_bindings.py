"""Selected-object external identity checks using synthetic COFF and PE data."""
import hashlib
import json
import struct

import pytest

from binary_comp.analyzers.coff_report import run_config, score_config
from binary_comp.core.coff import COFF
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_common_providers import consumer, pointer_provider
from test_coff_contract import DATA_FLAGS, coff_bytes


def call_object(function, *, storage=2, defined=False, weak=False, referenced=True):
    code = b"\xe8" + bytes(4) + b"\xc3" if referenced else b"\xc3"
    sections = [{"data": code, "relocations": [(1, 1, 0x14)] if referenced else []}]
    symbols = [{"name": function, "section": 1, "type": 0x20}]
    if defined:
        sections.append({"data": b"\xc3"})
    if weak:
        symbols.append({"name": "_optional", "storage": 105,
                        "aux": struct.pack("<II", 3, 3) + bytes(10)})
    symbols.append({"name": "_shared", "storage": storage,
                    "section": 2 if defined else 0, "type": 0x20})
    return COFF.from_bytes(coff_bytes(sections, symbols))


def write_project(tmp_path, *, conflicts=("retail",), storage=2, defined=False,
                  weak=False, referenced=True, targets=("retail",)):
    modules = []
    for name, function in [("First", "_first"), ("Second", "_second")]:
        modules.append({"name": name, "functions": [{"label": function, "symbol": function}],
                        "bindings": {}})
    target_config = {}
    for version_index, version in enumerate(targets):
        objects = tmp_path / version
        objects.mkdir()
        text = bytearray(b"\x90" * 0x100)
        for index, module in enumerate(modules):
            origin = TEXT_VA + index * 0x10
            target = TEXT_VA + 0x80 + version_index * 0x20
            if version in conflicts:
                target += index * 0x10
            function = module["functions"][0]
            function[version] = {"va": hex(origin), "size": 6 if referenced else 1}
            bindings = {"_shared": {"va": hex(target)}}
            if weak:
                # This optional spelling is deliberately not the resolved target.
                bindings["_optional"] = {"va": hex(TEXT_VA + 0x180 + index)}
            module["bindings"][version] = bindings
            obj = call_object(function["symbol"], storage=storage, defined=defined,
                              weak=weak, referenced=referenced)
            (objects / (module["name"] + ".obj")).write_bytes(obj.data)
            code = (b"\xe8" + struct.pack("<i", target - origin - 5) + b"\xc3"
                    if referenced else b"\xc3")
            text[index * 0x10:index * 0x10 + len(code)] = code
        original = tmp_path / (version + ".exe")
        write_tiny_pe(original, bytes(text))
        target_config[version] = {"kind": "i386-coff", "object_dir": version,
                                  "original_exe": original.name,
                                  "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps(modules))
    config = tmp_path / "binary-comp.json"
    config.write_text(json.dumps({"targets": target_config, "coff": {"modules": manifest.name}}))
    return config, manifest, modules


@pytest.mark.parametrize("defined", [False, True])
def test_individually_exact_functions_cannot_disagree_on_external_identity(tmp_path, defined):
    config, _, _ = write_project(tmp_path, defined=defined)
    output = tmp_path / "report"
    for name in ("First", "Second"):
        assert run_config(config, "retail", output, module_names=[name]) == 0
        row, = json.loads((output / "report.json").read_text())["code"]
        assert row["relocated_bytes"]["exact"] and row["checked_fixups"] == 1
    with pytest.raises(ValueError, match=r"Conflicting external binding for retail _shared: First=.*Second="):
        run_config(config, "retail", output)
    assert not (output / "report.json").exists()
    assert not (output / "report.txt").exists()


def test_agreeing_shared_external_bindings_keep_both_exact_bodies(tmp_path):
    config, _, _ = write_project(tmp_path, conflicts=())
    output = tmp_path / "report"
    assert run_config(config, "retail", output) == 0
    result = json.loads((output / "report.json").read_text())
    assert len(result["code"]) == 2
    assert all(row["relocated_bytes"]["exact"] for row in result["code"])
    assert "within each selected target" in result["semantics"]["external_bindings"]


@pytest.mark.parametrize("storage", [3, 6])
def test_static_and_local_native_names_remain_per_object(tmp_path, storage):
    config, _, _ = write_project(tmp_path, storage=storage, defined=True)
    assert run_config(config, "retail", tmp_path / "report") == 0


@pytest.mark.parametrize("filter_kind", ["module", "function", "file"])
def test_filters_check_only_selected_objects(tmp_path, filter_kind):
    config, _, _ = write_project(tmp_path)
    options = {"module": {"module_names": ["First"]}, "function": {"function_name": "_first"},
               "file": {"file_filter": "First"}}[filter_kind]
    # Unselected object files need not be available.
    (tmp_path / "retail/Second.obj").unlink()
    output = tmp_path / "report"
    assert run_config(config, "retail", output, **options) == 0
    assert {row["module"] for row in json.loads((output / "report.json").read_text())["code"]} == {"First"}


def test_score_does_not_load_unselected_modules_or_claim_their_consistency(tmp_path):
    config, _, _ = write_project(tmp_path)
    candidate = tmp_path / "candidate.obj"
    candidate.write_bytes((tmp_path / "retail/First.obj").read_bytes())
    (tmp_path / "retail/First.obj").unlink()
    (tmp_path / "retail/Second.obj").unlink()
    assert score_config(config, "retail", "_first", candidate, module_names=["First"]) == 0


def test_targets_have_independent_external_address_registries(tmp_path):
    config, _, _ = write_project(tmp_path, conflicts=(), targets=("retail", "debug"))
    output = tmp_path / "report"
    assert run_config(config, "all", output) == 0
    result = json.loads((output / "report.json").read_text())
    assert len(result["code"]) == 4
    assert {row["version"] for row in result["code"]} == {"retail", "debug"}


def test_conflict_in_one_target_does_not_reject_an_unselected_target(tmp_path):
    config, _, _ = write_project(tmp_path, conflicts=("debug",), targets=("retail", "debug"))
    output = tmp_path / "report"
    assert run_config(config, "retail", output) == 0
    with pytest.raises(ValueError, match="Conflicting external binding for debug _shared"):
        run_config(config, "all", output)
    assert not (output / "report.json").exists()


@pytest.mark.parametrize("conflict", [False, True])
def test_weak_spellings_are_local_but_native_strong_fallback_names_agree(tmp_path, conflict):
    config, _, _ = write_project(tmp_path, conflicts=("retail",) if conflict else (), weak=True)
    if conflict:
        with pytest.raises(ValueError, match="Conflicting external binding for retail _shared"):
            run_config(config, "retail", tmp_path / "report")
    else:
        # _optional has different addresses; it is a weak name, not the bound fallback.
        assert run_config(config, "retail", tmp_path / "report") == 0


def test_manifest_code_addresses_participate_without_an_explicit_binding(tmp_path):
    config, manifest, modules = write_project(tmp_path)
    second = modules[1]
    second["functions"][0] = {"symbol": "_shared", "label": "definition",
                               "retail": {"va": hex(TEXT_VA + 0x90), "size": 1}}
    second["bindings"]["retail"].clear()
    (tmp_path / "retail/Second.obj").write_bytes(coff_bytes([
        {"data": b"\xc3"}], [{"name": "_shared", "section": 1, "type": 0x20}]))
    original = tmp_path / "retail.exe"
    text = bytearray(original.read_bytes())
    text[0x200 + 0x90] = 0xC3
    original.write_bytes(text)
    settings = json.loads(config.read_text())
    settings["targets"]["retail"]["original_sha256"] = hashlib.sha256(text).hexdigest()
    config.write_text(json.dumps(settings))
    manifest.write_text(json.dumps(modules))
    for name in ("First", "Second"):
        assert run_config(config, "retail", tmp_path / "report", module_names=[name]) == 0
    with pytest.raises(ValueError, match="Conflicting external binding for retail _shared"):
        run_config(config, "retail", tmp_path / "report")


def test_unreferenced_native_external_assertions_are_still_checked(tmp_path):
    config, _, _ = write_project(tmp_path, referenced=False)
    with pytest.raises(ValueError, match="Conflicting external binding for retail _shared"):
        run_config(config, "retail", tmp_path / "report")


@pytest.mark.parametrize("common", [False, True])
@pytest.mark.parametrize("conflict", [False, True])
def test_data_only_definitions_and_common_requests_share_external_identity(tmp_path, common, conflict):
    config, manifest, modules = write_project(tmp_path, referenced=False)
    for index, module in enumerate(modules):
        module["functions"] = []
        address = DATA_VA + (8 * index if conflict else 0)
        module["bindings"]["retail"] = {"_shared": {"va": hex(address), "data_size": 4}}
        obj = coff_bytes([] if common else [{"name": ".data", "data": bytes(4), "flags": DATA_FLAGS}],
                         [{"name": "_shared", "value": 4 if common else 0,
                           "section": 0 if common else 1}])
        (tmp_path / "retail" / (module["name"] + ".obj")).write_bytes(obj)
    original = tmp_path / "retail.exe"
    write_tiny_pe(original, b"\xc3", {0: bytes(32)})
    settings = json.loads(config.read_text())
    settings["targets"]["retail"]["original_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    config.write_text(json.dumps(settings))
    manifest.write_text(json.dumps(modules))
    output = tmp_path / "report"
    for name in ("First", "Second"):
        assert run_config(config, "retail", output, module_names=[name]) == 0
    if conflict:
        with pytest.raises(ValueError, match="Conflicting external binding for retail _shared"):
            run_config(config, "retail", output)
    else:
        assert run_config(config, "retail", output) == 0
        result = json.loads((output / "report.json").read_text())
        assert result["code"] == []
        assert len(result["data"]) == 2
        assert all(row["exact"] for row in result["data"])


def test_different_names_may_share_one_address(tmp_path):
    config, manifest, modules = write_project(tmp_path, conflicts=())
    second = modules[1]
    binding = second["bindings"]["retail"].pop("_shared")
    second["bindings"]["retail"]["_alias"] = binding
    raw = (tmp_path / "retail/Second.obj").read_bytes().replace(b"_shared\0", b"_alias\0\0")
    (tmp_path / "retail/Second.obj").write_bytes(raw)
    manifest.write_text(json.dumps(modules))
    assert run_config(config, "retail", tmp_path / "report") == 0


@pytest.mark.parametrize("kind", ["absolute", "debug", "manifest_only"])
def test_nonaddress_special_records_and_manifest_only_names_are_not_coalesced(tmp_path, kind):
    config, manifest, modules = write_project(tmp_path, conflicts=())
    for index, module in enumerate(modules):
        obj = call_object(module["functions"][0]["symbol"])
        if kind != "manifest_only":
            symbols = [dict(symbol) for symbol in obj.symbols.values()]
            symbols.append({"name": "_constant", "section": -1 if kind == "absolute" else -2, "value": 7})
            obj = COFF.from_bytes(coff_bytes(obj.sections, symbols))
        (tmp_path / "retail" / (module["name"] + ".obj")).write_bytes(obj.data)
        module["bindings"]["retail"]["_constant"] = {"va": hex(index + 7)}
    manifest.write_text(json.dumps(modules))
    assert run_config(config, "retail", tmp_path / "report") == 0


def test_explicit_provider_external_dependencies_participate_in_the_registry(tmp_path):
    config, manifest, modules = write_project(tmp_path)
    first = modules[0]
    first["functions"][0]["symbol"] = first["functions"][0]["label"] = "_read"
    first["functions"][0]["retail"]["size"] = 1
    first["bindings"]["retail"] = {"_buffer": {"va": hex(DATA_VA), "data_size": 4},
                                       "_target": {"va": hex(TEXT_VA + 0x80)}}
    first["common_definitions"] = {"retail": {"_buffer": "State.obj"}}
    obj, _ = consumer(referenced=False)
    (tmp_path / "retail/First.obj").write_bytes(obj.data)
    (tmp_path / "retail/State.obj").write_bytes(pointer_provider().data)
    second = modules[1]
    second["bindings"]["retail"] = {"_target": {"va": hex(TEXT_VA + 0x90)}}
    obj = call_object("_second")
    raw = obj.data.replace(b"_shared\0", b"_target\0")
    (tmp_path / "retail/Second.obj").write_bytes(raw)
    original = tmp_path / "retail.exe"
    code = b"\xc3" + b"\x90" * 15 + b"\xe8" + struct.pack("<i", 0x90 - 0x10 - 5) + b"\xc3"
    write_tiny_pe(original, code, {0: struct.pack("<I", TEXT_VA + 0x80)})
    settings = json.loads(config.read_text())
    settings["targets"]["retail"]["original_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    config.write_text(json.dumps(settings))
    manifest.write_text(json.dumps(modules))
    for name in ("First", "Second"):
        assert run_config(config, "retail", tmp_path / "report", module_names=[name]) == 0
    with pytest.raises(ValueError, match=r"_target: First provider State.obj=.*Second="):
        run_config(config, "retail", tmp_path / "report")
