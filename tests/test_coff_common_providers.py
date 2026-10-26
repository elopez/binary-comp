"""Explicit COMMON providers, using synthetic native objects and references."""
import hashlib
import json
import struct

import pytest

from binary_comp.analyzers.coff_report import load_selection, run_config, score_config
from binary_comp.config import ConfigError
from binary_comp.core.coff import COFF, compare_function, validate_data_binding
from conftest import DATA_VA, TEXT_VA
from test_coff_contract import DATA_FLAGS, coff_bytes, entry, reference


def consumer(size=4, *, referenced=True, symbol=None):
    code = b"\xa1" + bytes(4) + b"\xc3" if referenced else b"\xc3"
    obj = COFF.from_bytes(coff_bytes([
        {"data": code, "relocations": [(1, 1, 6)] if referenced else []}], [
        {"name": "_read", "section": 1, "type": 0x24},
        {"name": "_buffer", "value": size, **(symbol or {})}]))
    return obj, entry("_read", TEXT_VA, len(code))


def provider(payload=struct.pack("<I", 7), *, symbol=None, flags=DATA_FLAGS):
    return COFF.from_bytes(coff_bytes([
        {"name": ".data", "data": payload, "flags": flags}], [
        {"name": "_buffer", "section": 1, **(symbol or {})}]))


def fixture(tmp_path, *, size=4, referenced=True):
    obj, function = consumer(size, referenced=referenced)
    payload = bytes(range(1, size + 1))
    strong = provider(payload)
    code = b"\xa1" + struct.pack("<I", DATA_VA) + b"\xc3" if referenced else b"\xc3"
    image = reference(tmp_path, code, payload + bytes(32 - size))
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": size}}
    return obj, function, strong, bindings, image


@pytest.mark.parametrize("size", [4, 24])
@pytest.mark.parametrize("constant", [False, True])
def test_explicit_strong_provider_checks_complete_scalar_or_array(tmp_path, size, constant):
    obj, function, strong, bindings, image = fixture(tmp_path, size=size)
    if constant:
        bindings["_buffer"] = {"va": hex(DATA_VA), "data_hex": bytes(range(1, size + 1)).hex()}
    with pytest.raises(ValueError, match="Rebuilt data differs"):
        compare_function(obj, function, "sample", bindings, image)
    assert compare_function(obj, function, "sample", bindings, image,
                            common_definitions={"_buffer": strong})[0]
    assert validate_data_binding(obj, "_buffer", bindings, image,
                                 common_definitions={"_buffer": strong}) == (bytes(range(1, size + 1)), 0)
    # Native COMMON and strong records are retained, without synthetic sections.
    assert obj.named["_buffer"]["section"] == 0
    assert len(obj.sections) == 1
    assert strong.named["_buffer"]["section"] == 1


@pytest.mark.parametrize("referenced", [False, True])
@pytest.mark.parametrize("mutation", ["missing_binding", "bare_va", "size", "bytes", "address", "truncated"])
def test_bad_provider_mapping_fails_even_without_a_code_reference(tmp_path, referenced, mutation):
    obj, function, strong, bindings, image = fixture(tmp_path, referenced=referenced)
    if mutation == "missing_binding":
        bindings.clear()
    elif mutation == "bare_va":
        del bindings["_buffer"]["data_size"]
    elif mutation == "size":
        bindings["_buffer"]["data_size"] = 3
    elif mutation == "bytes":
        strong = provider(bytes(4))
    else:
        bindings["_buffer"]["va"] = hex(DATA_VA + (1 if mutation == "address" else 30))
    with pytest.raises(ValueError):
        compare_function(obj, function, "sample", bindings, image,
                         common_definitions={"_buffer": strong})


@pytest.mark.parametrize("symbol", [
    {"value": 0}, {"section": 1, "value": 0}, {"type": 0x24},
    {"storage": 105, "aux": struct.pack("<II", 0, 3) + bytes(10)},
])
def test_provider_mapping_requires_recognized_consumer_common_even_when_unused(tmp_path, symbol):
    obj, function, strong, bindings, image = fixture(tmp_path, referenced=False)
    obj, function = consumer(referenced=False, symbol=symbol)
    with pytest.raises(ValueError, match="recognized consumer COMMON"):
        compare_function(obj, function, "sample", bindings, image,
                         common_definitions={"_buffer": strong})


@pytest.mark.parametrize("mutation", [
    "common", "undefined", "missing", "function", "code", "static", "aux", "void", "weak", "short", "long",
])
def test_provider_must_be_same_name_strong_data_with_exact_extent(tmp_path, mutation):
    obj, function, strong, bindings, image = fixture(tmp_path, referenced=False)
    if mutation in ("common", "undefined", "missing"):
        strong = COFF.from_bytes(coff_bytes([], [{
            "name": "_other" if mutation == "missing" else "_buffer",
            "value": 4 if mutation == "common" else 0}]))
    elif mutation == "code":
        strong = provider(flags=0x60000020)
    elif mutation in ("short", "long"):
        strong = provider(bytes(3 if mutation == "short" else 8))
    else:
        symbol = {"function": {"type": 0x24}, "static": {"storage": 3},
                  "aux": {"aux": bytes(18)}, "void": {"type": 1},
                  "weak": {"storage": 105, "aux": struct.pack("<II", 0, 3) + bytes(10)}}[mutation]
        strong = provider(symbol=symbol)
    with pytest.raises(ValueError, match="strong data definition|extent differs"):
        compare_function(obj, function, "sample", bindings, image,
                         common_definitions={"_buffer": strong})


def pointer_provider(*, data_dependency=False):
    # The complete provider is a pointer with one native DIR32 fixup.
    return COFF.from_bytes(coff_bytes([
        {"name": ".data", "data": bytes(4), "flags": DATA_FLAGS, "relocations": [(0, 1, 6)]},
        {"name": ".rdata" if data_dependency else ".text", "data": b"word" if data_dependency else b"\xc3",
         "flags": DATA_FLAGS if data_dependency else 0x60000020}], [
        {"name": "_buffer", "section": 1},
        {"name": "_target", "section": 2, "type": 0 if data_dependency else 0x20}]))


@pytest.mark.parametrize("mutation", [None, "missing", "wrong", "bare_owned_data"])
def test_provider_fixups_use_explicit_bindings_and_validate_owned_data(tmp_path, mutation):
    obj, function = consumer(referenced=False)
    address = DATA_VA + 8 if mutation == "bare_owned_data" else TEXT_VA + 0x80
    payload = struct.pack("<I", address) + bytes(4) + b"word"
    image = reference(tmp_path, b"\xc3", payload)
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 4}, "_target": {"va": hex(address)}}
    if mutation == "missing":
        del bindings["_target"]
    elif mutation == "wrong":
        bindings["_target"]["va"] = hex(address + 1)
    strong = pointer_provider(data_dependency=mutation == "bare_owned_data")
    if mutation is None:
        assert validate_data_binding(obj, "_buffer", bindings, image,
                                     common_definitions={"_buffer": strong}) == (struct.pack("<I", address), 1)
    else:
        with pytest.raises(ValueError, match="Unresolved relocation|Rebuilt data differs|complete data validation"):
            compare_function(obj, function, "sample", bindings, image,
                             common_definitions={"_buffer": strong})


def test_provider_self_reference_is_not_confused_with_same_name_delegation(tmp_path):
    obj, function = consumer(referenced=False)
    strong = COFF.from_bytes(coff_bytes([
        {"name": ".data", "data": bytes(4), "flags": DATA_FLAGS, "relocations": [(0, 0, 6)]}],
        [{"name": "_buffer", "section": 1}]))
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 4}}
    with pytest.raises(ValueError, match="Cyclic data binding"):
        compare_function(obj, function, "sample", bindings,
                         reference(tmp_path, b"\xc3", struct.pack("<I", DATA_VA)),
                         common_definitions={"_buffer": strong})


def test_weak_reference_to_common_uses_the_same_complete_provider_gate(tmp_path):
    obj = COFF.from_bytes(coff_bytes([
        {"data": b"\xa1" + bytes(4) + b"\xc3", "relocations": [(1, 1, 6)]}], [
        {"name": "_read", "section": 1, "type": 0x24},
        {"name": "_weak", "storage": 105, "aux": struct.pack("<II", 3, 3) + bytes(10)},
        {"name": "_buffer", "value": 4}]))
    payload = struct.pack("<I", 7)
    image = reference(tmp_path, b"\xa1" + struct.pack("<I", DATA_VA) + b"\xc3", payload)
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 4}}
    strong = provider(payload)
    assert compare_function(obj, entry("_read", TEXT_VA, 6), "sample", bindings, image,
                            common_definitions={"_buffer": strong})[0]
    del bindings["_buffer"]["data_size"]
    with pytest.raises(ValueError, match="complete data validation"):
        compare_function(obj, entry("_read", TEXT_VA, 6), "sample", bindings, image,
                         common_definitions={"_buffer": strong})


def write_project(tmp_path, *, referenced=True, data_only=False):
    obj, function, strong, bindings, image = fixture(tmp_path, referenced=referenced)
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "Sample.obj").write_bytes(obj.data)
    (objects / "State.obj").write_bytes(strong.data)
    module = {"name": "Sample", "functions": [] if data_only else [function],
              "bindings": {"sample": bindings},
              "common_definitions": {"sample": {"_buffer": "State.obj"}}}
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps([module]))
    config = tmp_path / "binary-comp.json"
    config.write_text(json.dumps({"coff": {"modules": manifest.name}, "targets": {
        "sample": {"kind": "i386-coff", "object_dir": "objects", "original_exe": str(image.path),
                   "original_sha256": hashlib.sha256(image.data).hexdigest()}}}))
    return config, manifest, module, obj, strong


@pytest.mark.parametrize("data_only", [False, True])
def test_report_records_one_overridden_contribution_and_provider_hash(tmp_path, data_only):
    config, _, _, obj, strong = write_project(tmp_path, data_only=data_only)
    output = tmp_path / "report"
    assert run_config(config, "sample", output) == 0
    result = json.loads((output / "report.json").read_text())
    row, = result["data"]
    assert row == {"version": "sample", "module": "Sample", "symbol": "_buffer", "exact": True,
                   "size": 4, "checked_fixups": 0, "storage": "common_overridden", "object_file_bytes": 0,
                   "provider_object": "State.obj", "provider_file_bytes": 4}
    assert result["inputs"]["objects_sha256"] == {"sample/Sample": hashlib.sha256(obj.data).hexdigest()}
    assert result["inputs"]["provider_objects_sha256"] == {"sample/State.obj": hashlib.sha256(strong.data).hexdigest()}
    assert len(result["code"]) == (0 if data_only else 1)


@pytest.mark.parametrize("mutation", ["bare_va", "wrong_bytes", "not_common", "missing_binding"])
def test_unreferenced_bad_mapping_invalidates_report_and_external_score(tmp_path, mutation):
    config, manifest, module, obj, _ = write_project(tmp_path, referenced=False)
    output = tmp_path / "report"
    output.mkdir()
    for filename in ("report.json", "report.txt"):
        (output / filename).write_text("old green summary")
    if mutation == "bare_va":
        del module["bindings"]["sample"]["_buffer"]["data_size"]
    elif mutation == "missing_binding":
        module["bindings"]["sample"].clear()
    elif mutation == "wrong_bytes":
        (tmp_path / "objects/State.obj").write_bytes(provider(bytes(4)).data)
    else:
        obj, _ = consumer(referenced=False, symbol={"value": 0})
        (tmp_path / "objects/Sample.obj").write_bytes(obj.data)
    manifest.write_text(json.dumps([module]))
    with pytest.raises(ValueError):
        run_config(config, "sample", output)
    assert not (output / "report.json").exists()
    assert not (output / "report.txt").exists()
    external = tmp_path / "candidate.obj"
    external.write_bytes(obj.data)
    with pytest.raises(ValueError):
        score_config(config, "sample", "_read", external)


@pytest.mark.parametrize("mutation", ["no_binding", "wrong_bytes"])
def test_data_only_provider_mapping_cannot_be_skipped(tmp_path, mutation):
    config, manifest, module, _, _ = write_project(tmp_path, data_only=True, referenced=False)
    if mutation == "no_binding":
        module["bindings"]["sample"].clear()
    else:
        (tmp_path / "objects/State.obj").write_bytes(provider(bytes(4)).data)
    manifest.write_text(json.dumps([module]))
    output = tmp_path / "report"
    with pytest.raises(ValueError):
        run_config(config, "sample", output)
    assert not (output / "report.json").exists()


def test_external_candidate_uses_configured_provider_directory_and_retains_common(tmp_path):
    config, _, _, obj, _ = write_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "candidate.obj"
    external.write_bytes(obj.data)
    (outside / "State.obj").write_bytes(provider(bytes(4)).data)
    assert score_config(config, "sample", "_read", external) == 0
    # A candidate that defines its own data cannot retain a stale provider map.
    strong_consumer = COFF.from_bytes(coff_bytes([
        {"data": b"\xa1" + bytes(4) + b"\xc3", "relocations": [(1, 1, 6)]},
        {"name": ".data", "data": bytes(range(1, 5)), "flags": DATA_FLAGS}], [
        {"name": "_read", "section": 1, "type": 0x24}, {"name": "_buffer", "section": 2}]))
    external.write_bytes(strong_consumer.data)
    with pytest.raises(ValueError, match="recognized consumer COMMON"):
        score_config(config, "sample", "_read", external)


@pytest.mark.parametrize("mapping", [
    [], {"unknown": {"_buffer": "State.obj"}}, {"sample": []}, {"sample": {"": "State.obj"}},
    {"sample": {"_buffer": "../State.obj"}}, {"sample": {"_buffer": "/State.obj"}},
    {"sample": {"_buffer": "sub\\State.obj"}}, {"sample": {"_buffer": "State.lib"}},
    {"sample": {"_buffer": "C:State.obj"}},
])
def test_provider_manifest_rejects_invalid_or_unknown_records(tmp_path, mapping):
    config, manifest, module, _, _ = write_project(tmp_path)
    module["common_definitions"] = mapping
    manifest.write_text(json.dumps([module]))
    with pytest.raises(ConfigError):
        load_selection(config, "sample")


@pytest.mark.parametrize("mapping", [[], {"": None}, {"_buffer": object()}, {"_absent": None}])
def test_native_api_rejects_invalid_provider_context(tmp_path, mapping):
    obj, function, _, bindings, image = fixture(tmp_path)
    with pytest.raises(ValueError):
        compare_function(obj, function, "sample", bindings, image, common_definitions=mapping)
