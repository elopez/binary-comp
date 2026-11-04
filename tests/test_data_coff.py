"""Synthetic native contracts and linked data windows, without source parsing."""

import hashlib
import json
from pathlib import Path
import struct

import pytest

from binary_comp.analyzers.coff_data import compare_linked_data
from binary_comp.analyzers.coff_report import load_selection
from binary_comp.cli import main
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_contract import DATA_FLAGS, coff_bytes
from test_exe_coff import PUBLIC_HEADER, map_line


PAYLOAD = b"tag\0"
SYMBOL = "?value@State@@2HA"


def write_json(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def project(tmp_path):
    for name in ("original.exe", "rebuilt.exe"):
        write_tiny_pe(tmp_path / name, data_overrides={0: PAYLOAD})
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "State.obj").write_bytes(coff_bytes([
        {"name": ".data", "data": b"skip" + PAYLOAD, "flags": DATA_FLAGS}],
        [{"name": SYMBOL, "section": 1, "value": 4}]))
    manifest = tmp_path / "modules.json"
    write_json(manifest, [{"name": "State", "bindings": {
        "retail": {SYMBOL: {"va": hex(DATA_VA), "data_size": len(PAYLOAD)}}}}])
    linked_map = tmp_path / "rebuilt.map"
    linked_map.write_text(map_line(SYMBOL, DATA_VA, "State.obj", flag="", segment=2))
    config = tmp_path / "binary-comp.json"
    write_json(config, {"targets": {"retail": {
        "kind": "i386-coff", "original_exe": "original.exe",
        "original_sha256": hashlib.sha256((tmp_path / "original.exe").read_bytes()).hexdigest(),
        "rebuilt_exe": "rebuilt.exe", "map": "rebuilt.map", "object_dir": "objects",
    }}, "coff": {"modules": "modules.json"}})
    return tmp_path


def compare(project):
    modules, targets, _ = load_selection(project / "binary-comp.json", "retail")
    return compare_linked_data(targets["retail"], modules)


def run(project, *args):
    return main(["data", "--config", str(project / "binary-comp.json"), "--target", "retail", *args])


def change_module(project, change):
    path = project / "modules.json"
    modules = json.loads(path.read_text())
    change(modules[0])
    write_json(path, modules)


def refresh_hash(project):
    path = project / "binary-comp.json"
    config = json.loads(path.read_text())
    config["targets"]["retail"]["original_sha256"] = hashlib.sha256((project / "original.exe").read_bytes()).hexdigest()
    write_json(path, config)


def test_complete_native_contract_and_cli_need_no_cpp_parser(project, monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise AssertionError("No C++ parser or build permitted")

    monkeypatch.setattr("binary_comp.cli.require_globals_source", unavailable)
    monkeypatch.setattr("binary_comp.source.cpp.make_cpp_parser", unavailable)
    monkeypatch.setattr("binary_comp.cli.build_source_function_address_map", unavailable)
    monkeypatch.setattr("binary_comp.cli.maybe_build", unavailable)
    result = compare(project)
    row, = result.rows
    assert result.matches
    assert row.owner == "State.obj" and row.native_offset == 4
    assert row.native_section == 1
    assert row.contract_status == "validated" and row.checked_fixups == 0
    assert row.placement_equal is row.raw_equal is True
    assert row.matching_bytes == row.size == 4
    assert row.compared_size == 4
    assert row.original_backing == row.rebuilt_backing == "file-backed"
    output = project / "diagnostic"
    assert run(project, "--output", str(output), "--verbose", "--module", "State") == 0
    assert "Raw-equal file-backed windows: 1/1" in capsys.readouterr().out
    saved = json.loads((output / "report.json").read_text())["targets"]["retail"]["rows"][0]
    assert saved["placement_equal"] and saved["raw_equal"]
    assert (output / "report.txt").is_file()


@pytest.mark.parametrize("changed", [False, True])
def test_map_placement_and_full_raw_bytes_are_independent(project, changed):
    write_tiny_pe(project / "rebuilt.exe", data_overrides={0: b"nope", 8: b"Tag\0" if changed else PAYLOAD})
    (project / "rebuilt.map").write_text(map_line(SYMBOL, DATA_VA + 8, "State.obj", flag=""))
    row, = compare(project).rows
    assert row.placement_equal is False
    assert row.raw_equal is (not changed)
    assert row.size == 4 and row.matching_bytes == (3 if changed else 4)
    assert run(project) == 1


def test_relocations_are_validated_on_object_but_linked_bytes_remain_raw(project):
    (project / "objects/State.obj").write_bytes(coff_bytes([
        {"name": ".data", "data": bytes(4), "flags": DATA_FLAGS, "relocations": [(0, 1, 6)]}],
        [{"name": SYMBOL, "section": 1}, {"name": "_target"}]))
    change_module(project, lambda module: module["bindings"]["retail"].update({"_target": {"va": hex(TEXT_VA)}}))
    write_tiny_pe(project / "original.exe", data_overrides={0: struct.pack("<I", TEXT_VA)})
    write_tiny_pe(project / "rebuilt.exe", data_overrides={0: struct.pack("<I", TEXT_VA + 16)})
    refresh_hash(project)
    row, = compare(project).rows
    assert row.contract_status == "validated" and row.checked_fixups == 1
    assert row.placement_equal is True and row.raw_equal is False
    assert row.matching_bytes == 3


@pytest.mark.parametrize("mutation", ["bytes", "short", "long", "undefined", "foreign", "weak"])
def test_invalid_or_unowned_objects_cannot_receive_linked_equality_credit(project, mutation):
    symbol = {"name": SYMBOL, "section": 1}
    payload = b"bad\0" if mutation == "bytes" else PAYLOAD
    if mutation in ("short", "long"):
        payload = bytes(3 if mutation == "short" else 8)
    elif mutation == "undefined":
        symbol["section"] = 0
    elif mutation == "foreign":
        symbol["name"] = "_different"
    elif mutation == "weak":
        symbol.update(storage=105, aux=struct.pack("<II", 0, 3) + bytes(10))
    (project / "objects/State.obj").write_bytes(coff_bytes([
        {"name": ".data", "data": payload, "flags": DATA_FLAGS}], [symbol]))
    row, = compare(project).rows
    assert row.contract_status != "validated"
    assert row.placement_equal is True
    assert row.raw_equal is None and row.matching_bytes is None
    assert run(project) == 1


@pytest.mark.parametrize("mutation", ["metadata", "label", "no-data-flags", "void", "link-info", "removed", "discardable"])
def test_native_metadata_labels_and_nondata_records_are_not_runtime_data(project, mutation):
    symbol = {"name": SYMBOL, "section": 1}
    flags = DATA_FLAGS
    if mutation == "metadata":
        symbol.update(name=".data", storage=3, aux=struct.pack("<IHHIHBBH", 4, 0, 0, 0, 0, 0, 0, 0))
        change_module(project, lambda module: module["bindings"]["retail"].update({".data": module["bindings"]["retail"].pop(SYMBOL)}))
    elif mutation == "label":
        symbol["storage"] = 6
    elif mutation == "no-data-flags":
        flags = 0
    elif mutation == "void":
        symbol["type"] = 1
    else:
        flags |= {"link-info": 0x200, "removed": 0x800, "discardable": 0x02000000}[mutation]
    (project / "objects/State.obj").write_bytes(coff_bytes([
        {"name": ".data", "data": PAYLOAD, "flags": flags}], [symbol]))
    (project / "rebuilt.map").write_text("Static symbols\n\n" + map_line(symbol["name"], DATA_VA, "State.obj", flag=""))
    row, = compare(project).rows
    assert row.contract_status != "validated"
    assert row.raw_equal is None and row.compared_size == 0


@pytest.mark.parametrize("native_type", [0, 4, 0x14, 0x11])
def test_ordinary_untyped_scalar_and_pointer_records_remain_supported(project, native_type):
    (project / "objects/State.obj").write_bytes(coff_bytes([
        {"name": ".data", "data": PAYLOAD, "flags": DATA_FLAGS}],
        [{"name": SYMBOL, "section": 1, "type": native_type}]))
    assert compare(project).matches


def test_native_bss_contract_can_compare_real_file_backed_zero_windows(project):
    (project / "objects/State.obj").write_bytes(coff_bytes([
        {"name": ".bss", "size": 4, "flags": 0xC0000080}],
        [{"name": SYMBOL, "section": 1}]))
    for name in ("original.exe", "rebuilt.exe"):
        write_tiny_pe(project / name, data_overrides={0: bytes(4)})
    refresh_hash(project)
    row, = compare(project).rows
    assert row.contract_status == "validated"
    assert row.raw_equal is True and row.compared_size == 4
    assert row.original_backing == row.rebuilt_backing == "file-backed"


@pytest.mark.parametrize("role", ["inline", "executable"])
def test_explicit_native_code_section_data_roles_remain_supported(project, role):
    if role == "inline":
        from test_coff_inline_alignment import switch_fixture

        coff, _, bindings, image = switch_fixture(project, tail=b"")
    else:
        from test_coff_executable_data import make_case

        coff, _, bindings, image = make_case(project)
    (project / "objects/State.obj").write_bytes(coff.data)
    (project / "original.exe").write_bytes(Path(image.path).read_bytes())
    (project / "rebuilt.exe").write_bytes((project / "original.exe").read_bytes())
    refresh_hash(project)
    change_module(project, lambda module: module.update(bindings={"retail": bindings}))
    (project / "rebuilt.map").write_text("".join(
        map_line(name, int(binding["va"], 16), "State.obj", flag="")
        for name, binding in bindings.items() if "data_size" in binding))
    result = compare(project)
    assert result.matches
    assert all(row.contract_status == "validated" and row.raw_equal for row in result.rows)


@pytest.mark.parametrize("entries,status", [
    (map_line("value", DATA_VA, "State.obj", flag=""), "missing"),
    (map_line(SYMBOL, DATA_VA, "Other.obj", flag=""), "missing"),
    (map_line(SYMBOL, DATA_VA, "State.obj"), "not-data"),
    (map_line(SYMBOL, DATA_VA, "State.obj", flag="") + map_line(SYMBOL, DATA_VA + 4, "State.obj", flag=""), "ambiguous"),
    (map_line(SYMBOL, DATA_VA, "one/State.obj", flag="") + map_line(SYMBOL, DATA_VA, "two/State.obj", flag=""), "ambiguous"),
])
def test_map_identity_is_exact_and_ambiguity_fails_closed(project, entries, status):
    (project / "rebuilt.map").write_text(entries)
    row, = compare(project).rows
    assert row.status == status and row.raw_equal is None
    assert run(project) == 1


@pytest.mark.parametrize("storage,heading,expected", [
    (2, PUBLIC_HEADER, True), (3, PUBLIC_HEADER, False),
    (2, "Static symbols\n\n", False), (2, "", False),
])
def test_cross_owner_public_resolution_requires_external_native_data(project, storage, heading, expected):
    (project / "objects/State.obj").write_bytes(coff_bytes([
        {"name": ".data", "data": PAYLOAD, "flags": DATA_FLAGS}],
        [{"name": SYMBOL, "section": 1, "storage": storage}]))
    (project / "rebuilt.map").write_text(heading + map_line(SYMBOL, DATA_VA, "Selected.obj", flag=""))
    row, = compare(project).rows
    assert row.raw_equal is (True if expected else None)


def test_duplicate_map_records_deduplicate_and_same_owner_wins(project):
    line = map_line(SYMBOL, DATA_VA, r"library.lib:out\STATE.OBJ", flag="")
    (project / "rebuilt.map").write_text(PUBLIC_HEADER + line + line + map_line(SYMBOL, DATA_VA + 8, "Other.obj", flag=""))
    row, = compare(project).rows
    assert row.raw_equal is True


@pytest.mark.parametrize("image", ["original", "rebuilt"])
@pytest.mark.parametrize("raw_size,backing", [(0, "virtual-zero-fill"), (2, "mixed-file-zero-fill")])
def test_virtual_or_mixed_windows_never_receive_raw_equality(project, image, raw_size, backing):
    path = project / f"{image}.exe"
    raw = bytearray(path.read_bytes())
    struct.pack_into("<I", raw, 0x1a0 + 16, raw_size)
    path.write_bytes(raw)
    refresh_hash(project)
    row, = compare(project).rows
    assert getattr(row, f"{image}_backing") == backing
    assert row.placement_equal is True
    assert row.raw_equal is None and row.matching_bytes is None
    assert run(project) == 1


def test_window_does_not_cross_sections_or_use_overlapping_raw_sections(project):
    raw = bytearray((project / "rebuilt.exe").read_bytes())
    struct.pack_into("<I", raw, 0x178 + 12, 0x2000)
    (project / "rebuilt.exe").write_bytes(raw)
    row, = compare(project).rows
    assert row.rebuilt_backing == "invalid-window" and row.raw_equal is None


def test_no_neighbor_caps_or_scanning_for_missing_private_symbols(project):
    (project / "rebuilt.map").write_text(map_line("_neighbor", DATA_VA + 1, "State.obj", flag=""))
    row, = compare(project).rows
    assert row.status == "missing" and row.rebuilt_address is None
    assert row.raw_equal is None


@pytest.mark.parametrize("provider", [False, True])
def test_common_request_needs_explicit_strong_provider_for_linked_credit(project, provider):
    (project / "objects/State.obj").write_bytes(coff_bytes([], [{"name": SYMBOL, "value": 4}]))
    if provider:
        (project / "objects/Storage.obj").write_bytes(coff_bytes([
            {"name": ".data", "data": PAYLOAD, "flags": DATA_FLAGS}], [{"name": SYMBOL, "section": 1}]))
        change_module(project, lambda module: module.update(common_definitions={"retail": {SYMBOL: "Storage.obj"}}))
        (project / "rebuilt.map").write_text(map_line(SYMBOL, DATA_VA, "Storage.obj", flag=""))
    row, = compare(project).rows
    assert row.raw_equal is (True if provider else None)
    assert row.contract_status == ("validated" if provider else "common-request")
    assert row.owner == ("Storage.obj" if provider else "State.obj")


def test_address_only_layout_only_and_check_only_bindings_add_no_rows(project):
    change_module(project, lambda module: module["bindings"]["retail"].update({
        "_address": {"va": hex(DATA_VA + 4)},
        "_layout": {"va": hex(DATA_VA + 8), "section_layout_only": True},
        "_check": {"va": hex(DATA_VA), "data_size": 4, "data_check_only": True}}))
    # Layout-only/check-only still participate in strict object validation, so
    # a malformed dependency may invalidate the row, but never adds coverage.
    assert len(compare(project).rows) == 1


def test_target_all_finishes_both_diagnostics_and_writes_both(project, capsys):
    path = project / "binary-comp.json"
    config = json.loads(path.read_text())
    config["targets"]["alternate"] = dict(config["targets"]["retail"])
    write_json(path, config)
    change_module(project, lambda module: module["bindings"].update(alternate=module["bindings"]["retail"]))
    (project / "rebuilt.map").write_text(map_line(SYMBOL, DATA_VA + 8, "State.obj", flag=""))
    output = project / "report"
    assert main(["data", "--config", str(path), "--target", "all", "--output", str(output)]) == 1
    assert set(json.loads((output / "report.json").read_text())["targets"]) == {"retail", "alternate"}
    text = capsys.readouterr().out
    assert "Target: retail" in text and "Target: alternate" in text


def test_target_all_focused_module_may_have_contracts_only_in_later_target(project, capsys):
    path = project / "binary-comp.json"
    config = json.loads(path.read_text())
    config["targets"]["alternate"] = dict(config["targets"]["retail"])
    write_json(path, config)
    change_module(project, lambda module: module.update(bindings={"alternate": module["bindings"]["retail"]}))
    output = project / "report"
    assert main(["data", "--config", str(path), "--target", "all", "--module", "State", "--output", str(output)]) == 0
    saved = json.loads((output / "report.json").read_text())["targets"]
    assert saved["retail"]["rows"] == [] and len(saved["alternate"]["rows"]) == 1
    assert "No configured complete data contracts for this target" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["retail", "all"])
def test_empty_selection_is_an_error_and_clears_stale_reports(project, target):
    change_module(project, lambda module: module.update(bindings={}))
    output = project / "report"
    output.mkdir()
    for name in ("report.json", "report.txt"):
        (output / name).write_text("stale")
    assert main(["data", "--config", str(project / "binary-comp.json"), "--target", target, "--output", str(output)]) == 2
    assert not (output / "report.json").exists() and not (output / "report.txt").exists()


@pytest.mark.parametrize("args", [
    ["--globals-source", "unused.cpp"], ["--section", ".data"], ["--address", "0x402000"],
    ["--size", "4"], ["--find-missing"], ["--min-address", "0"], ["--max-address", "0"],
    ["--skip-range", "0:4"],
])
def test_unsupported_native_options_fail_explicitly(project, args, capsys):
    assert run(project, *args) == 2
    assert "unsupported" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["digest", "object", "map", "extent"])
def test_required_inputs_and_malformed_contracts_are_errors_and_clear_stale_output(project, failure):
    if failure == "digest":
        (project / "original.exe").write_bytes(b"bad")
    elif failure == "object":
        (project / "objects/State.obj").unlink()
    elif failure == "map":
        (project / "rebuilt.map").unlink()
    else:
        change_module(project, lambda module: module["bindings"]["retail"][SYMBOL].update(data_hex=PAYLOAD.hex()))
    output = project / "report"
    output.mkdir()
    for name in ("report.json", "report.txt"):
        (output / name).write_text("stale")
    assert run(project, "--output", str(output)) == 2
    assert not (output / "report.json").exists() and not (output / "report.txt").exists()


@pytest.mark.parametrize("failure", ["missing", "malformed", "unknown-target"])
def test_failed_config_selection_clears_stale_output(project, failure):
    if failure == "missing":
        (project / "binary-comp.json").unlink()
    elif failure == "malformed":
        (project / "binary-comp.json").write_text("{")
    else:
        path = project / "binary-comp.json"
        config = json.loads(path.read_text())
        config["targets"] = {}
        write_json(path, config)
    output = project / "report"
    output.mkdir()
    for name in ("report.json", "report.txt"):
        (output / name).write_text("stale")
    assert run(project, "--output", str(output)) == 2
    assert not (output / "report.json").exists() and not (output / "report.txt").exists()
