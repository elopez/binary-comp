"""Portable native COFF CLI fixtures, independent of any compiler installation."""
import hashlib
import json
import struct

import pytest

from binary_comp.analyzers.coff_report import load_selection, report
from binary_comp.cli import main
from binary_comp.config import load_project_target
from conftest import DATA_VA, TEXT_VA, write_tiny_pe


pytest.importorskip("capstone")


def write_object(path, *, code=None, value=7, symbol_type=0x20, storage=2,
                 extra_symbols=(), symbol_name=b"answer"):
    # answer: mov eax,[number]; ret. The original has a resolved address;
    # this object has a native DIR32 fixup to its complete data definition.
    code = bytes.fromhex("a1 00 00 00 00 c3") if code is None else code
    header_size = 20 + 2 * 40
    data_at = header_size + len(code)
    reloc_at = data_at + 4
    symbols_at = reloc_at + 10
    raw = bytearray(struct.pack("<HHIIIHH", 0x14c, 2, 0, symbols_at, 2 + len(extra_symbols), 0, 0))
    for name, size, offset, relocs, count, flags in (
            (b".text", len(code), header_size, reloc_at, 1, 0x60001020),
            (b".data", 4, data_at, 0, 0, 0xc0000040)):
        raw += struct.pack("<8sIIIIIIHHI", name, 0, 0, size, offset, relocs, 0, count, 0, flags)
    raw += code + struct.pack("<I", value) + struct.pack("<IIH", 1, 1, 6)
    raw += struct.pack("<8sIhHBB", symbol_name, 0, 1, symbol_type, storage, 0)
    raw += struct.pack("<8sIhHBB", b"number", 0, 2, 0, 2, 0)
    for symbol in extra_symbols:
        raw += struct.pack("<8sIhHBB", *symbol, 0)
    raw += struct.pack("<I", 4)
    path.write_bytes(raw)


@pytest.fixture
def project(tmp_path):
    original = tmp_path / "original.exe"
    write_tiny_pe(original, b"\xa1" + struct.pack("<I", DATA_VA + 0x10) + b"\xc3")
    targets = {}
    for name in ("retail", "debug"):
        objects = tmp_path / name
        objects.mkdir()
        write_object(objects / "Sample.obj")
        targets[name] = {"kind": "i386-coff", "original_exe": "original.exe",
                         "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                         "object_dir": name}
    modules = [{"name": "Sample", "functions": [{"label": "read_value", "symbol": "answer",
                **{name: {"va": hex(TEXT_VA), "size": 6} for name in targets}}],
                "bindings": {name: {"number": {"va": hex(DATA_VA + 0x10), "data_size": 4}}
                             for name in targets}}]
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps(modules))
    config = tmp_path / "binary-comp.json"
    config.write_text(json.dumps({"targets": targets, "coff": {"modules": "modules.json"}}))
    return config, manifest, tmp_path / "report"


def arguments(project, command="report", target="all"):
    config, _, output = project
    return [command, "--config", str(config), "--target", target, "--no-build", "--output", str(output)]


def test_coff_targets_need_no_sources_rebuilt_executable_or_map(project):
    _, target = load_project_target(project[0], "retail")
    assert target.kind == "i386-coff"
    assert target.source_dirs == ()
    assert target.rebuilt_exe == target.map_path == ""
    assert target.object_dir.endswith("retail")


def test_report_aggregates_arbitrary_targets_and_preserves_strict_metrics(project, capsys):
    assert main(arguments(project) + ["--module", "Sample"]) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert {row["version"] for row in result["code"]} == {"retail", "debug"}
    assert len(result["code"]) == len(result["data"]) == 2
    for row in result["code"]:
        assert not row["raw_bytes"]["exact"]
        assert row["relocated_bytes"]["exact"]
        assert row["checked_fixups"] == 1
        assert row["body_size"] == row["decoded_bytes"]["rebuilt"] == 6
        assert (project[2] / row["review"]).is_file()
    assert all(row["exact"] and row["size"] == 4 for row in result["data"])
    assert "function: 0/2 raw-byte equal; 2/2 relocated-byte equal" in capsys.readouterr().out


def test_compare_prints_native_full_review_without_disassembly_exports(project, capsys):
    assert main(arguments(project, "compare", "retail") + ["read_value"]) == 0
    text = capsys.readouterr().out
    assert "Original vs raw COFF body" in text
    assert "Left: rebuilt after explicit fixups" in text
    assert "does not establish equality" in text


def test_changed_defined_data_is_checked_and_mismatches_fail_the_cli(project, capsys):
    write_object(project[0].parent / "retail/Sample.obj", value=8)
    assert main(arguments(project, target="retail")) == 1
    result = json.loads((project[2] / "report.json").read_text())
    assert not result["data"][0]["exact"]
    assert "Rebuilt data differs" in result["data"][0]["error"]
    assert "relocated_bytes" not in result["code"][0]
    assert "FAIL DATA" in capsys.readouterr().out


def test_rejected_extent_keeps_diagnostic_metrics_and_nonzero_exit(project, capsys):
    write_object(project[0].parent / "retail/Sample.obj", code=bytes.fromhex("a1 00 00 00 00 c3 c3"))
    assert main(arguments(project, target="retail")) == 1
    result = json.loads((project[2] / "report.json").read_text())
    row = result["code"][0]
    assert "Non-NOP" in row["error"]
    assert row["diagnostic_only"]
    assert row["coff_window_size"] == row["relocated_bytes"]["rebuilt_size"] == 7
    assert "Strict matcher error" in (project[2] / row["review"]).read_text()
    assert "diagnostic complete COFF section" in capsys.readouterr().out


@pytest.mark.parametrize("selection", [["--module", "missing"], ["missing_function"]])
def test_invalid_selection_removes_stale_success_summaries(project, capsys, selection):
    assert main(arguments(project)) == 0
    command = "compare" if selection == ["missing_function"] else "report"
    assert main(arguments(project, command, "retail") + selection) == 2
    assert not (project[2] / "report.json").exists()
    assert not (project[2] / "report.txt").exists()
    assert "error:" in capsys.readouterr().err


def test_original_hash_failure_removes_stale_summary(project, capsys):
    assert main(arguments(project)) == 0
    config = json.loads(project[0].read_text())
    config["targets"]["retail"]["original_sha256"] = "0" * 64
    project[0].write_text(json.dumps(config))
    assert main(arguments(project, target="retail")) == 2
    assert not (project[2] / "report.json").exists()
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_missing_object_does_not_reuse_previous_success(project, capsys):
    assert main(arguments(project)) == 0
    (project[0].parent / "retail/Sample.obj").unlink()
    assert main(arguments(project, target="retail")) == 2
    assert not (project[2] / "report.json").exists()
    assert "error:" in capsys.readouterr().err


def test_optional_source_root_resolves_relative_to_config(project):
    source_root = project[0].parent / "sources"
    source_root.mkdir()
    source = source_root / "sample.cpp"
    source.write_text("int answer() { return 7; }\n")
    modules = json.loads(project[1].read_text())
    modules[0]["source"] = "sample.cpp"
    project[1].write_text(json.dumps(modules))
    config = json.loads(project[0].read_text())
    config["coff"]["source_root"] = "sources"
    project[0].write_text(json.dumps(config))
    assert main(arguments(project, target="retail")) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert result["inputs"]["sources_sha256"] == {
        "sample.cpp": hashlib.sha256(source.read_bytes()).hexdigest()}


@pytest.mark.parametrize("binding", [42, {"va": "bad address"},
                                     {"va": "0x402010", "data_size": True},
                                     {"va": "0x402010", "data_hex": []}])
def test_invalid_manifest_bindings_produce_configuration_errors(project, capsys, binding):
    modules = json.loads(project[1].read_text())
    modules[0]["bindings"]["retail"]["number"] = binding
    project[1].write_text(json.dumps(modules))
    assert main(arguments(project, target="retail")) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["compare", "report"])
@pytest.mark.parametrize("failure", ["unknown target", "malformed JSON", "malformed targets"])
def test_early_cli_failures_clear_owned_summaries_only(project, capsys, command, failure):
    assert main(arguments(project)) == 0
    result = json.loads((project[2] / "report.json").read_text())
    review = project[2] / result["code"][0]["review"]
    extra = ["read_value"] if command == "compare" else []
    target = "retail"
    if failure == "unknown target":
        target = "missing"
    elif failure == "malformed JSON":
        project[0].write_text("{broken")
    else:
        project[0].write_text(json.dumps({"targets": []}))
    assert main(arguments(project, command, target) + extra) == 2
    assert not (project[2] / "report.json").exists()
    assert not (project[2] / "report.txt").exists()
    assert review.is_file()
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["duplicate symbol", "duplicate across roles",
                                     "duplicate window", "overlap across modules"])
def test_counted_code_must_be_canonical_across_modules_and_roles(project, capsys, mutation):
    modules = json.loads(project[1].read_text())
    duplicate = json.loads(json.dumps(modules[0]["functions"][0]))
    if mutation == "duplicate across roles":
        modules[0]["code_labels"] = [duplicate]
    elif mutation == "overlap across modules":
        duplicate["symbol"] = "second"
        duplicate["retail"]["va"] = hex(TEXT_VA + 2)
        modules.append({"name": "Other", "functions": [duplicate]})
    else:
        if mutation == "duplicate window":
            duplicate["symbol"] = "alias"
        modules[0]["functions"].append(duplicate)
    project[1].write_text(json.dumps(modules))
    modules, targets, _ = load_selection(project[0], "retail")
    with pytest.raises(ValueError, match="Duplicate counted|Overlapping original"):
        report(modules, targets, ["retail"], project[2])
    assert main(arguments(project, target="retail")) == 2
    assert not (project[2] / "report.json").exists()
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["label as function", "function as label", "local case as label"])
def test_native_symbol_roles_are_enforced_for_api_and_cli(project, capsys, mutation):
    modules = json.loads(project[1].read_text())
    if mutation == "label as function":
        write_object(project[0].parent / "retail/Sample.obj", symbol_type=0, storage=6)
        expected = "native function type"
    else:
        modules[0]["code_labels"] = modules[0].pop("functions")
        expected = "native storage 6/type 0"
        if mutation == "local case as label":
            write_object(project[0].parent / "retail/Sample.obj",
                         extra_symbols=((b"case", 5, 1, 0, 6),))
            entry = modules[0]["code_labels"][0]
            entry["symbol"] = "case"
            entry["retail"] = {"va": hex(TEXT_VA + 5), "size": 1}
            expected = "Local function labels"
    project[1].write_text(json.dumps(modules))
    modules, targets, _ = load_selection(project[0], "retail")
    with pytest.raises(ValueError, match=expected):
        report(modules, targets, ["retail"], project[2])
    assert main(arguments(project, target="retail")) == 2
    assert not (project[2] / "report.json").exists()
    assert "error:" in capsys.readouterr().err


def test_address_only_aliases_preserve_one_canonical_count(project):
    modules = json.loads(project[1].read_text())
    modules[0]["bindings"]["retail"]["shared_alias"] = {"va": hex(TEXT_VA)}
    project[1].write_text(json.dumps(modules))
    assert main(arguments(project, target="retail")) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert len(result["code"]) == 1
    assert result["code"][0]["kind"] == "function"
    assert result["code"][0]["relocated_bytes"]["exact"]


def test_native_eh_label_is_counted_only_in_its_own_role(project):
    write_object(project[0].parent / "retail/Sample.obj", symbol_type=0, storage=6)
    modules = json.loads(project[1].read_text())
    modules[0]["code_labels"] = modules[0].pop("functions")
    project[1].write_text(json.dumps(modules))
    assert main(arguments(project, target="retail")) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert len(result["code"]) == 1
    assert result["code"][0]["kind"] == "label"
    assert result["code"][0]["relocated_bytes"]["exact"]


def test_local_code_label_names_are_scoped_to_their_objects(project):
    root = project[0].parent
    code = b"\xa1" + struct.pack("<I", DATA_VA + 0x10) + b"\xc3"
    write_tiny_pe(root / "original.exe", code + b"\x90" * 10 + code)
    config = json.loads(project[0].read_text())
    config["targets"]["retail"]["original_sha256"] = hashlib.sha256(
        (root / "original.exe").read_bytes()).hexdigest()
    project[0].write_text(json.dumps(config))
    modules = []
    for name, offset in (("First", 0), ("Second", 16)):
        write_object(root / "retail" / (name + ".obj"), symbol_type=0, storage=6,
                     symbol_name=b"$L1")
        modules.append({"name": name, "code_labels": [
            {"label": "cleanup", "symbol": "$L1", "retail": {"va": hex(TEXT_VA + offset), "size": 6}}],
            "bindings": {"retail": {"number": {"va": hex(DATA_VA + 0x10), "data_size": 4}}}})
    project[1].write_text(json.dumps(modules))
    modules, targets, _ = load_selection(project[0], "retail")
    assert report(modules, targets, ["retail"], project[2]) == 0
    assert main(arguments(project, target="retail")) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert len(result["code"]) == 2
    assert all(row["kind"] == "label" and row["relocated_bytes"]["exact"]
               for row in result["code"])


@pytest.mark.parametrize("symbol_type, accepted", [(0x24, True), (0x30, False)])
def test_analyzer_uses_native_derived_function_type(project, capsys, symbol_type, accepted):
    write_object(project[0].parent / "retail/Sample.obj", symbol_type=symbol_type)
    modules, targets, _ = load_selection(project[0], "retail")
    if accepted:
        assert report(modules, targets, ["retail"], project[2]) == 0
        assert main(arguments(project, target="retail")) == 0
    else:
        with pytest.raises(ValueError, match="native function type"):
            report(modules, targets, ["retail"], project[2])
        assert main(arguments(project, target="retail")) == 2
        assert "native function type" in capsys.readouterr().err
