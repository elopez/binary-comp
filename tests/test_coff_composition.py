"""Composed manifests and build roots, using synthetic COFF and PE fixtures."""
import hashlib
import json
from pathlib import Path
import struct

import pytest

from binary_comp.analyzers.coff_report import load_selection, run_config, score_config
from binary_comp.cli import main
from binary_comp.config import ConfigError, load_project_target
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_common_providers import consumer, pointer_provider, write_project as common_project
from test_coff_contract import coff_bytes
from test_coff_external_bindings import write_project as shared_project


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_project(tmp_path, *, conflicts=()):
    config, old_manifest, modules = shared_project(tmp_path, conflicts=conflicts)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    paths = [manifests / "core.json", manifests / "support.json"]
    for path, module in zip(paths, modules):
        path.write_text(json.dumps([module]))
    second_root = tmp_path / "support-build"
    second_root.mkdir()
    (tmp_path / "retail/Second.obj").rename(second_root / "Second.obj")
    settings = json.loads(config.read_text())
    settings["coff"]["modules"] = ["../manifests/" + path.name for path in paths]
    target = settings["targets"]["retail"]
    del target["object_dir"]
    target["object_dirs"] = ["../retail", "../support-build"]
    target["original_exe"] = "../retail.exe"
    # Resolve every path relative to a configuration outside the build roots.
    directory = tmp_path / "settings"
    directory.mkdir()
    config = directory / "binary-comp.json"
    config.write_text(json.dumps(settings))
    old_manifest.unlink()
    return config, paths, modules


def report_args(config, output, *extra):
    return ["report", "--config", str(config), "--target", "retail", "--no-build",
            "--output", str(output), *extra]


def assert_no_summary(output):
    assert not (output / "report.json").exists()
    assert not (output / "report.txt").exists()


def test_composed_report_resolves_paths_and_hashes_every_manifest_and_object(tmp_path, monkeypatch):
    config, manifests, _ = compose_project(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    result = json.loads((output / "report.json").read_text())
    assert {row["module"] for row in result["code"]} == {"First", "Second"}
    for row in result["code"]:
        assert row["body_size"] == row["decoded_bytes"]["rebuilt"] == 6
        assert row["checked_fixups"] == 1
        assert not row["raw_bytes"]["exact"]
        assert row["relocated_bytes"]["exact"]
    hashes = result["inputs"]["module_manifests_sha256"]
    assert {Path(path).resolve(): digest for path, digest in hashes.items()} == {
        path: sha(path) for path in manifests}
    assert result["inputs"]["modules_sha256"] == hashlib.sha256(
        json.dumps([sha(path) for path in manifests]).encode()).hexdigest()
    assert result["inputs"]["objects_sha256"] == {
        "retail/First": sha(tmp_path / "retail/First.obj"),
        "retail/Second": sha(tmp_path / "support-build/Second.obj")}
    _, target = load_project_target(config, "retail")
    assert target.object_dir is None
    assert tuple(Path(path).resolve() for path in target.object_dirs) == (
        tmp_path / "retail", tmp_path / "support-build")


def test_each_composed_manifest_hash_tracks_its_exact_file_bytes(tmp_path):
    config, manifests, _ = compose_project(tmp_path)
    output = tmp_path / "report"
    assert run_config(config, "retail", output) == 0
    before = json.loads((output / "report.json").read_text())["inputs"]
    manifests[1].write_text(manifests[1].read_text() + "\n")
    assert run_config(config, "retail", output) == 0
    after = json.loads((output / "report.json").read_text())["inputs"]
    assert after["modules_sha256"] != before["modules_sha256"]
    old = list(before["module_manifests_sha256"].values())
    new = list(after["module_manifests_sha256"].values())
    assert new == [old[0], sha(manifests[1])]


def test_cross_manifest_external_conflict_rejects_individually_exact_modules(tmp_path, capsys):
    config, _, _ = compose_project(tmp_path, conflicts=("retail",))
    output = tmp_path / "report"
    for name in ("First", "Second"):
        assert main(report_args(config, output, "--module", name)) == 0
        row, = json.loads((output / "report.json").read_text())["code"]
        assert row["relocated_bytes"]["exact"]
    assert main(report_args(config, output)) == 2
    assert "Conflicting external binding for retail _shared" in capsys.readouterr().err
    assert_no_summary(output)


@pytest.mark.parametrize("filter_kind", ["module", "function", "file"])
@pytest.mark.parametrize("unselected", ["missing", "ambiguous"])
def test_composed_filters_load_only_selected_objects(tmp_path, filter_kind, unselected):
    config, manifests, _ = compose_project(tmp_path, conflicts=("retail",))
    second = tmp_path / "support-build/Second.obj"
    if unselected == "missing":
        second.unlink()
    else:
        (tmp_path / "retail/Second.obj").write_bytes(second.read_bytes())
    options = {"module": {"module_names": ["First"]},
               "function": {"function_name": "_first"}, "file": {"file_filter": "First"}}[filter_kind]
    output = tmp_path / "report"
    assert run_config(config, "retail", output, **options) == 0
    result = json.loads((output / "report.json").read_text())
    assert {row["module"] for row in result["code"]} == {"First"}
    # Provenance retains both configured inputs even when the object scope narrows.
    assert set(result["inputs"]["module_manifests_sha256"].values()) == {sha(path) for path in manifests}


@pytest.mark.parametrize("selection", [["--module", "absent"], ["--filter", "absent"]])
def test_composed_unknown_report_selection_clears_success(tmp_path, capsys, selection):
    config, _, _ = compose_project(tmp_path)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    assert main(report_args(config, output, *selection)) == 2
    assert "error:" in capsys.readouterr().err
    assert_no_summary(output)


def test_composed_unknown_function_clears_success(tmp_path, capsys):
    config, _, _ = compose_project(tmp_path)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    assert main(["compare", "absent", "--config", str(config), "--target", "retail",
                 "--no-build", "--output", str(output)]) == 2
    assert "COFF function not found" in capsys.readouterr().err
    assert_no_summary(output)


@pytest.mark.parametrize("mutation", ["missing", "identical duplicate", "different duplicate", "same root"])
def test_object_lookup_requires_one_match_and_clears_success(tmp_path, capsys, mutation):
    config, _, _ = compose_project(tmp_path)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    second = tmp_path / "support-build/Second.obj"
    if mutation == "missing":
        second.unlink()
        error = "COFF object not found: Second.obj"
    else:
        if mutation == "same root":
            settings = json.loads(config.read_text())
            settings["targets"]["retail"]["object_dirs"].append("../support-build")
            config.write_text(json.dumps(settings))
        else:
            (tmp_path / "retail/Second.obj").write_bytes(
                second.read_bytes() if mutation == "identical duplicate" else b"not a COFF object")
        error = "Ambiguous COFF object: Second.obj"
    assert main(report_args(config, output)) == 2
    assert error in capsys.readouterr().err
    assert_no_summary(output)


def test_duplicate_module_names_across_manifests_fail_even_with_a_filter(tmp_path, capsys):
    config, manifests, modules = compose_project(tmp_path)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    modules[1]["name"] = "First"
    manifests[1].write_text(json.dumps([modules[1]]))
    assert main(report_args(config, output, "--module", "First")) == 2
    assert "Duplicate COFF module: First" in capsys.readouterr().err
    assert_no_summary(output)


@pytest.mark.parametrize("value", [None, "", [], {}, True, [""], [None], ["valid.json", 3]])
def test_malformed_manifest_selection_is_a_configuration_error(tmp_path, capsys, value):
    config, _, _ = compose_project(tmp_path)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    settings = json.loads(config.read_text())
    settings["coff"]["modules"] = value
    config.write_text(json.dumps(settings))
    assert main(report_args(config, output)) == 2
    assert "coff.modules" in capsys.readouterr().err
    assert_no_summary(output)


@pytest.mark.parametrize("mutation", ["missing", "not a list", "malformed JSON"])
def test_each_manifest_is_required_and_validated(tmp_path, capsys, mutation):
    config, manifests, _ = compose_project(tmp_path)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    if mutation == "missing":
        manifests[1].unlink()
    else:
        manifests[1].write_text("{}" if mutation == "not a list" else "{broken")
    assert main(report_args(config, output)) == 2
    assert "error:" in capsys.readouterr().err
    assert_no_summary(output)


@pytest.mark.parametrize("value", [None, "retail", [], {}, True, [""], [None], ["retail", 3]])
def test_malformed_object_roots_are_configuration_errors(tmp_path, capsys, value):
    config, _, _ = compose_project(tmp_path)
    output = tmp_path / "report"
    assert main(report_args(config, output)) == 0
    settings = json.loads(config.read_text())
    settings["targets"]["retail"]["object_dirs"] = value
    config.write_text(json.dumps(settings))
    with pytest.raises(ConfigError, match="object_dirs"):
        load_project_target(config, "retail")
    assert main(report_args(config, output)) == 2
    assert "object_dirs" in capsys.readouterr().err
    assert_no_summary(output)


@pytest.mark.parametrize("legacy", [None, "../retail"])
def test_object_dir_and_object_dirs_are_mutually_exclusive(tmp_path, legacy):
    config, _, _ = compose_project(tmp_path)
    settings = json.loads(config.read_text())
    settings["targets"]["retail"]["object_dir"] = legacy
    config.write_text(json.dumps(settings))
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_selection(config, "retail")


@pytest.mark.parametrize("list_form", [False, True])
def test_one_manifest_preserves_legacy_digest_and_directory_mode(tmp_path, list_form):
    config, manifest, _ = shared_project(tmp_path, conflicts=())
    if list_form:
        settings = json.loads(config.read_text())
        settings["coff"]["modules"] = [manifest.name]
        config.write_text(json.dumps(settings))
    output = tmp_path / "report"
    assert run_config(config, "retail", output) == 0
    result = json.loads((output / "report.json").read_text())
    assert result["inputs"]["modules_sha256"] == sha(manifest)
    assert "module_manifests_sha256" not in result["inputs"]
    _, target = load_project_target(config, "retail")
    assert target.object_dir == str(tmp_path / "retail")
    assert target.object_dirs == ()


def test_score_uses_composed_selection_without_loading_other_objects(tmp_path, capsys):
    config, _, _ = compose_project(tmp_path, conflicts=("retail",))
    candidate = tmp_path / "candidate.obj"
    candidate.write_bytes((tmp_path / "retail/First.obj").read_bytes())
    (tmp_path / "retail/First.obj").unlink()
    (tmp_path / "support-build/Second.obj").unlink()
    assert main(["compare", "_first", "--config", str(config), "--target", "retail",
                 "--module", "First", "--no-build", "--score", "--object", str(candidate)]) == 0
    assert capsys.readouterr().out == "0\n"


def provider_roots(tmp_path):
    config, manifest, module, obj, strong = common_project(tmp_path)
    directory = tmp_path / "provider-build"
    directory.mkdir()
    (tmp_path / "objects/State.obj").rename(directory / "State.obj")
    settings = json.loads(config.read_text())
    del settings["targets"]["sample"]["object_dir"]
    settings["targets"]["sample"]["object_dirs"] = ["objects", "provider-build"]
    config.write_text(json.dumps(settings))
    return config, manifest, module, obj, strong


def test_common_provider_uses_all_roots_and_ignores_candidate_directory(tmp_path):
    config, _, _, obj, strong = provider_roots(tmp_path)
    output = tmp_path / "report"
    assert run_config(config, "sample", output) == 0
    result = json.loads((output / "report.json").read_text())
    assert result["inputs"]["provider_objects_sha256"] == {"sample/State.obj": hashlib.sha256(strong.data).hexdigest()}
    row, = result["data"]
    assert row["exact"] and row["size"] == 4 and row["storage"] == "common_overridden"
    assert result["code"][0]["relocated_bytes"]["exact"]
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = outside / "candidate.obj"
    candidate.write_bytes(obj.data)
    (outside / "State.obj").write_bytes(b"not a provider")
    assert score_config(config, "sample", "_read", candidate) == 0


@pytest.mark.parametrize("mutation", ["missing", "ambiguous"])
def test_provider_lookup_uses_the_same_failure_gate_for_report_and_score(tmp_path, capsys, mutation):
    config, _, _, obj, _ = provider_roots(tmp_path)
    output = tmp_path / "report"
    assert run_config(config, "sample", output) == 0
    capsys.readouterr()
    provider = tmp_path / "provider-build/State.obj"
    if mutation == "missing":
        provider.unlink()
        error = "COFF object not found: State.obj"
    else:
        (tmp_path / "objects/State.obj").write_bytes(provider.read_bytes())
        error = "Ambiguous COFF object: State.obj"
    assert main(["report", "--config", str(config), "--target", "sample", "--no-build",
                 "--output", str(output)]) == 2
    assert_no_summary(output)
    candidate = tmp_path / "candidate.obj"
    candidate.write_bytes(obj.data)
    assert main(["compare", "_read", "--config", str(config), "--target", "sample",
                 "--no-build", "--score", "--object", str(candidate)]) == 2
    streams = capsys.readouterr()
    assert streams.err.count(error) == 2
    assert streams.out == ""


def test_provider_external_dependency_conflict_across_manifests_is_checked(tmp_path):
    config, manifest, module, _, _ = provider_roots(tmp_path)
    module["functions"][0]["sample"]["size"] = 1
    module["bindings"]["sample"]["_target"] = {"va": hex(TEXT_VA + 0x80)}
    obj, _ = consumer(referenced=False)
    (tmp_path / "objects/Sample.obj").write_bytes(obj.data)
    (tmp_path / "provider-build/State.obj").write_bytes(pointer_provider().data)
    (tmp_path / "provider-build/Other.obj").write_bytes(coff_bytes(
        [{"data": b"\xc3"}], [{"name": "_target", "section": 1, "type": 0x20}]))
    other = {"name": "Other", "functions": [{"label": "target", "symbol": "_target",
              "sample": {"va": hex(TEXT_VA + 0x90), "size": 1}}]}
    manifest.write_text(json.dumps([module]))
    second = tmp_path / "other.json"
    second.write_text(json.dumps([other]))
    original = tmp_path / "original.exe"
    code = bytearray(b"\x90" * 0x91)
    code[0] = code[0x80] = code[0x90] = 0xC3
    write_tiny_pe(original, code, {0: struct.pack("<I", TEXT_VA + 0x80)})
    settings = json.loads(config.read_text())
    settings["coff"]["modules"] = [manifest.name, second.name]
    settings["targets"]["sample"]["original_exe"] = original.name
    settings["targets"]["sample"]["original_sha256"] = sha(original)
    config.write_text(json.dumps(settings))
    output = tmp_path / "report"
    for name in ("Sample", "Other"):
        assert run_config(config, "sample", output, module_names=[name]) == 0
    with pytest.raises(ValueError, match=r"_target: Sample provider State.obj=.*Other="):
        run_config(config, "sample", output)
    assert_no_summary(output)
