"""Synthetic partial-instruction contracts never establish complete ownership."""
import hashlib
import json
import struct
from copy import deepcopy

import pytest

from binary_comp.analyzers.coff_report import load_selection, score_config
from binary_comp.cli import main
from binary_comp.config import ConfigError
from binary_comp.core.coff import COFF, compare_code_fragment
from conftest import DATA_VA, TEXT_VA, write_tiny_pe
from test_coff_contract import coff_bytes, reference

pytest.importorskip("capstone")


def fragment(va=TEXT_VA + 2, offset=0, size=6):
    return {"label": "partial read", "symbol": "reader", "sample": {
        "va": hex(va), "owner_va": hex(TEXT_VA), "object_offset": offset, "size": size}}


@pytest.fixture
def sample(tmp_path):
    raw = b"\xa1" + bytes(4) + b"\xc3\x90"
    coff = COFF.from_bytes(coff_bytes([
        {"data": raw, "relocations": [(1, 1, 6)]},
    ], [{"name": "reader", "section": 1, "type": 0x20}, {"name": "outside"}]))
    original = b"\x90\x90\xa1" + struct.pack("<I", DATA_VA) + b"\xc3\x90"
    return coff, fragment(), {"outside": {"va": hex(DATA_VA)}}, reference(tmp_path, original)


def compare(sample):
    return compare_code_fragment(sample[0], sample[1], "sample", sample[2], sample[3])


def test_independent_owner_and_object_offsets_preserve_fixup_and_raw_bytes(sample):
    exact, actual, expected, count, raw = compare(sample)
    assert exact and actual == expected and count == 1 and raw != actual
    # The original owner is independently declared, and no owner binding is inferred.
    assert "reader" not in sample[2]


@pytest.mark.parametrize("change", ["actual_start", "actual_end", "original_start", "original_end",
                                     "before_owner", "outside_window", "crossing_relocation", "unsupported"])
def test_invalid_instruction_windows_and_fixups_fail_closed(sample, change):
    coff, entry, bindings, image = sample
    target = entry["sample"]
    if change == "actual_start":
        target.update(object_offset=1, size=5)
    elif change == "actual_end":
        target["size"] = 4
    elif change == "original_start":
        target.update(va=hex(TEXT_VA + 3), size=5)
    elif change == "original_end":
        target["size"] = 4
    elif change == "before_owner":
        target["owner_va"] = hex(TEXT_VA + 3)
    elif change == "outside_window":
        target["size"] = 8
    elif change == "crossing_relocation":
        # A boundary is an instruction boundary but still splits a malformed relocation.
        coff.sections[0]["relocations"] = [(0, 1, 6)]
        target.update(va=hex(TEXT_VA + 7), object_offset=5, size=1)
        coff.sections[0]["relocations"] = [(3, 1, 6)]
    else:
        coff.sections[0]["relocations"] = [(1, 1, 7)]
    with pytest.raises(ValueError):
        compare(sample)


def test_explicit_owner_binding_is_checked_against_declared_origin(sample):
    sample[2]["reader"] = {"va": hex(TEXT_VA + 1)}
    with pytest.raises(ValueError, match="binding"):
        compare(sample)


def test_executable_and_typed_owner_required(sample):
    for field, value in (("type", 0), ("storage", 6)):
        trial = deepcopy(sample)
        trial[0].named["reader"][field] = value
        with pytest.raises(ValueError, match="typed executable"):
            compare(trial)
    trial = deepcopy(sample)
    trial[0].sections[0]["flags"] &= ~0x20000000
    with pytest.raises(ValueError, match="typed executable"):
        compare(trial)



def test_native_section_definition_is_metadata_not_data(tmp_path):
    code = b"\x90\xc3"
    auxiliary = struct.pack("<IHHIHBBH", len(code), 0, 0, 0, 0, 0, 0, 0)
    coff = COFF.from_bytes(coff_bytes([{"data": code}], [
        {"name": ".text", "section": 1, "storage": 3, "aux": auxiliary},
        {"name": "reader", "section": 1, "type": 0x20},
    ]))
    assert compare_code_fragment(coff, fragment(TEXT_VA, 0, 2), "sample", {},
                                 reference(tmp_path, code))[0]


def test_native_data_in_owner_is_not_a_code_fragment(sample):
    coff, _, _, _ = sample
    coff.symbols[99] = {"name": "inline", "section": 1, "storage": 3,
                        "type": 0, "value": 5, "aux_count": 0}
    with pytest.raises(ValueError, match="native data"):
        compare(sample)

def test_unbacked_or_non_executable_original_fails(sample):
    trial = deepcopy(sample)
    trial[1]["sample"].update(va=hex(DATA_VA), owner_va=hex(DATA_VA))
    with pytest.raises(ValueError, match="file-backed executable"):
        compare(trial)


@pytest.fixture
def project(tmp_path):
    original = tmp_path / "original.exe"
    write_tiny_pe(original, b"\x90\x90\xa1" + struct.pack("<I", DATA_VA) + b"\xc3\x90")
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "Sample.obj").write_bytes(coff_bytes([
        {"data": b"\xa1" + bytes(4) + b"\xc3\x90", "relocations": [(1, 1, 6)]},
    ], [{"name": "reader", "section": 1, "type": 0x20}, {"name": "outside"}]))
    config = tmp_path / "config.json"
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps([{"name": "Sample", "code_fragments": [fragment()],
                                     "bindings": {"sample": {"outside": {"va": hex(DATA_VA)}}}}]))
    config.write_text(json.dumps({"targets": {"sample": {"kind": "i386-coff", "original_exe": "original.exe",
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "object_dir": "objects"}},
        "coff": {"modules": "modules.json"}}))
    return config, manifest, tmp_path / "report", objects / "Sample.obj"


def args(project, command="report"):
    return [command, "--config", str(project[0]), "--target", "sample", "--no-build", "--output", str(project[2])]


def test_report_and_filtered_compare_keep_partial_totals_separate(project, capsys):
    assert main(args(project)) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert result["code"] == [] and len(result["fragments"]) == 1
    row = result["fragments"][0]
    assert row["partial"] and row["relocated_bytes"]["exact"] and not row["raw_bytes"]["exact"]
    assert "outside this window are unverified" in (project[2] / row["review"]).read_text()
    assert "0/1 raw-byte equal; 1/1 relocated-byte equal" in capsys.readouterr().out
    assert main(args(project, "compare") + ["partial read"]) == 0
    assert "Partial instruction fragment" in capsys.readouterr().out
    with pytest.raises(ConfigError, match="exactly one native function"):
        score_config(project[0], "sample", "partial read", project[3], module_names=["Sample"])


@pytest.mark.parametrize("key,value", [("size", True), ("size", 0), ("object_offset", True),
                                       ("object_offset", -1), ("owner_va", True),
                                       ("owner_va", "-1"), ("va", "0x100000000")])
def test_manifest_rejects_bad_domains(project, key, value):
    modules = json.loads(project[1].read_text())
    modules[0]["code_fragments"][0]["sample"][key] = value
    project[1].write_text(json.dumps(modules))
    with pytest.raises((ValueError, ConfigError)):
        load_selection(project[0], "sample")


def test_original_and_native_overlap_are_rejected_even_when_filtered(project):
    modules = json.loads(project[1].read_text())
    other = deepcopy(modules[0]["code_fragments"][0])
    other["label"] = "other"
    # Original ranges disjoint but native ranges overlap.
    other["sample"].update(va=hex(TEXT_VA + 8), size=1, object_offset=5)
    modules[0]["code_fragments"].append(other)
    project[1].write_text(json.dumps(modules))
    assert main(args(project, "compare") + ["partial read"]) == 2
    other["sample"]["va"] = hex(TEXT_VA + 7)
    project[1].write_text(json.dumps(modules))
    with pytest.raises(ValueError, match="overlaps"):
        load_selection(project[0], "sample", function_name="partial read")


def test_disjoint_fragments_share_one_owner_without_counting_a_function(project):
    modules = json.loads(project[1].read_text())
    modules[0]["code_fragments"][0]["sample"]["size"] = 5
    modules[0]["code_fragments"].append({**fragment(TEXT_VA + 7, 5, 1), "label": "return"})
    project[1].write_text(json.dumps(modules))
    assert main(args(project)) == 0
    result = json.loads((project[2] / "report.json").read_text())
    assert not result["code"] and len(result["fragments"]) == 2
    assert all(row["relocated_bytes"]["exact"] for row in result["fragments"])


def test_fragment_owner_identity_conflict_across_selected_modules_fails(project):
    modules = json.loads(project[1].read_text())
    (project[3].parent / "Other.obj").write_bytes(project[3].read_bytes())
    modules.append({"name": "Other", "bindings": {"sample": {"reader": {"va": hex(TEXT_VA + 1)}}}})
    project[1].write_text(json.dumps(modules))
    assert main(args(project)) == 2


def test_original_disjoint_complete_owner_cannot_reuse_native_function(project):
    modules = json.loads(project[1].read_text())
    modules[0]["functions"] = [{"label": "complete", "symbol": "reader",
                                 "sample": {"va": hex(TEXT_VA + 32), "size": 1}}]
    # Explicit owner identity must remain consistent with the complete owner.
    modules[0]["code_fragments"][0]["sample"]["owner_va"] = hex(TEXT_VA + 32)
    modules[0]["code_fragments"][0]["sample"]["va"] = hex(TEXT_VA + 34)
    project[1].write_text(json.dumps(modules))
    assert main(args(project, "compare") + ["partial read"]) == 2


def test_direct_fragment_rejects_complete_section_layout_overlap(tmp_path):
    from test_coff_section_layout import make_case

    coff, _, bindings, image = make_case(tmp_path)
    for binding in bindings.values():
        binding["va"] = hex(int(binding["va"], 16) - DATA_VA + TEXT_VA)
    entry = {"label": "read", "symbol": "caller", "sample": {
        "va": hex(TEXT_VA), "owner_va": hex(TEXT_VA), "object_offset": 0, "size": 6}}
    with pytest.raises(ValueError, match="Section layout original range"):
        compare_code_fragment(coff, entry, "sample", bindings, image)


def test_direct_fragment_rejects_complete_executable_data_overlap(tmp_path):
    from test_coff_executable_data import make_case

    coff, _, bindings, image = make_case(tmp_path)
    bindings["table"]["va"] = hex(TEXT_VA)
    entry = {"label": "read", "symbol": "caller", "sample": {
        "va": hex(TEXT_VA), "owner_va": hex(TEXT_VA), "object_offset": 0, "size": 6}}
    with pytest.raises(ValueError, match="overlaps complete executable data"):
        compare_code_fragment(coff, entry, "sample", bindings, image)


@pytest.mark.parametrize("bad_read", [None, b"\x90"])
def test_direct_fragment_rejects_short_or_missing_reference_prefix(sample, bad_read):
    class ShortReader:
        def section_for_va(self, va):
            return sample[3].section_for_va(va)

        def read(self, va, size):
            return bad_read

    with pytest.raises(ValueError, match="not completely file-backed"):
        compare_code_fragment(sample[0], sample[1], "sample", sample[2], ShortReader())
