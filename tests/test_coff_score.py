"""External candidate scoring uses synthetic native COFF and PE fixtures only."""
from copy import deepcopy
import hashlib
import json
import struct

import pytest

from binary_comp.cli import main
from conftest import DATA_VA, write_tiny_pe
from test_coff_cli import project, write_object
from test_coff_contract import DATA_FLAGS, coff_bytes


def score_arguments(project, *, target="retail", function="read_value", modules=("Sample",)):
    config, _, _ = project
    result = ["compare", function, "--config", str(config), "--target", target,
              "--no-build", "--score", "--object", str(config.parent / "candidate.obj")]
    for module in modules:
        result += ["--module", module]
    return result


@pytest.fixture
def candidate(project):
    path = project[0].parent / "candidate.obj"
    write_object(path)
    return path


def test_score_uses_external_object_without_builds_reviews_or_input_mutations(
        project, candidate, capsys, monkeypatch):
    monkeypatch.chdir(project[0].parent)
    # A candidate must not be confused with, or overwrite, an earlier report.
    output = project[0].parent / "binary-comp-report"
    output.mkdir()
    for name in ("report.json", "report.txt"):
        (output / name).write_text("previous report\n")
    (project[0].parent / "retail/Sample.obj").unlink()

    def unexpected(*args, **kwargs):
        pytest.fail("scoring must not build, disassemble or generate reports")

    monkeypatch.setattr("binary_comp.analyzers.function_compare.maybe_build", unexpected)
    monkeypatch.setattr("binary_comp.analyzers.coff_report.compare_windows", unexpected)
    before = {path: path.read_bytes() for path in project[0].parent.rglob("*") if path.is_file()}
    assert main(score_arguments(project)) == 0
    assert capsys.readouterr().out == "0\n"
    assert {path: path.read_bytes() for path in project[0].parent.rglob("*") if path.is_file()} == before


def test_score_accepts_unique_implicit_module_and_symbol_name(project, candidate, capsys):
    assert main(score_arguments(project, function="answer", modules=())) == 0
    assert capsys.readouterr().out == "0\n"


def test_score_counts_changed_instruction_bytes(project, candidate, capsys):
    write_object(candidate, code=bytes.fromhex("a1 00 00 00 00 cb"))
    assert main(score_arguments(project)) == 0
    assert capsys.readouterr().out == "1\n"


def test_score_does_not_ignore_changed_register(project, candidate, capsys):
    original = project[0].parent / "original.exe"
    write_tiny_pe(original, b"\xb8" + struct.pack("<I", DATA_VA + 0x10) + b"\xc3")
    config = json.loads(project[0].read_text())
    config["targets"]["retail"]["original_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    project[0].write_text(json.dumps(config))
    write_object(candidate, code=bytes.fromhex("b9 00 00 00 00 c3"))
    assert main(score_arguments(project)) == 0
    assert capsys.readouterr().out == "1\n"


@pytest.mark.parametrize("code, expected", [
    ("a1 00 00 00 00", 1),
    ("a1 00 00 00 00 c3 c3", 1),
    ("a1 00 00 00 00 c3 31 c0 c3", 3),
    ("a1 00 00 00 00 c3" + " 90" * 11, 11),
])
def test_rejected_extents_get_positive_diagnostic_scores(project, candidate, capsys, code, expected):
    write_object(candidate, code=bytes.fromhex(code))
    assert main(score_arguments(project)) == 0
    output = capsys.readouterr()
    assert output.out == f"{expected}\n"
    assert not output.err


def test_score_keeps_native_bounded_nop_alignment_rule(project, candidate, capsys):
    write_object(candidate, code=bytes.fromhex("a1 00 00 00 00 c3") + b"\x90" * 10)
    assert main(score_arguments(project)) == 0
    assert capsys.readouterr().out == "0\n"


@pytest.mark.parametrize("failure", ["missing", "malformed", "missing function", "wrong role",
                                     "unknown binding", "unsupported fixup", "invalid symbol index",
                                     "crossing fixup", "constant data", "reference hash"])
@pytest.mark.parametrize("extent", ["accepted", "rejected"])
def test_invalid_candidates_never_print_a_score(project, candidate, capsys, failure, extent):
    code = bytes.fromhex("a1 00 00 00 00 c3") + (b"\xc3" if extent == "rejected" else b"")
    write_object(candidate, code=code)
    if failure == "missing":
        candidate.unlink()
    elif failure == "malformed":
        candidate.write_bytes(b"invalid object")
    elif failure == "missing function":
        write_object(candidate, code=code, symbol_name=b"other")
    elif failure == "wrong role":
        write_object(candidate, code=code, symbol_type=0, storage=6)
    elif failure == "unknown binding":
        modules = json.loads(project[1].read_text())
        del modules[0]["bindings"]["retail"]["number"]
        project[1].write_text(json.dumps(modules))
    elif failure in ("unsupported fixup", "invalid symbol index", "crossing fixup"):
        raw = bytearray(candidate.read_bytes())
        reloc_at = struct.unpack_from("<I", raw, 20 + 24)[0]
        offset, index, kind = {"unsupported fixup": (1, 1, 0x99),
                               "invalid symbol index": (1, 99, 6),
                               "crossing fixup": (len(code) - 2, 1, 6)}[failure]
        struct.pack_into("<IIH", raw, reloc_at, offset, index, kind)
        candidate.write_bytes(raw)
    elif failure == "constant data":
        write_object(candidate, code=code, value=8)
    else:
        config = json.loads(project[0].read_text())
        config["targets"]["retail"]["original_sha256"] = "0" * 64
        project[0].write_text(json.dumps(config))
    assert main(score_arguments(project)) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "error:" in output.err
    assert not project[2].exists()


def test_score_validates_unreferenced_configured_data(project, candidate, capsys):
    modules = json.loads(project[1].read_text())
    modules[0]["bindings"]["retail"]["unused"] = {"va": hex(DATA_VA + 0x14), "data_size": 4}
    project[1].write_text(json.dumps(modules))

    def candidate_bytes(value):
        return coff_bytes([
            {"data": bytes.fromhex("a1 00 00 00 00 c3"), "relocations": [(1, 1, 6)]},
            {"name": ".data", "data": struct.pack("<II", 7, value), "flags": DATA_FLAGS},
        ], [{"name": "answer", "section": 1, "type": 0x20},
            {"name": "number", "section": 2}, {"name": "unused", "section": 2, "value": 4}])

    candidate.write_bytes(candidate_bytes(0))
    assert main(score_arguments(project)) == 0
    assert capsys.readouterr().out == "0\n"
    candidate.write_bytes(candidate_bytes(8))
    assert main(score_arguments(project)) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "Rebuilt data differs from original: unused" in output.err


@pytest.mark.parametrize("failure", ["all targets", "multiple modules", "implicit modules",
                                     "duplicate functions", "code label", "unknown module",
                                     "unknown function", "non-coff target", "malformed binding"])
def test_score_requires_an_unambiguous_native_function_selection(project, candidate, capsys, failure):
    args = score_arguments(project)
    modules = json.loads(project[1].read_text())
    if failure == "all targets":
        args = score_arguments(project, target="all")
    elif failure in ("multiple modules", "implicit modules"):
        other = deepcopy(modules[0])
        other["name"] = "Other"
        modules.append(other)
        args = score_arguments(project, modules=("Sample", "Other") if failure == "multiple modules" else ())
    elif failure == "duplicate functions":
        modules[0]["functions"].append(deepcopy(modules[0]["functions"][0]))
    elif failure == "code label":
        modules[0]["code_labels"] = modules[0].pop("functions")
    elif failure == "unknown module":
        args = score_arguments(project, modules=("Unknown",))
    elif failure == "unknown function":
        args = score_arguments(project, function="unknown")
    elif failure == "non-coff target":
        config = json.loads(project[0].read_text())
        config["targets"]["retail"]["kind"] = "pe"
        project[0].write_text(json.dumps(config))
    else:
        modules[0]["bindings"]["retail"]["number"] = {"va": "invalid"}
    project[1].write_text(json.dumps(modules))
    # Selection must fail before trying the object or original reference.
    candidate.unlink()
    (project[0].parent / "original.exe").unlink()
    assert main(args) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "error:" in output.err
    assert "No such file" not in output.err


@pytest.mark.parametrize("invalid", ["build", "no object", "disassembly", "output", "no score"])
def test_score_rejects_incompatible_cli_options(project, candidate, capsys, invalid):
    args = score_arguments(project)
    if invalid == "build":
        args.remove("--no-build")
    elif invalid == "no object":
        index = args.index("--object")
        del args[index:index + 2]
    elif invalid == "disassembly":
        args.insert(2, "disassembly.txt")
    elif invalid == "output":
        args += ["--output", str(project[2])]
    else:
        args.remove("--score")
    assert main(args) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "error:" in output.err
    assert not project[2].exists()
