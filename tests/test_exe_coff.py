"""Linked PE diagnostics use explicit contracts without a compiler or C++ parser."""

import hashlib
import json
import struct

import pytest

from binary_comp.analyzers import exe
from binary_comp.analyzers.coff_report import load_selection
from binary_comp.cli import main
from conftest import TEXT_VA, write_tiny_pe
from test_coff_contract import coff_bytes


CODE = bytes.fromhex("b8 11 22 33 44 c3")
INLINE = "?read@InlineValue@@QBEHXZ"
PUBLIC_HEADER = "  Address         Publics by Value              Rva+Base       Lib:Object\n\n"


def map_line(symbol="_answer", address=TEXT_VA, obj="Sample.obj", flag="f", segment=1):
    return f" {segment:04x}:00000000 {symbol} {address:08x} {flag} {obj}\n"


def function(symbol="_answer", address=TEXT_VA, size=len(CODE)):
    return {"label": symbol, "symbol": symbol, "retail": {"va": hex(address), "size": size}}


@pytest.fixture
def project(tmp_path):
    original = tmp_path / "original.exe"
    rebuilt = tmp_path / "rebuilt.exe"
    write_tiny_pe(original, CODE)
    write_tiny_pe(rebuilt, CODE)
    linked_map = tmp_path / "rebuilt.map"
    linked_map.write_text(map_line())
    manifest = tmp_path / "modules.json"
    manifest.write_text(json.dumps([{"name": "Sample", "functions": [function()]}]))
    sources = tmp_path / "src"
    sources.mkdir()
    (sources / "sample.cpp").write_text("/* 0x00401000 */ int answer() { return 1; }\n")
    config = tmp_path / "binary-comp.json"
    config.write_text(json.dumps({
        "targets": {"retail": {
            "kind": "i386-coff", "original_exe": "original.exe",
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "rebuilt_exe": "rebuilt.exe", "map": "rebuilt.map",
            "object_dir": "not-built", "source_dirs": "src",
        }},
        "coff": {"modules": "modules.json", "source_root": "src"},
    }))
    return {"config": config, "manifest": manifest, "original": original,
            "rebuilt": rebuilt, "map": linked_map}


def write_json(path, data):
    path.write_text(json.dumps(data))


def refresh_hash(project):
    config = json.loads(project["config"].read_text())
    config["targets"]["retail"]["original_sha256"] = hashlib.sha256(
        project["original"].read_bytes()).hexdigest()
    write_json(project["config"], config)


def compare(project, **options):
    modules, targets, _ = load_selection(project["config"], "retail")
    return exe.compare_executable(targets["retail"], exe.ExeCompareOptions(
        include_functions=True, **options), coff_modules=modules)


def run(project, *, functions=True):
    args = ["exe", "--config", str(project["config"]), "--target", "retail"]
    return main(args + (["--functions"] if functions else []))


def native_object(project, symbols=None, *, code=CODE, flags=0x60000020):
    directory = project["config"].parent / "not-built"
    directory.mkdir(exist_ok=True)
    path = directory / "Sample.obj"
    if symbols is None:
        symbols = [{"name": INLINE, "section": 1, "type": 0x20}]
    path.write_bytes(coff_bytes([{"data": code, "flags": flags}], symbols))
    return path


@pytest.fixture
def external_inline(project):
    native_object(project)
    write_json(project["manifest"], [{"name": "Sample", "functions": [function(INLINE)]}])
    project["map"].write_text(PUBLIC_HEADER + map_line(INLINE, obj="Other.obj"))
    return project


def test_complete_raw_windows_need_no_source_parser_or_built_objects(project, monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise AssertionError("C++ parser must not be used")

    monkeypatch.setattr(exe, "load_source_groups", unavailable)
    monkeypatch.setattr(exe, "map_source_groups", unavailable)
    monkeypatch.setattr("binary_comp.source.cpp.make_cpp_parser", unavailable)
    monkeypatch.setattr("binary_comp.cli.maybe_build", unavailable)
    assert not (project["config"].parent / "not-built").exists()
    result = compare(project)
    assert result.functions is None
    row, = result.linked_functions.rows
    assert row.original_addr == row.rebuilt_addr == TEXT_VA
    assert row.raw_equal is True
    assert row.configured_size == row.compared_size == row.matching_bytes == len(CODE)
    assert run(project) == 0
    text = capsys.readouterr().out
    assert "Configured linked RAW windows" in text
    assert "Raw-equal configured windows: 1/1" in text
    assert "does not verify source/COFF equality, linked function extents" in text
    assert "Function Address Mapping" not in text


def test_raw_operand_changes_are_not_relocation_corrected(project):
    # Both instructions still decode as MOV EAX, immediate, but the full operand differs.
    write_tiny_pe(project["rebuilt"], bytes.fromhex("b8 11 22 33 55 c3"))
    row, = compare(project).linked_functions.rows
    assert row.raw_equal is False
    assert row.matching_bytes == len(CODE) - 1
    assert row.compared_size == len(CODE)


@pytest.mark.parametrize("changed", [False, True])
def test_linked_address_comes_from_map_identity_not_original_location(project, changed):
    raw = bytearray(project["rebuilt"].read_bytes())
    raw[0x200:0x200 + len(CODE)] = b"\xcc" * len(CODE)
    linked = CODE[:-1] + (b"\xcc" if changed else CODE[-1:])
    raw[0x220:0x220 + len(linked)] = linked
    project["rebuilt"].write_bytes(raw)
    project["map"].write_text(map_line(address=TEXT_VA + 0x20))
    row, = compare(project).linked_functions.rows
    assert row.rebuilt_addr == TEXT_VA + 0x20
    assert row.raw_equal is (not changed)


def test_extent_is_not_capped_or_shortened_at_next_map_symbol(project):
    size = 0x10001
    for key in ("original", "rebuilt"):
        write_tiny_pe(project[key], b"\x90" * size)
        raw = bytearray(project[key].read_bytes())
        # Expand the synthetic PE's text allocation and move its data section.
        struct.pack_into("<IIII", raw, 0x178 + 8, size, 0x1000, size, 0x200)
        struct.pack_into("<IIII", raw, 0x1a0 + 8, 0x20, 0x20000, 0x20, 0x200 + size)
        if key == "rebuilt":
            raw[0x200 + size - 1] = 0xcc
        project[key].write_bytes(raw)
    refresh_hash(project)
    write_json(project["manifest"], [{"name": "Sample", "functions": [function(size=size)]}])
    project["map"].write_text(map_line() + map_line("_neighbor", TEXT_VA + 1, "Other.obj"))
    row, = compare(project, max_function_bytes=1).linked_functions.rows
    assert row.configured_size == row.compared_size == size
    assert row.matching_bytes == size - 1
    assert row.raw_equal is False


@pytest.mark.parametrize("image", ["original", "rebuilt"])
@pytest.mark.parametrize("failure", ["zero-fill", "non-executable"])
def test_complete_executable_file_backing_is_required(project, image, failure, capsys):
    raw = bytearray(project[image].read_bytes())
    if failure == "zero-fill":
        struct.pack_into("<I", raw, 0x178 + 16, 2)
    else:
        struct.pack_into("<I", raw, 0x178 + 36, 0x40000040)
    project[image].write_bytes(raw)
    refresh_hash(project)
    if image == "original":
        assert run(project) == 2
        assert "error:" in capsys.readouterr().err
    else:
        row, = compare(project).linked_functions.rows
        assert row.status == "invalid-window"
        assert row.raw_equal is None
        assert row.matching_bytes is None
        assert row.compared_size == 0


def test_overlapping_non_executable_section_cannot_supply_executable_window_bytes(project):
    raw = bytearray(project["rebuilt"].read_bytes())
    struct.pack_into("<I", raw, 0x178 + 16, 2)  # Executable text has only two file bytes.
    struct.pack_into("<I", raw, 0x1a0 + 12, 0x1000)  # Non-executable data overlaps its VA.
    raw[0x400:0x400 + len(CODE)] = CODE
    project["rebuilt"].write_bytes(raw)
    row, = compare(project).linked_functions.rows
    assert row.status == "invalid-window"
    assert row.raw_equal is None
    assert row.compared_size == 0


@pytest.mark.parametrize("obj", [r"C:\Build\SAMPLE.OBJ", r"support.lib:dir\Sample.obj",
                                 "out/Sample.obj", r"C:\Build Files\Sample.obj",
                                 r'"C:\Build Files\Sample.obj"'])
def test_exact_names_are_scoped_by_case_insensitive_object_member(project, obj):
    project["map"].write_text(map_line(obj=obj, segment=3) + map_line(address=TEXT_VA + 16, obj="Other.obj"))
    row, = compare(project).linked_functions.rows
    assert row.rebuilt_addr == TEXT_VA
    assert row.raw_equal is True


@pytest.mark.parametrize("entries,status", [
    (map_line("answer"), "missing"),
    (map_line(obj="Other.obj"), "missing"),
    (map_line(obj=""), "missing"),
    (map_line(flag=""), "not-function"),
    (map_line() + map_line(address=TEXT_VA + 16), "ambiguous"),
    (map_line(obj="one/Sample.obj") + map_line(obj="two/Sample.obj"), "ambiguous"),
    (map_line(obj=r"C:\One Build\Sample.obj") + map_line(obj=r"C:\Two Build\Sample.obj"), "ambiguous"),
    (map_line() + map_line(flag=""), "ambiguous"),
])
def test_missing_untyped_and_ambiguous_map_identities_cannot_match(project, entries, status):
    project["map"].write_text(entries)
    row, = compare(project).linked_functions.rows
    assert row.status == status
    assert row.raw_equal is None
    assert row.compared_size == 0


def test_repeated_map_records_are_not_distinct_identities(project):
    project["map"].write_text(map_line() * 2)
    row, = compare(project).linked_functions.rows
    assert row.raw_equal is True


def test_incremental_map_flag_does_not_hide_object_identity(project):
    project["map"].write_text(map_line(flag="f i"))
    row, = compare(project).linked_functions.rows
    assert row.raw_equal is True


def test_collapsed_linked_functions_are_compared_as_separate_contracts(project):
    write_tiny_pe(project["original"], b"\xc3")
    write_tiny_pe(project["rebuilt"], b"\xc3")
    refresh_hash(project)
    write_json(project["manifest"], [
        {"name": "Sample", "functions": [function(size=1)]},
        {"name": "Other", "functions": [function("_other", TEXT_VA + 16, 1)]},
    ])
    project["map"].write_text(map_line() + map_line("_other", obj="Other.obj"))
    rows = compare(project).linked_functions.rows
    assert len(rows) == 2
    assert all(row.raw_equal is True and row.rebuilt_addr == TEXT_VA for row in rows)


def test_only_explicit_functions_are_counted_and_alias_bindings_do_not_resolve_names(project):
    write_json(project["manifest"], [{
        "name": "Sample", "functions": [function()],
        "code_labels": [function("_label", TEXT_VA + 16, 1)],
        "code_regions": [{"label": "region", "symbol": "_region",
                          "entries": [{"symbol": "_region", "offset": 0}],
                          "retail": {"va": hex(TEXT_VA + 32), "size": 4}}],
        "bindings": {"retail": {"_alias": {"va": hex(TEXT_VA)}}},
    }])
    project["map"].write_text(map_line("_alias") + map_line("_label", TEXT_VA + 16)
                              + map_line("_region", TEXT_VA + 32))
    row, = compare(project).linked_functions.rows
    assert row.symbol == "_answer"
    assert row.status == "missing"


@pytest.mark.parametrize("key", ["original", "rebuilt", "map", "manifest"])
def test_missing_inputs_fail_the_cli(project, key, capsys):
    project[key].unlink()
    assert run(project) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["rebuilt_exe", "map", "original_sha256"])
def test_missing_required_configuration_fails_the_cli(project, key, capsys):
    config = json.loads(project["config"].read_text())
    del config["targets"]["retail"][key]
    write_json(project["config"], config)
    assert run(project) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("functions", [False, True])
def test_original_hash_is_authenticated_for_coff_executable_comparison(project, functions, capsys):
    assert run(project, functions=functions) == 0
    raw = bytearray(project["original"].read_bytes())
    raw[0x200] ^= 1
    project["original"].write_bytes(raw)
    assert run(project, functions=functions) == 2
    assert "Original SHA-256 mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("image", ["original", "rebuilt"])
def test_truncated_raw_sections_fail_the_cli(project, image, capsys):
    project[image].write_bytes(project[image].read_bytes()[:0x300])
    refresh_hash(project)
    assert run(project) == 2
    assert "Truncated PE section" in capsys.readouterr().err


@pytest.mark.parametrize("extent", [{"va": hex(TEXT_VA), "size": 0},
                                     {"va": "0xffffffff", "size": 2},
                                     {"va": hex(TEXT_VA + 0x1ff), "size": 2}])
def test_invalid_or_incomplete_original_extents_fail_the_cli(project, extent, capsys):
    entry = function()
    entry["retail"] = extent
    write_json(project["manifest"], [{"name": "Sample", "functions": [entry]}])
    assert run(project) == 2
    assert "error:" in capsys.readouterr().err


def test_public_selected_definition_can_come_from_another_object(external_inline):
    row, = compare(external_inline).linked_functions.rows
    assert row.raw_equal is True
    assert row.rebuilt_addr == TEXT_VA
    assert "exact external function defined in sample.obj" in row.detail
    assert "selected Other.obj" in row.detail


def test_owned_identity_takes_precedence_without_opening_object(external_inline, monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("Owned MAP resolution must not open an object")

    external_inline["map"].write_text(PUBLIC_HEADER + map_line(INLINE, obj="Sample.obj")
                                      + map_line(INLINE, TEXT_VA + 16, "Other.obj"))
    monkeypatch.setattr("binary_comp.analyzers.coff_report.load_object", unavailable)
    row, = compare(external_inline).linked_functions.rows
    assert row.raw_equal is True
    assert row.rebuilt_addr == TEXT_VA
    assert row.detail == ""


@pytest.mark.parametrize("record", [
    {"storage": 3},  # A private function cannot borrow a same-named public function.
    {"section": 0},
    {"type": 0},
    {"storage": 6},
    {"value": len(CODE)},
    {"name": "?other@InlineValue@@QBEHXZ"},
])
def test_public_fallback_needs_exact_defined_external_function(external_inline, record):
    symbol = {"name": INLINE, "section": 1, "type": 0x20, **record}
    native_object(external_inline, [symbol])
    row, = compare(external_inline).linked_functions.rows
    assert row.status == "missing"
    assert row.raw_equal is None
    assert row.rebuilt_addr is None
    assert "external proof failed" in row.detail


def test_weak_fallback_to_an_external_function_does_not_prove_requested_name(external_inline):
    native_object(external_inline, [
        {"name": INLINE, "storage": 105, "type": 0x20,
         "aux": struct.pack("<II10x", 2, 3)},
        {"name": "?other@InlineValue@@QBEHXZ", "section": 1, "type": 0x20},
    ])
    row, = compare(external_inline).linked_functions.rows
    assert row.status == "missing"
    assert row.raw_equal is None
    assert "Undefined symbol" in row.detail


@pytest.mark.parametrize("failure", ["missing", "malformed", "non-executable"])
def test_unavailable_native_linkage_proof_stays_diagnostic(external_inline, failure, capsys):
    path = external_inline["config"].parent / "not-built" / "Sample.obj"
    if failure == "missing":
        path.unlink()
    elif failure == "malformed":
        path.write_bytes(b"invalid")
    else:
        native_object(external_inline, flags=0x40000040)
    row, = compare(external_inline).linked_functions.rows
    assert row.status == "missing"
    assert row.raw_equal is None
    assert "external proof failed" in row.detail
    assert run(external_inline) == 0
    assert "Raw-equal configured windows: 0/1" in capsys.readouterr().out


@pytest.mark.parametrize("prefix", [
    "", "Static symbols\n", "Address Unknown symbols Rva+Base Lib:Object\n",
    PUBLIC_HEADER + map_line("_unrelated", obj="Other.obj") + "Static symbols\n",
    PUBLIC_HEADER + map_line("_unrelated", obj="Other.obj") + "Address Unknown symbols\n",
    "Address Static symbols Rva+Base Lib:Object\n",
])
def test_unknown_or_static_map_rows_cannot_supply_public_fallback(external_inline, prefix, monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("No PUBLIC candidate should require native proof")

    external_inline["map"].write_text(prefix + map_line(INLINE, obj="Other.obj"))
    monkeypatch.setattr("binary_comp.analyzers.coff_report.load_object", unavailable)
    row, = compare(external_inline).linked_functions.rows
    assert row.status == "missing"
    assert row.raw_equal is None


@pytest.mark.parametrize("second_address", [TEXT_VA, TEXT_VA + 16])
def test_multiple_public_identities_remain_ambiguous(external_inline, second_address):
    external_inline["map"].write_text(PUBLIC_HEADER + map_line(INLINE, obj="Other.obj")
                                      + map_line(INLINE, second_address, "Third.obj"))
    row, = compare(external_inline).linked_functions.rows
    assert row.status == "ambiguous"
    assert row.raw_equal is None
    assert row.rebuilt_addr is None


def test_public_linkage_proof_does_not_establish_coff_body_equality(external_inline):
    native_object(external_inline, code=b"\xc3")
    row, = compare(external_inline).linked_functions.rows
    assert row.raw_equal is True
    assert row.compared_size == len(CODE)


def test_public_fallback_still_compares_exact_raw_operands(external_inline):
    write_tiny_pe(external_inline["rebuilt"], bytes.fromhex("b8 11 22 33 55 c3"))
    row, = compare(external_inline).linked_functions.rows
    assert row.raw_equal is False
    assert row.matching_bytes == len(CODE) - 1
