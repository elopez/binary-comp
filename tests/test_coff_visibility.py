"""Storage-only function visibility edits on generic synthetic native objects."""
from copy import deepcopy
import hashlib
import json
import struct

import pytest

from binary_comp.cli import main
from binary_comp.core.coff import COFF
from binary_comp.core.coff_extension import compare_coff_extension
from binary_comp.core.coff_visibility import globalize_coff_data, globalize_coff_functions
from test_coff_contract import coff_bytes
from test_coff_extension import section_aux


FUNCTIONS = ["_static_one", "_static_two"]


def fixture():
    sections = [
        {"name": ".text", "data": bytes(4) + b"\xc3", "flags": 0x60501020,
         "relocations": [(0, 10, 20)]},
        {"name": ".text", "data": b"\x90\xc3", "flags": 0x60101020},
        {"name": ".data", "data": b"xyz", "flags": 0xC0000040},
    ]
    symbols = [
        {"name": ".text", "storage": 3, "section": 1, "aux": section_aux(5, 1, selection=1)},
        {"name": FUNCTIONS[0], "storage": 3, "type": 0x24, "section": 1},
        {"name": "$local", "storage": 6, "value": 4, "section": 1},
        {"name": ".text", "storage": 3, "section": 2, "aux": section_aux(2, selection=1)},
        {"name": FUNCTIONS[1], "storage": 3, "type": 0x20, "section": 2},
        {"name": ".data", "storage": 3, "section": 3, "aux": section_aux(3)},
        {"name": "_data", "section": 3},
        {"name": "_external"},
    ]
    return sections, symbols


def pack(parts=None):
    return COFF.from_bytes(coff_bytes(*(parts or fixture())))


def test_two_functions_change_only_their_declared_primary_storage_bytes():
    source = pack()
    original_data, original_symbols = source.data, deepcopy(source.symbols)
    output, witness = globalize_coff_functions(source, reversed(FUNCTIONS))
    result = COFF.from_bytes(output)
    offsets = {row["storage_offset"] for row in witness["functions"]}
    assert len(output) == len(original_data)
    assert {i for i, pair in enumerate(zip(original_data, output)) if pair[0] != pair[1]} == offsets
    assert len(offsets) == witness["changed_bytes"] == 2
    assert all(original_data[i] == 3 and output[i] == 2 for i in offsets)
    assert result.sections == source.sections
    assert result.auxiliary_records == source.auxiliary_records
    assert result.sections[2]["data"] == b"xyz"  # Ordinary data is not padded.
    assert source.data == original_data and source.symbols == original_symbols
    assert witness["raw_sections_unchanged"] and witness["relocations_unchanged"]
    assert witness["input_sha256"] == hashlib.sha256(original_data).hexdigest()
    assert witness["output_sha256"] == hashlib.sha256(output).hexdigest()
    for row in witness["functions"]:
        assert row["storage_offset"] == struct.unpack_from("<I", original_data, 8)[0] + row["symbol_index"] * 18 + 16
        assert (row["old_storage"], row["new_storage"], row["value"]) == (3, 2, 0)
        assert row["type"] in (0x20, 0x24)
        assert result.named[row["name"]]["storage"] == 2


def test_unselected_static_function_stays_static():
    source = pack()
    output, witness = globalize_coff_functions(source, FUNCTIONS[:1])
    result = COFF.from_bytes(output)
    assert witness["changed_bytes"] == 1
    assert result.named[FUNCTIONS[1]]["storage"] == 3


@pytest.mark.parametrize("names", [[], [""], ["missing"], [FUNCTIONS[0], FUNCTIONS[0]], [None], "_static_one"])
def test_names_must_be_explicit_distinct_nonempty_and_present(names):
    with pytest.raises(ValueError):
        globalize_coff_functions(pack(), names)


@pytest.mark.parametrize("change", ["global", "data", "interior", "other_storage", "auxiliary", "common", "weak", "ambiguous"])
def test_nonstatic_or_ambiguous_function_shapes_are_rejected(change):
    parts = fixture()
    function = parts[1][1]
    name = FUNCTIONS[0]
    if change == "global":
        function["storage"] = 2
    elif change == "data":
        name = "_data"
    elif change == "interior":
        function["value"] = 1
    elif change == "other_storage":
        function["storage"] = 6
    elif change == "auxiliary":
        function["aux"] = bytes(18)
    elif change == "common":
        name = "_common"
        parts[1].append({"name": name, "value": 8})
    elif change == "weak":
        name = "_weak"
        parts[1].append({"name": name, "type": 0x20, "storage": 105,
                         "aux": struct.pack("<II10x", 10, 2)})
    else:
        parts[1].append({"name": name, "type": 0x20})
    with pytest.raises(ValueError):
        globalize_coff_functions(pack(parts), [name])


@pytest.mark.parametrize("change", ["untyped", "not_comdat", "not_code", "not_executable", "data_flags", "not_leader", "second_function"])
def test_only_a_sole_typed_executable_code_comdat_leader_can_be_exposed(change):
    parts = fixture()
    if change == "untyped":
        parts[1][1]["type"] = 0
    elif change == "not_comdat":
        parts[0][0]["flags"] &= ~0x1000
        parts[1][0]["aux"] = section_aux(5, 1)
    elif change == "not_code":
        parts[0][0]["flags"] &= ~0x20
    elif change == "not_executable":
        parts[0][0]["flags"] &= ~0x20000000
    elif change == "data_flags":
        parts[0][0]["flags"] |= 0x40
    elif change == "not_leader":
        parts[1][1], parts[1][2] = parts[1][2], parts[1][1]
    else:
        parts[1].append({"name": "_second_entry", "type": 0x20, "storage": 3, "section": 1})
    with pytest.raises(ValueError):
        globalize_coff_functions(pack(parts), FUNCTIONS[:1])


def test_raw_section_alias_of_selected_metadata_is_rejected_even_when_native_views_parse():
    source = pack()
    index = next(i for i, s in source.symbols.items() if s["name"] == FUNCTIONS[0])
    offset = struct.unpack_from("<I", source.data, 8)[0] + index * 18 + 16
    raw = bytearray(source.data)
    struct.pack_into("<I", raw, 20 + 20, offset)
    aliased = COFF.from_bytes(raw)
    assert compare_coff_extension(aliased, aliased)["equivalent"]
    with pytest.raises(ValueError, match="Symbol table overlaps"):
        globalize_coff_functions(aliased, FUNCTIONS[:1])


def test_preflight_does_not_ignore_unsupported_section_metadata():
    parts = fixture()
    aux = bytearray(parts[1][0]["aux"])
    aux[15] = 1
    parts[1][0]["aux"] = bytes(aux)
    with pytest.raises(ValueError, match="section-definition"):
        globalize_coff_functions(pack(parts), FUNCTIONS[:1])


def test_cli_writes_a_separate_object_and_explicit_metadata_witness(tmp_path, capsys):
    source, output = tmp_path / "input.obj", tmp_path / "output.obj"
    original = pack().data
    source.write_bytes(original)
    assert main(["coff-globalize", str(source), str(output), "--function", FUNCTIONS[0],
                 "--function", FUNCTIONS[1]]) == 0
    witness = json.loads(capsys.readouterr().out)
    assert witness["operation"] == "coff-function-visibility"
    assert witness["changed_bytes"] == 2
    assert source.read_bytes() == original
    assert COFF(output).named[FUNCTIONS[0]]["storage"] == 2


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_cli_rejects_all_same_file_output_aliases(tmp_path, capsys, alias):
    source, output = tmp_path / "input.obj", tmp_path / "alias.obj"
    original = pack().data
    source.write_bytes(original)
    if alias == "same_path":
        output = source
    elif alias == "symlink":
        output.symlink_to(source)
    else:
        output.hardlink_to(source)
    assert main(["coff-globalize", str(source), str(output), "--function", FUNCTIONS[0]]) == 1
    assert "separate files" in json.loads(capsys.readouterr().out)["error"]
    assert source.read_bytes() == original


def test_cli_validation_failure_leaves_existing_output_unchanged(tmp_path, capsys):
    source, output = tmp_path / "input.obj", tmp_path / "output.obj"
    source.write_bytes(pack().data)
    output.write_bytes(b"existing output")
    assert main(["coff-globalize", str(source), str(output), "--function", "missing"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
    assert output.read_bytes() == b"existing output"


DATA_NAMES = ["_slot_zero", "_slot_inner"]


def data_fixture():
    sections, symbols = fixture()
    symbols.extend([
        {"name": DATA_NAMES[0], "storage": 3, "section": 3},
        {"name": DATA_NAMES[1], "storage": 3, "section": 3, "value": 2},
    ])
    # An existing code relocation refers to the interior static data symbol.
    sections[0]["relocations"] = [(0, 12, 20)]
    return sections, symbols


def test_static_data_visibility_changes_only_the_named_storage_bytes():
    source = pack(data_fixture())
    original_symbols = deepcopy(source.symbols)
    output, witness = globalize_coff_data(source, reversed(DATA_NAMES))
    result = COFF.from_bytes(output)
    assert witness["operation"] == "coff-data-visibility"
    assert witness["input_sha256"] == hashlib.sha256(source.data).hexdigest()
    assert witness["output_sha256"] == hashlib.sha256(output).hexdigest()
    assert witness["changed_bytes"] == len(witness["data"]) == 2
    offsets = {row["storage_offset"] for row in witness["data"]}
    assert {i for i, (old, new) in enumerate(zip(source.data, output)) if old != new} == offsets
    assert len(source.data) == len(output)
    assert all(source.data[offset] == 3 and output[offset] == 2 for offset in offsets)
    assert {row["value"] for row in witness["data"]} == {0, 2}
    assert all((row["old_storage"], row["new_storage"], row["type"], row["section"]) == (3, 2, 0, 3)
               for row in witness["data"])
    assert result.sections == source.sections and result.auxiliary_records == source.auxiliary_records
    assert result.sections[0]["relocations"] == [(0, 12, 20)]
    assert source.symbols == original_symbols
    for row in witness["data"]:
        assert result.named[row["name"]]["storage"] == 2


@pytest.mark.parametrize("change", ["public", "typed", "function", "aux", "common", "weak", "ambiguous", "end"])
def test_data_visibility_rejects_unsupported_symbol_records(change):
    parts = data_fixture()
    symbol = parts[1][-1]
    if change == "public":
        symbol["storage"] = 2
    elif change == "typed":
        symbol["type"] = 4
    elif change == "function":
        symbol["type"] = 0x20
    elif change == "aux":
        symbol["aux"] = bytes(18)
    elif change == "common":
        symbol.update(storage=2, section=0, value=3)
    elif change == "weak":
        symbol.update(storage=105, section=0, value=0, aux=struct.pack("<II10x", 10, 2))
    elif change == "ambiguous":
        parts[1].append({"name": DATA_NAMES[1]})
    else:
        symbol["value"] = 3
    with pytest.raises(ValueError):
        globalize_coff_data(pack(parts), [DATA_NAMES[1]])


@pytest.mark.parametrize("change", ["code", "executable", "bss", "comdat", "no_data", "remove", "discardable"])
def test_data_visibility_requires_file_backed_regular_initialized_storage(change):
    parts = data_fixture()
    section = parts[0][2]
    if change == "code":
        section["flags"] |= 0x20
    elif change == "executable":
        section["flags"] |= 0x20000000
    elif change == "bss":
        section.update(flags=0xC0000080, data=b"", size=3)
    elif change == "comdat":
        section["flags"] |= 0x1000
        parts[1][5]["aux"] = section_aux(3, selection=1)
    elif change == "no_data":
        section["flags"] &= ~0x40
    elif change == "remove":
        section["flags"] |= 0x800
    else:
        section["flags"] |= 0x02000000
    with pytest.raises(ValueError):
        globalize_coff_data(pack(parts), [DATA_NAMES[1]])


def test_data_visibility_validates_selection_and_full_object():
    for names in ([], [""], ["missing"], [DATA_NAMES[0], DATA_NAMES[0]], DATA_NAMES[0]):
        with pytest.raises(ValueError):
            globalize_coff_data(pack(data_fixture()), names)
    parts = data_fixture()
    malformed = bytearray(parts[1][0]["aux"])
    malformed[15] = 1
    parts[1][0]["aux"] = bytes(malformed)
    with pytest.raises(ValueError, match="section-definition"):
        globalize_coff_data(pack(parts), [DATA_NAMES[0]])


def test_data_visibility_rejects_symbol_table_alias_of_raw_section():
    source = pack(data_fixture())
    index = next(i for i, symbol in source.symbols.items() if symbol["name"] == DATA_NAMES[0])
    storage = struct.unpack_from("<I", source.data, 8)[0] + index * 18 + 16
    raw = bytearray(source.data)
    struct.pack_into("<I", raw, 20 + 20, storage)
    with pytest.raises(ValueError, match="Symbol table overlaps"):
        globalize_coff_data(COFF.from_bytes(raw), [DATA_NAMES[0]])


def test_data_visibility_cli_writes_witness_and_rejects_mixed_mode(tmp_path, capsys):
    source, output = tmp_path / "input.obj", tmp_path / "output.obj"
    original = pack(data_fixture()).data
    source.write_bytes(original)
    assert main(["coff-globalize", str(source), str(output), "--data", DATA_NAMES[0],
                 "--data", DATA_NAMES[1]]) == 0
    witness = json.loads(capsys.readouterr().out)
    assert witness["operation"] == "coff-data-visibility" and witness["changed_bytes"] == 2
    assert source.read_bytes() == original and COFF(output).named[DATA_NAMES[1]]["storage"] == 2
    with pytest.raises(SystemExit):
        main(["coff-globalize", str(source), str(output), "--data", DATA_NAMES[0],
              "--function", FUNCTIONS[0]])


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_data_visibility_cli_rejects_same_file_aliases(tmp_path, capsys, alias):
    source, output = tmp_path / "input.obj", tmp_path / "alias.obj"
    original = pack(data_fixture()).data
    source.write_bytes(original)
    if alias == "same_path":
        output = source
    elif alias == "symlink":
        output.symlink_to(source)
    else:
        output.hardlink_to(source)
    assert main(["coff-globalize", str(source), str(output), "--data", DATA_NAMES[0]]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["operation"] == "coff-data-visibility" and "separate files" in result["error"]
    assert source.read_bytes() == original


def test_read_only_initialized_data_is_a_valid_regular_section():
    parts = data_fixture()
    parts[0][2]["flags"] = 0x40000040
    output, witness = globalize_coff_data(pack(parts), [DATA_NAMES[1]])
    assert witness["data"][0]["value"] == 2
    assert COFF.from_bytes(output).named[DATA_NAMES[1]]["storage"] == 2


def test_data_visibility_cli_invalid_input_preserves_output(tmp_path, capsys):
    source, output = tmp_path / "input.obj", tmp_path / "output.obj"
    source.write_bytes(pack(data_fixture()).data)
    output.write_bytes(b"unchanged")
    assert main(["coff-globalize", str(source), str(output), "--data", "missing"]) == 1
    assert json.loads(capsys.readouterr().out)["operation"] == "coff-data-visibility"
    assert output.read_bytes() == b"unchanged"
