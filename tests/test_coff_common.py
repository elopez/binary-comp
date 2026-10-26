"""Sectionless COMMON contracts using synthetic COFF records and PE images."""
from copy import deepcopy
import struct

import pytest

from binary_comp.core.coff import COFF, MAX_COMMON_BYTES, compare_function, validate_data_binding
from conftest import DATA_VA, TEXT_VA
from test_coff_contract import coff_bytes, entry, reference, report_fixture


def common_object(size=4, **symbol):
    return COFF.from_bytes(coff_bytes([], [{"name": "_buffer", "value": size, **symbol}]))


@pytest.mark.parametrize("size,kind", [(4, 0), (4, 4), (4, 0x11), (24, 0x34), (28, 0)])
def test_common_contributions_have_sizes_and_zeros_but_no_native_section(tmp_path, size, kind):
    coff = common_object(size, type=kind)
    image = reference(tmp_path, b"\xc3", bytes(32))
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": size}}
    assert coff.common_size("_buffer") == coff.definition_extent("_buffer") == size
    assert coff.materialize_common("_buffer") == bytes(size)
    assert coff.named["_buffer"]["section"] == 0
    assert coff.named["_buffer"]["value"] == size
    assert coff.sections == []
    with pytest.raises(ValueError, match="Undefined symbol"):
        coff.definition("_buffer")
    assert validate_data_binding(coff, "_buffer", bindings, image) == (bytes(size), 0)


def test_each_common_value_is_a_size_not_an_offset_in_a_shared_section(tmp_path):
    coff = COFF.from_bytes(coff_bytes([], [
        {"name": "_small", "value": 4}, {"name": "_large", "value": 24}]))
    bindings = {"_small": {"va": hex(DATA_VA), "data_size": 4},
                "_large": {"va": hex(DATA_VA + 8), "data_size": 24}}
    image = reference(tmp_path, b"\xc3", bytes(32))
    assert coff.definition_extent("_small") == 4
    assert coff.definition_extent("_large") == 24
    for name, size in [("_small", 4), ("_large", 24)]:
        assert validate_data_binding(coff, name, bindings, image) == (bytes(size), 0)


@pytest.mark.parametrize("symbol", [
    {"value": 0},                         # a pure external reference
    {"type": 0x20}, {"type": 0x24},     # typed functions, including return type
    {"type": 1}, {"type": 0x31},        # void and array of void are not data
    {"aux": bytes(18)},                  # no auxiliary COMMON allocation format
    {"storage": 105, "aux": struct.pack("<II", 2, 3) + bytes(10)},
])
def test_undefined_code_weak_and_auxiliary_records_are_not_common_data(tmp_path, symbol):
    coff = common_object(**symbol)
    assert coff.common_size("_buffer") is None
    with pytest.raises(ValueError, match="Not a COMMON"):
        coff.materialize_common("_buffer")
    with pytest.raises(ValueError, match="Undefined"):
        coff.definition_extent("_buffer")
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 4}}
    with pytest.raises(ValueError, match="Undefined"):
        validate_data_binding(coff, "_buffer", bindings, reference(tmp_path, b"\xc3", bytes(4)))


@pytest.mark.parametrize("storage", [3, 6])
def test_nonexternal_sectionless_symbol_is_not_promoted_to_a_common_owner(storage):
    coff = common_object(storage=storage)
    assert "_buffer" not in coff.named
    with pytest.raises(KeyError):
        coff.definition_extent("_buffer")


@pytest.mark.parametrize("size", [0, 3, 5, -1, True])
def test_common_binding_requires_the_whole_native_requested_size(tmp_path, size):
    coff = common_object()
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": size}}
    with pytest.raises(ValueError):
        validate_data_binding(coff, "_buffer", bindings, reference(tmp_path, b"\xc3", bytes(8)))


def test_mismatched_common_extent_is_rejected_before_reading_the_reference():
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 0xFFFFFFFF}}
    with pytest.raises(ValueError, match="extent differs from COMMON"):
        validate_data_binding(common_object(), "_buffer", bindings, None)


@pytest.mark.parametrize("position", [0, 3, 23])
def test_nonzero_original_bytes_anywhere_in_a_common_array_are_rejected(tmp_path, position):
    original = bytearray(24)
    original[position] = 1
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 24}}
    with pytest.raises(ValueError, match="Rebuilt data differs"):
        validate_data_binding(common_object(24), "_buffer", bindings,
                              reference(tmp_path, b"\xc3", original))


def test_common_constant_binding_still_requires_full_zero_payload(tmp_path):
    coff = common_object()
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_hex": "00000000"}}
    assert validate_data_binding(coff, "_buffer", bindings,
                                 reference(tmp_path, b"\xc3", bytes(4))) == (bytes(4), 0)
    bindings["_buffer"]["data_hex"] = "00000001"
    with pytest.raises(ValueError, match="Rebuilt data differs"):
        validate_data_binding(coff, "_buffer", bindings,
                              reference(tmp_path, b"\xc3", bytes.fromhex("00000001")))


@pytest.mark.parametrize("address", [DATA_VA + 29, DATA_VA + 32, -1, 0x100000000])
def test_wrong_or_truncated_original_common_windows_fail(tmp_path, address):
    bindings = {"_buffer": {"va": hex(address), "data_size": 4}}
    with pytest.raises(ValueError):
        validate_data_binding(common_object(), "_buffer", bindings,
                              reference(tmp_path, b"\xc3", bytes(32)))


@pytest.mark.parametrize("offset", [-3, 0, 3])
@pytest.mark.parametrize("other_common", [False, True])
def test_common_contribution_cannot_overlap_another_configured_data_owner(
        tmp_path, offset, other_common):
    symbols = [{"name": "_buffer", "value": 4},
               {"name": "_other", "value": 4 if other_common else 0,
                "section": 0 if other_common else 1}]
    sections = [] if other_common else [{"name": ".data", "data": bytes(4), "flags": 0xC0000040}]
    coff = COFF.from_bytes(coff_bytes(sections, symbols))
    bindings = {"_buffer": {"va": hex(DATA_VA + 4), "data_size": 4},
                "_other": {"va": hex(DATA_VA + 4 + offset), "data_size": 4}}
    with pytest.raises(ValueError, match="overlaps configured data"):
        validate_data_binding(coff, "_buffer", bindings,
                              reference(tmp_path, b"\xc3", bytes(16)))


@pytest.mark.parametrize("inline", [{"inline_data": True}, {"inline_index_for": "_other"}])
def test_common_cannot_claim_an_inline_code_section(tmp_path, inline):
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 4, **inline}}
    with pytest.raises(ValueError, match="no inline section"):
        validate_data_binding(common_object(), "_buffer", bindings,
                              reference(tmp_path, b"\xc3", bytes(4)))


@pytest.mark.parametrize("size", [MAX_COMMON_BYTES + 1, 0xFFFFFFFF])
def test_large_common_sizes_are_metadata_without_unbounded_materialization(size):
    coff = common_object(size)
    assert coff.common_size("_buffer") == coff.definition_extent("_buffer") == size
    with pytest.raises(ValueError, match="materialization limit"):
        coff.materialize_common("_buffer")
    # The guard runs before any reference read or large bytes allocation.
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": size}}
    with pytest.raises(ValueError, match="materialization limit"):
        validate_data_binding(coff, "_buffer", bindings, None)


def test_common_request_cannot_cross_the_i386_address_domain():
    class ZeroReference:
        def read(self, address, size):
            return bytes(size)

    bindings = {"_buffer": {"va": "0xfffffffe", "data_size": 4}}
    with pytest.raises(ValueError, match="outside i386 address domain"):
        validate_data_binding(common_object(), "_buffer", bindings, ZeroReference())


@pytest.mark.parametrize("order", [False, True])
def test_common_definition_wins_over_an_unresolved_declaration(order):
    symbols = [{"name": "_buffer"}, {"name": "_buffer", "value": 4}]
    if order:
        symbols.reverse()
    coff = COFF.from_bytes(coff_bytes([], symbols))
    assert coff.common_size("_buffer") == 4
    assert all(coff.resolve_symbol(i)["value"] == 4 for i in (0, 1))


@pytest.mark.parametrize("order", [False, True])
def test_real_section_definition_wins_over_a_common_declaration(order):
    symbols = [{"name": "_buffer", "section": 1}, {"name": "_buffer", "value": 4}]
    if order:
        symbols.reverse()
    coff = COFF.from_bytes(coff_bytes([
        {"name": ".data", "data": bytes(8), "flags": 0xC0000040}], symbols))
    assert coff.common_size("_buffer") is None
    assert coff.definition_extent("_buffer") == 8
    assert all(coff.resolve_symbol(i)["section"] == 1 for i in (0, 1))


def test_conflicting_common_sizes_in_one_object_are_not_coalesced():
    with pytest.raises(ValueError, match="Conflicting COMMON sizes"):
        COFF.from_bytes(coff_bytes([], [
            {"name": "_buffer", "value": 4}, {"name": "_buffer", "value": 8}]))


@pytest.mark.parametrize("order", [False, True])
def test_strong_common_definition_wins_over_a_same_name_weak_fallback(order):
    common = {"name": "_buffer", "value": 4}
    weak = {"name": "_buffer", "storage": 105,
            "aux": struct.pack("<II", 3, 3) + bytes(10)}
    symbols = [weak, common] if order else [common, weak]
    symbols.append({"name": "_fallback"})
    coff = COFF.from_bytes(coff_bytes([], symbols))
    assert coff.common_size("_buffer") == 4
    assert coff.resolve_symbol(0)["value"] == 4
    assert coff.resolve_symbol(1 if not order else 2)["value"] == 4


def common_reference(tmp_path):
    raw = b"\xa1" + bytes(4) + b"\xc3"
    coff = COFF.from_bytes(coff_bytes([
        {"data": raw, "relocations": [(1, 1, 6)]}], [
        {"name": "_read", "section": 1, "type": 0x24},
        {"name": "_buffer", "value": 24}]))
    image = reference(tmp_path, b"\xa1" + struct.pack("<I", DATA_VA) + b"\xc3", bytes(32))
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 24}}
    return coff, entry("_read", TEXT_VA, 6), bindings, image


def test_complete_function_reference_checks_the_whole_common_contribution(tmp_path):
    coff, function, bindings, image = common_reference(tmp_path)
    exact, actual, expected, fixups = compare_function(coff, function, "sample", bindings, image)
    assert exact and actual == expected and fixups == 1
    for mutation in ["address", "extent", "tail"]:
        changed = deepcopy(bindings)
        original = image
        if mutation == "address":
            changed["_buffer"]["va"] = hex(DATA_VA + 4)
        elif mutation == "extent":
            changed["_buffer"]["data_size"] = 4
        else:
            original = reference(tmp_path, expected, bytes(23) + b"\1" + bytes(8))
        if mutation == "address":
            assert not compare_function(coff, function, "sample", changed, original)[0]
        else:
            with pytest.raises(ValueError):
                compare_function(coff, function, "sample", changed, original)


@pytest.mark.parametrize("original_kind", ["zero", "nonzero", "truncated"])
@pytest.mark.parametrize("weak", [False, True])
def test_referenced_common_cannot_bypass_full_data_validation_with_only_an_address(
        tmp_path, original_kind, weak):
    coff, function, bindings, image = common_reference(tmp_path)
    if weak:
        coff = COFF.from_bytes(coff_bytes([
            {"data": b"\xa1" + bytes(4) + b"\xc3", "relocations": [(1, 1, 6)]}], [
            {"name": "_read", "section": 1, "type": 0x24},
            {"name": "_weak", "storage": 105,
             "aux": struct.pack("<II", 3, 3) + bytes(10)},
            {"name": "_buffer", "value": 24}]))
    address = DATA_VA + 16 if original_kind == "truncated" else DATA_VA
    expected = b"\xa1" + struct.pack("<I", address) + b"\xc3"
    original_data = bytes(23) + b"\1" + bytes(8) if original_kind == "nonzero" else bytes(32)
    image = reference(tmp_path, expected, original_data)
    if original_kind == "truncated":
        with pytest.raises(ValueError, match="no complete file-backed section"):
            image.read(address, 24)
    bindings["_buffer"]["va"] = hex(address)
    del bindings["_buffer"]["data_size"]
    with pytest.raises(ValueError, match="Defined data symbol needs complete data validation"):
        compare_function(coff, function, "sample", bindings, image)


def test_pure_undefined_reference_can_still_use_an_address_only_binding(tmp_path):
    coff = COFF.from_bytes(coff_bytes([
        {"data": b"\xa1" + bytes(4) + b"\xc3", "relocations": [(1, 1, 6)]}], [
        {"name": "_read", "section": 1, "type": 0x24}, {"name": "_external"}]))
    expected = b"\xa1" + struct.pack("<I", DATA_VA) + b"\xc3"
    image = reference(tmp_path, expected, bytes(4))
    assert compare_function(coff, entry("_read", TEXT_VA, 6), "sample",
                            {"_external": {"va": hex(DATA_VA)}}, image)[0]


def test_report_cannot_go_green_when_a_common_reference_has_no_data_row(tmp_path, monkeypatch):
    coff, function, bindings, image = common_reference(tmp_path)
    del bindings["_buffer"]["data_size"]
    failures, result, *_ = report_fixture(tmp_path, monkeypatch, coff, [function], bindings, image)
    assert failures > 0
    assert result["data"] == []
    assert "needs complete data validation" in result["code"][0]["error"]
    assert "relocated_bytes" not in result["code"][0]


def test_common_report_counts_zero_file_bytes_and_zero_data_fixups(tmp_path, monkeypatch):
    coff, function, bindings, image = common_reference(tmp_path)
    failures, result, output, *_ = report_fixture(
        tmp_path, monkeypatch, coff, [function], bindings, image)
    assert failures == 0
    assert result["code"][0]["relocated_bytes"]["exact"]
    assert result["code"][0]["checked_fixups"] == 1
    assert result["data"] == [{"version": "sample", "module": "Synthetic", "symbol": "_buffer",
                               "exact": True, "size": 24, "checked_fixups": 0,
                               "storage": "common", "object_file_bytes": 0}]
    assert "not final linker allocation" in result["semantics"]["data"]
    assert "24 bytes; 0 checked data fixups" in (output / "report.txt").read_text()


def test_data_only_common_report_fails_after_a_nonzero_unused_byte(tmp_path, monkeypatch):
    coff = common_object(24)
    bindings = {"_buffer": {"va": hex(DATA_VA), "data_size": 24}}
    for bad in [False, True]:
        image = reference(tmp_path, b"\xc3", bytes(23) + (b"\1" if bad else b"\0"))
        failures, result, *_ = report_fixture(tmp_path, monkeypatch, coff, [], bindings, image)
        assert bool(failures) is bad
        assert result["code"] == []
        assert result["data"][0]["exact"] is not bad
        if bad:
            assert "Rebuilt data differs" in result["data"][0]["error"]
