"""Strict COFF comparison contracts, using only synthetic object/image bytes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from binary_comp.core.coff import (
    BodyExtentError, COFF, apply_fixups, code_window, compare_function,
    validate_data_binding,
)
from binary_comp.core.pe import FileBackedPEImage
from conftest import DATA_VA, TEXT_VA, write_tiny_pe


CODE_FLAGS = 0x60000020
DATA_FLAGS = 0x40000040
BSS_FLAGS = 0xC0000080


def coff_bytes(sections, symbols):
    """Build i386 COFF records; relocation indices include auxiliary records."""
    header_size = 20 + 40 * len(sections)
    payload = bytearray()
    headers = bytearray()
    for section in sections:
        data = section.get("data", b"")
        flags = section.get("flags", CODE_FLAGS)
        bss = bool(flags & 0x80)
        size = section["size"] if bss else len(data)
        raw_offset = 0 if bss else header_size + len(payload)
        if not bss:
            payload += data
        relocations = section.get("relocations", [])
        reloc_offset = header_size + len(payload) if relocations else 0
        payload += b"".join(struct.pack("<IIH", *item) for item in relocations)
        headers += struct.pack("<8sIIIIIIHHI", section.get("name", ".text").encode(),
                               0, 0, size, raw_offset, reloc_offset, 0,
                               len(relocations), 0, flags)
    strings = bytearray(b"\0" * 4)
    records = bytearray()
    count = 0
    for symbol in symbols:
        name = symbol["name"].encode()
        if len(name) > 8:
            name = struct.pack("<II", 0, len(strings))
            strings += symbol["name"].encode() + b"\0"
        aux = symbol.get("aux", b"")
        assert len(aux) % 18 == 0
        records += struct.pack("<8sIhHBB", name, symbol.get("value", 0),
                               symbol.get("section", 0), symbol.get("type", 0),
                               symbol.get("storage", 2), len(aux) // 18)
        records += aux
        count += 1 + len(aux) // 18
    struct.pack_into("<I", strings, 0, len(strings))
    header = struct.pack("<HHIIIHH", 0x14C, len(sections), 0,
                         header_size + len(payload), count, 0, 0)
    return header + headers + payload + records + strings


def reference(tmp_path, code, data=None):
    path = tmp_path / ("original-" + hashlib.sha256(code + (data or b"")).hexdigest()[:8] + ".exe")
    write_tiny_pe(path, code, {0: data} if data is not None else None)
    return FileBackedPEImage(path)


def entry(symbol, address, size):
    return {"symbol": symbol, "label": symbol,
            "sample": {"va": hex(address), "size": size}}


def check(fixture):
    coff, function, bindings, image = fixture
    return compare_function(coff, function, "sample", bindings, image)


@pytest.fixture
def linked_function(tmp_path):
    # A field load, a literal address, and a call. Only the two actual fixups
    # may change; the field displacement and every other operand remain exact.
    raw = bytes.fromhex("8b 41 24 68 00 00 00 00 e8 00 00 00 00 c3 90 90")
    symbols = [{"name": "_function", "section": 1, "type": 0x20},
               {"name": "_literal", "section": 2}, {"name": "_callee"}]
    coff = COFF.from_bytes(coff_bytes([
        {"data": raw, "relocations": [(4, 1, 6), (9, 2, 0x14)]},
        {"name": ".rdata", "data": b"tag\0", "flags": DATA_FLAGS},
    ], symbols))
    expected = bytearray(raw[:14])
    struct.pack_into("<I", expected, 4, DATA_VA)
    struct.pack_into("<i", expected, 9, (TEXT_VA + 0x80) - (TEXT_VA + 13))
    bindings = {"_literal": {"va": hex(DATA_VA), "data_size": 4},
                "_callee": {"va": hex(TEXT_VA + 0x80)}}
    return coff, entry("_function", TEXT_VA, 14), bindings, reference(tmp_path, expected, b"tag\0")


def test_explicit_absolute_and_relative_fixups_preserve_raw_and_relocated_bytes(linked_function):
    ok, actual, expected, fixups = check(linked_function)
    assert ok and actual == expected
    assert fixups == 2
    assert linked_function[0].function("_function")["data"][:14] != expected


@pytest.mark.parametrize("mutation", ["callee", "relative_addend", "absolute_addend", "field"])
def test_wrong_callee_addends_and_nonrelocated_operands_never_normalize(linked_function, mutation):
    coff, _, bindings, _ = fixture = deepcopy(linked_function)
    if mutation == "callee":
        bindings["_callee"]["va"] = hex(TEXT_VA + 0x81)
    else:
        section = coff.function("_function")
        raw = bytearray(section["data"])
        raw[{"relative_addend": 9, "absolute_addend": 4, "field": 2}[mutation]] += 1
        section["data"] = bytes(raw)
    assert not check(fixture)[0]


@pytest.mark.parametrize("mutation", ["missing", "extent", "unvalidated", "unused_byte", "constant"])
def test_referenced_data_must_validate_its_complete_definition(linked_function, mutation):
    coff, _, bindings, _ = fixture = deepcopy(linked_function)
    if mutation == "missing":
        del bindings["_literal"]
    elif mutation == "extent":
        bindings["_literal"]["data_size"] = 3
    elif mutation == "unvalidated":
        del bindings["_literal"]["data_size"]
    elif mutation == "constant":
        bindings["_literal"] = {"va": hex(DATA_VA), "data_hex": b"bad\0".hex()}
    else:
        coff.definition("_literal")[1]["data"] = b"tagX"
    with pytest.raises(ValueError):
        check(fixture)


@pytest.mark.parametrize("relocations", [
    [(4, 1, 6), (5, 2, 6)],  # overlapping fixups
    [(12, 2, 6)],             # crosses the complete code boundary
    [(4, 1, 0x999)],          # unsupported relocation kind
    [(4, 999, 6)],            # nonexistent symbol index
])
def test_invalid_fixups_fail_closed(linked_function, relocations):
    fixture = deepcopy(linked_function)
    fixture[0].function("_function")["relocations"] = relocations
    with pytest.raises(ValueError):
        check(fixture)


def test_non_nop_tail_is_not_trimmed_but_complete_diagnostic_window_survives(linked_function):
    coff, function, bindings, image = fixture = deepcopy(linked_function)
    section = coff.function("_function")
    section["data"] = section["data"][:14] + b"\xcc\xc3"
    with pytest.raises(BodyExtentError, match="Non-NOP"):
        check(fixture)
    window = code_window(coff, section, TEXT_VA, 14, bindings, image, diagnostic=True)
    actual, fixups = apply_fixups(coff, window["data"], window["relocations"],
                                  TEXT_VA, window["bindings"], image)
    assert len(actual) == 16 and actual[-2:] == b"\xcc\xc3"
    assert fixups == 2


@pytest.fixture
def weak_and_bss(tmp_path):
    # The weak symbol's auxiliary record occupies index 2; its strong fallback
    # is index 3. Index 0 is an earlier unresolved declaration of that fallback.
    symbols = [{"name": "_impl"},
               {"name": "_weak", "storage": 105,
                "aux": struct.pack("<II", 3, 3) + bytes(10)},
               {"name": "_impl", "section": 1, "type": 0x20},
               {"name": "_vtable", "section": 2},
               {"name": "_history", "section": 3}]
    coff = COFF.from_bytes(coff_bytes([
        {"data": b"\xc3"},
        {"name": ".rdata", "data": bytes(8), "flags": DATA_FLAGS,
         "relocations": [(0, 1, 6), (4, 0, 6)]},
        {"name": ".bss", "size": 8, "flags": BSS_FLAGS},
    ], symbols))
    bindings = {"_impl": {"va": hex(TEXT_VA)},
                "_vtable": {"va": hex(DATA_VA), "data_size": 8},
                "_history": {"va": hex(DATA_VA + 8), "data_size": 8}}
    image = reference(tmp_path, b"\xc3", struct.pack("<II", TEXT_VA, TEXT_VA) + bytes(8))
    return coff, bindings, image


def test_forward_declarations_weak_fallbacks_vtable_and_bss_use_complete_storage(weak_and_bss):
    coff, bindings, image = weak_and_bss
    assert coff.resolve_symbol(0)["section"] == 1
    assert coff.resolve_symbol(1)["name"] == "_impl"
    assert validate_data_binding(coff, "_vtable", bindings, image) == (struct.pack("<II", TEXT_VA, TEXT_VA), 2)
    assert coff.definition("_history")[1]["flags"] & 0x80
    assert validate_data_binding(coff, "_history", bindings, image) == (bytes(8), 0)


@pytest.mark.parametrize("mutation", ["cycle", "unknown", "partial_bss", "unused_bss_byte", "pointer_addend"])
def test_weak_and_storage_errors_cannot_hide_behind_address_bindings(weak_and_bss, mutation):
    coff, bindings, image = deepcopy(weak_and_bss)
    name = "_vtable"
    if mutation in ("cycle", "unknown"):
        coff.symbols[1]["weak_fallback"] = 1 if mutation == "cycle" else 999
    elif mutation == "pointer_addend":
        coff.definition(name)[1]["data"] = b"\1" + bytes(7)
    else:
        name = "_history"
        if mutation == "partial_bss":
            bindings[name]["data_size"] = 4
        else:
            coff.definition(name)[1]["data"] = bytes(7) + b"\1"
    with pytest.raises(ValueError):
        validate_data_binding(coff, name, bindings, image)


@pytest.fixture
def eh_labels(tmp_path):
    cleanup = b"\xe8" + bytes(4) + b"\xc3"
    handler = b"\x68" + bytes(4) + b"\xe9" + bytes(4)
    symbols = [{"name": "$cleanup", "section": 1, "storage": 6},
               {"name": "$handler", "section": 1, "storage": 6, "value": 6},
               {"name": "_delete"}, {"name": "$unwind", "section": 2},
               {"name": "_frame_handler"}]
    coff = COFF.from_bytes(coff_bytes([
        {"name": ".text$x", "data": cleanup + handler,
         "relocations": [(1, 2, 0x14), (7, 3, 6), (12, 4, 0x14)]},
        {"name": ".xdata", "data": bytes(8), "flags": DATA_FLAGS,
         "relocations": [(0, 0, 6)]},
    ], symbols))
    expected = bytearray(b"\x90" * 32)
    expected[:6] = b"\xe8" + struct.pack("<i", 0x80 - 5) + b"\xc3"
    expected[16:26] = b"\x68" + struct.pack("<I", DATA_VA) + b"\xe9" + struct.pack("<i", 0x90 - 26)
    bindings = {"$cleanup": {"va": hex(TEXT_VA)},
                "$unwind": {"va": hex(DATA_VA), "data_size": 8},
                "_delete": {"va": hex(TEXT_VA + 0x80)},
                "_frame_handler": {"va": hex(TEXT_VA + 0x90)}}
    image = reference(tmp_path, expected, struct.pack("<II", TEXT_VA, 0))
    return coff, bindings, image


def test_shared_section_eh_entries_keep_separate_code_and_complete_unwind_data(eh_labels):
    coff, bindings, image = eh_labels
    for name, address, size in [("$cleanup", TEXT_VA, 6), ("$handler", TEXT_VA + 16, 10)]:
        assert len(coff.function(name)["data"]) == size
        assert compare_function(coff, entry(name, address, size), "sample", bindings, image)[0]
    assert validate_data_binding(coff, "$unwind", bindings, image) == (struct.pack("<II", TEXT_VA, 0), 1)


@pytest.mark.parametrize("mutation", ["neighbor", "crossing_fixup", "missing_cleanup", "partial_unwind", "unwind_tail"])
def test_eh_code_boundaries_and_unwind_dependencies_fail_closed(eh_labels, mutation):
    coff, bindings, image = deepcopy(eh_labels)
    function = entry("$handler", TEXT_VA + 16, 10)
    if mutation == "neighbor":
        function = entry("$cleanup", TEXT_VA, 7)
    elif mutation == "crossing_fixup":
        function = entry("$cleanup", TEXT_VA, 6)
        coff.sections[0]["relocations"].append((4, 2, 0x14))
    elif mutation == "missing_cleanup":
        del bindings["$cleanup"]
    elif mutation == "partial_unwind":
        bindings["$unwind"]["data_size"] = 4
    else:
        coff.definition("$unwind")[1]["data"] = bytes(7) + b"\1"
    with pytest.raises(ValueError):
        compare_function(coff, function, "sample", bindings, image)


@pytest.fixture
def inline_switch(tmp_path):
    # Byte-index lookup, indirect jump, two returning cases, native MOV EDI,EDI
    # alignment, then a complete pointer table and its byte-index suffix.
    code = bytes.fromhex("8b 44 24 04 0f b6 80 00 00 00 00 ff 24 85 00 00 00 00 31 c0 c3 b8 01 00 00 00 c3 90 90 90 8b ff")
    raw = code + bytes(8) + b"\0\1\0\x90"
    symbols = [{"name": "_switch", "section": 1, "type": 0x20},
               {"name": "$case0", "section": 1, "value": 18, "storage": 6},
               {"name": "$case1", "section": 1, "value": 21, "storage": 6},
               {"name": "$pointers", "section": 1, "value": 32, "storage": 6},
               {"name": "$indices", "section": 1, "value": 40, "storage": 6}]
    coff = COFF.from_bytes(coff_bytes([
        {"data": raw, "relocations": [(7, 4, 6), (14, 3, 6), (32, 1, 6), (36, 2, 6)]},
    ], symbols))
    expected = bytearray(raw)
    for offset, value in [(7, TEXT_VA + 40), (14, TEXT_VA + 32),
                          (32, TEXT_VA + 18), (36, TEXT_VA + 21)]:
        struct.pack_into("<I", expected, offset, value)
    bindings = {symbol["name"]: {"va": hex(TEXT_VA + symbol.get("value", 0))} for symbol in symbols}
    bindings["_switch"]["size"] = 32
    bindings["$pointers"].update(data_size=8, inline_data=True)
    bindings["$indices"].update(data_size=3, inline_data=True, inline_index_for="$pointers")
    return coff, entry("_switch", TEXT_VA, 32), bindings, reference(tmp_path, expected)


def test_inline_pointer_and_index_tables_have_separate_complete_spans(inline_switch):
    coff, function, bindings, image = inline_switch
    assert check(inline_switch)[0]
    window = code_window(coff, coff.function("_switch"), TEXT_VA, 32, bindings, image)
    assert len(window["data"]) == 32
    assert window["data"][-2:] == b"\x8b\xff"
    assert window["inline_data_bytes"] == 11
    assert validate_data_binding(coff, "$pointers", bindings, image)[1] == 2
    assert validate_data_binding(coff, "$indices", bindings, image) == (b"\0\1\0", 0)
    with pytest.raises(ValueError, match="not separate code entries"):
        coff.function("$case0")


@pytest.mark.parametrize("mutation", ["pointer_size", "index_size", "missing_index", "relation", "index_range", "index_byte", "case_address", "case_addend", "pointer_fixup", "gap", "padding_fixup"])
def test_inline_tables_cannot_hide_partial_payloads_gaps_or_bad_targets(inline_switch, mutation):
    coff, _, bindings, _ = fixture = deepcopy(inline_switch)
    section = coff.function("_switch")
    raw = bytearray(section["data"])
    if mutation == "pointer_size":
        bindings["$pointers"]["data_size"] = 4
    elif mutation == "index_size":
        bindings["$indices"]["data_size"] = 2
    elif mutation == "missing_index":
        del bindings["$indices"]
    elif mutation == "relation":
        del bindings["$indices"]["inline_index_for"]
    elif mutation == "index_range":
        raw[40] = 2
    elif mutation == "index_byte":
        raw[40] = 1
    elif mutation == "case_address":
        bindings["$case0"]["va"] = hex(TEXT_VA + 19)
    elif mutation == "case_addend":
        raw[32] = 1
    elif mutation == "pointer_fixup":
        section["relocations"][-1] = (36, 2, 0x14)
    elif mutation == "padding_fixup":
        section["relocations"].append((43, 1, 6))
    else:
        coff.named["$indices"]["value"] = 41
        bindings["$indices"]["va"] = hex(TEXT_VA + 41)
    section["data"] = bytes(raw)
    with pytest.raises(ValueError):
        check(fixture)


def test_alignment_instruction_is_never_silently_excluded_from_code(inline_switch):
    fixture = deepcopy(inline_switch)
    fixture[1]["sample"]["size"] -= 2
    with pytest.raises(BodyExtentError, match="Non-NOP"):
        check(fixture)
    fixture = deepcopy(inline_switch)
    section = fixture[0].function("_switch")
    raw = bytearray(section["data"])
    raw[31] ^= 1
    section["data"] = bytes(raw)
    assert not check(fixture)[0]


def aligned_switch_tables(tmp_path, gap=b"\x8b\xff", tail=b"\x90" * 5):
    # Two native pointer/index pairs. The first index payload's length, not
    # its padding, determines how many byte indices are validated and counted.
    indices = bytes(i % 2 for i in range(4 - len(gap)))
    raw = b"\x31\xc0\xc3\x90" + bytes(8) + indices + gap + bytes(8) + b"\1\0\1" + tail
    symbols = [{"name": "_dispatch", "section": 1, "type": 0x20},
               {"name": "$zero", "section": 1, "value": 0, "storage": 6},
               {"name": "$return", "section": 1, "value": 2, "storage": 6},
               {"name": "$first", "section": 1, "value": 4, "storage": 6},
               {"name": "$first_index", "section": 1, "value": 12, "storage": 6},
               {"name": "$second", "section": 1, "value": 16, "storage": 6},
               {"name": "$second_index", "section": 1, "value": 24, "storage": 6}]
    relocations = [(4, 1, 6), (8, 2, 6), (16, 2, 6), (20, 1, 6)]
    coff = COFF.from_bytes(coff_bytes([{"data": raw, "relocations": relocations}], symbols))
    expected = bytearray(raw)
    for offset, index, _ in relocations:
        struct.pack_into("<I", expected, offset, TEXT_VA + symbols[index]["value"])
    bindings = {s["name"]: {"va": hex(TEXT_VA + s.get("value", 0))} for s in symbols}
    bindings["_dispatch"]["size"] = 4
    for name in ("$first", "$second"):
        bindings[name].update(inline_data=True, data_size=8)
    bindings["$first_index"].update(inline_data=True, data_size=len(indices), inline_index_for="$first")
    bindings["$second_index"].update(inline_data=True, data_size=3, inline_index_for="$second")
    return coff, entry("_dispatch", TEXT_VA, 4), bindings, reference(tmp_path, expected)


@pytest.mark.parametrize("gap", [b"", b"\x90", b"\x90\x90", b"\x8b\xff", b"\x8d\x49\0"])
def test_complete_inline_pairs_allow_exact_native_four_byte_alignment(tmp_path, gap):
    fixture = aligned_switch_tables(tmp_path, gap)
    coff, _, bindings, image = fixture
    assert check(fixture)[0]
    window = code_window(coff, coff.function("_dispatch"), TEXT_VA, 4, bindings, image)
    assert window["inline_data_bytes"] == 23 - len(gap)
    assert window["inline_alignment_bytes"] == len(gap) + 5
    assert len(window["data"]) == 4
    assert validate_data_binding(coff, "$first_index", bindings, image)[0] == bytes(
        i % 2 for i in range(4 - len(gap)))


def test_final_inline_alignment_accepts_only_bounded_exact_short_instructions(tmp_path):
    fixture = aligned_switch_tables(tmp_path, tail=b"\x8b\xff\x8d\x49\0")
    assert check(fixture)[0]
    coff, _, bindings, image = fixture
    window = code_window(coff, coff.function("_dispatch"), TEXT_VA, 4, bindings, image)
    assert window["inline_alignment_bytes"] == 7
    assert window["excluded_nop_padding"] == 0


@pytest.mark.parametrize("mutation", ["changed_gap", "unknown_gap", "changed_tail", "short_index",
                                     "long_index", "padding_fixup", "crossing_fixup", "next_symbol",
                                     "missing_pair", "excess_tail"])
def test_inline_alignment_never_hides_changed_bytes_partial_tables_or_fixups(tmp_path, mutation):
    coff, _, bindings, image = fixture = aligned_switch_tables(tmp_path)
    section = coff.function("_dispatch")
    raw = bytearray(section["data"])
    if mutation == "changed_gap":
        raw[14:16] = b"\x90\x90"  # Known padding still must equal the reference.
    elif mutation == "unknown_gap":
        raw[14:16] = b"\xeb\0"
    elif mutation == "changed_tail":
        raw[27:32] = b"\x8b\xff\x8d\x49\0"
    elif mutation == "short_index":
        bindings["$first_index"]["data_size"] = 1
    elif mutation == "long_index":
        bindings["$first_index"]["data_size"] = 3
    elif mutation == "padding_fixup":
        section["relocations"].append((14, 1, 6))
    elif mutation == "crossing_fixup":
        section["relocations"].append((11, 1, 6))
    elif mutation == "next_symbol":
        coff.named["$second"]["value"] = 17
        bindings["$second"]["va"] = hex(TEXT_VA + 17)
    elif mutation == "missing_pair":
        del bindings["$second"]
        del bindings["$second_index"]
    elif mutation == "excess_tail":
        raw += b"\x90" * 16
    section["data"] = bytes(raw)
    with pytest.raises(ValueError):
        check(fixture)
    if mutation in ("changed_gap", "unknown_gap", "short_index", "long_index", "padding_fixup",
                    "crossing_fixup", "next_symbol"):
        with pytest.raises(ValueError):
            validate_data_binding(coff, "$first_index", bindings, image)


def test_pre_table_lea_alignment_stays_in_the_complete_code_window(tmp_path):
    coff, function, bindings, image = aligned_switch_tables(tmp_path)
    section = coff.function("_dispatch")
    section["data"] = b"\xc3\x8d\x49\0" + section["data"][4:]
    expected = b"\xc3\x8d\x49\0" + image.read(TEXT_VA + 4, 28)
    fixture = coff, function, bindings, reference(tmp_path, expected)
    assert check(fixture)[0]
    function["sample"]["size"] = 1
    with pytest.raises(BodyExtentError, match="Non-NOP"):
        check(fixture)


def report_fixture(tmp_path, monkeypatch, coff, functions, bindings, image, code_labels=None):
    pytest.importorskip("capstone")
    from binary_comp.analyzers import coff_report

    module = {"name": "Synthetic", "functions": functions,
              "code_labels": code_labels or [], "bindings": {"sample": bindings}}
    targets = {"sample": {"original_exe": image.path, "object_dir": str(tmp_path),
                           "original_sha256": hashlib.sha256(image.data).hexdigest()}}
    monkeypatch.setattr(coff_report, "load_object", lambda module, target: coff)
    output = tmp_path / "report"
    failures = coff_report.report([module], targets, ["sample"], output)
    return failures, json.loads((output / "report.json").read_text()), output, module, targets


def test_report_retains_rejected_full_window_without_claiming_acceptance(linked_function, tmp_path, monkeypatch):
    coff, function, bindings, image = deepcopy(linked_function)
    section = coff.function("_function")
    section["data"] = section["data"][:14] + b"\xcc\xc3"
    failures, result, output, *_ = report_fixture(tmp_path, monkeypatch, coff, [function], bindings, image)
    row, = result["code"]
    assert failures > 0
    assert "Non-NOP" in row["error"]
    assert row["diagnostic_only"] is True
    assert row["comparison_window"] == "complete_coff_function_section"
    assert row["coff_window_size"] == 16
    assert row["decoded_bytes"] == {"original": 14, "rebuilt": 16}
    assert not row["raw_bytes"]["exact"] and not row["relocated_bytes"]["exact"]
    assert "including any compiler padding" in (output / row["review"]).read_text()
    assert "0/1 relocated-byte equal" in (output / "report.txt").read_text()


@pytest.mark.parametrize("unreferenced_data", [False, True])
def test_rejected_diagnostics_do_not_fabricate_metrics_after_binding_or_data_errors(
        linked_function, weak_and_bss, tmp_path, monkeypatch, unreferenced_data):
    if unreferenced_data:
        coff, bindings, image = deepcopy(weak_and_bss)
        function = entry("_impl", TEXT_VA, 1)
        coff.function("_impl")["data"] += b"\xcc\xc3"
        coff.definition("_history")[1]["data"] = bytes(7) + b"\1"
    else:
        coff, function, bindings, image = deepcopy(linked_function)
        section = coff.function("_function")
        section["data"] = section["data"][:14] + b"\xcc\xc3"
        del bindings["_callee"]
    failures, result, _, *_ = report_fixture(tmp_path, monkeypatch, coff, [function], bindings, image)
    row, = result["code"]
    assert failures > 0 and "error" in row and "diagnostic_error" in row
    assert "relocated_bytes" not in row and "decoded_bytes" not in row
    assert "mnemonic_similarity" not in row and "review" not in row


def test_report_keeps_eh_labels_and_unwind_data_separate(eh_labels, tmp_path, monkeypatch):
    coff, bindings, image = eh_labels
    labels = [entry("$cleanup", TEXT_VA, 6), entry("$handler", TEXT_VA + 16, 10)]
    failures, result, _, *_ = report_fixture(tmp_path, monkeypatch, coff, [], bindings, image, labels)
    assert failures == 0
    assert [row["kind"] for row in result["code"]] == ["label", "label"]
    assert all(row["relocated_bytes"]["exact"] for row in result["code"])
    assert sum(row["size"] for row in result["data"]) == 8
    assert sum(row["checked_fixups"] for row in result["data"]) == 1


def test_report_decodes_code_only_and_counts_each_native_table_once(inline_switch, tmp_path, monkeypatch):
    coff, function, bindings, image = inline_switch
    failures, result, output, module, targets = report_fixture(
        tmp_path, monkeypatch, coff, [function], bindings, image)
    assert failures == 0
    row, = result["code"]
    assert row["relocated_bytes"]["exact"] and not row["raw_bytes"]["exact"]
    assert row["decoded_bytes"] == {"original": 32, "rebuilt": 32}
    assert row["inline_data_bytes"] == 11
    assert row["checked_fixups"] == 2
    assert sum(row["size"] for row in result["data"]) == 11
    assert sum(row["checked_fixups"] for row in result["data"]) == 2
    from binary_comp.analyzers import coff_report

    module["functions"][0]["sample"]["size"] = 33
    assert coff_report.report([module], targets, ["sample"], output) > 0
    rejected = json.loads((output / "report.json").read_text())["code"][0]
    assert "overlaps native inline data" in rejected["error"]
    assert "decoded_bytes" not in rejected
    module["functions"][0]["sample"]["size"] = 30
    assert coff_report.report([module], targets, ["sample"], output) > 0
    diagnostic = json.loads((output / "report.json").read_text())["code"][0]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["comparison_window"] == "coff_code_before_inline_data"
    assert diagnostic["decoded_bytes"]["rebuilt"] == 32
    assert "0/1 relocated-byte equal" in (output / "report.txt").read_text()


def test_report_keeps_inline_alignment_separate_and_rejects_changed_padding(tmp_path, monkeypatch):
    coff, function, bindings, image = aligned_switch_tables(tmp_path)
    failures, result, output, module, targets = report_fixture(
        tmp_path, monkeypatch, coff, [function], bindings, image)
    assert failures == 0
    row, = result["code"]
    assert row["relocated_bytes"]["exact"]
    assert row["decoded_bytes"] == {"original": 4, "rebuilt": 4}
    assert (row["inline_data_bytes"], row["inline_alignment_bytes"]) == (21, 7)
    assert sum(item["size"] for item in result["data"]) == 21
    section = coff.function("_dispatch")
    section["data"] = section["data"][:14] + b"\x90\x90" + section["data"][16:]
    from binary_comp.analyzers import coff_report

    assert coff_report.report([module], targets, ["sample"], output) > 0
    rejected = json.loads((output / "report.json").read_text())["code"][0]
    assert "alignment differs" in rejected["error"]
    assert "decoded_bytes" not in rejected
    assert "relocated_bytes" not in rejected


def test_invalid_inputs_remove_stale_reports_and_reviews(linked_function, tmp_path, monkeypatch):
    from binary_comp.analyzers import coff_report

    coff, function, bindings, image = linked_function
    failures, result, output, module, targets = report_fixture(
        tmp_path, monkeypatch, coff, [function], bindings, image)
    assert failures == 0
    review = output / result["code"][0]["review"]
    assert review.is_file()
    del bindings["_callee"]
    assert coff_report.report([module], targets, ["sample"], output) > 0
    row = json.loads((output / "report.json").read_text())["code"][0]
    assert "relocated_bytes" not in row and not review.exists()
    # A changed original must invalidate an earlier summary before raising.
    image_path = Path(image.path)
    image_path.write_bytes(image_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        coff_report.report([module], targets, ["sample"], output)
    assert not (output / "report.json").exists()
    assert not (output / "report.txt").exists()
