"""Executable-level PE and function layout comparison."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from binary_comp.config import ConfigError, ProjectTarget
from binary_comp.core.mapfile import function_starts_from_map, parse_msvc_map_by_obj, parse_msvc_map_symbols
from binary_comp.core.pe import EXECUTABLE_FLAG, FileBackedPEImage, PEImage, Section
from binary_comp.source.functions import load_source_groups, map_source_groups, object_member_name


DEFAULT_BYTE_SECTIONS = (".text", ".rdata", ".data")
FUNCTION_START_RE = re.compile(r"/\*\s*Function start:\s*(0x[0-9a-fA-F]+)\s*\*/")


@dataclass(frozen=True)
class ExeCompareOptions:
    byte_sections: tuple[str, ...] = DEFAULT_BYTE_SECTIONS
    include_functions: bool = False
    max_function_bytes: int = 0x10000


@dataclass(frozen=True)
class SectionComparison:
    name: str
    original: Section | None
    rebuilt: Section | None

    @property
    def matches(self) -> bool:
        return (
            self.original is not None
            and self.rebuilt is not None
            and self.original.start == self.rebuilt.start
            and self.original.virtual_size == self.rebuilt.virtual_size
        )


@dataclass(frozen=True)
class SectionByteComparison:
    name: str
    matching: int
    compared: int

    @property
    def percent(self) -> float:
        return self.matching / self.compared * 100 if self.compared else 0.0


@dataclass(frozen=True)
class FunctionComparisonRow:
    original_addr: int
    rebuilt_addr: int | None
    original_size: int
    rebuilt_size: int
    raw_match_percent: float
    status: str


@dataclass(frozen=True)
class FunctionComparisonSummary:
    total: int
    library_excluded: int
    mapped: int
    correct_address: int
    shifted: int
    missing: int
    byte_identical: int
    rows: tuple[FunctionComparisonRow, ...]


@dataclass(frozen=True)
class LinkedFunctionComparisonRow:
    module: str
    label: str
    symbol: str
    original_addr: int
    configured_size: int
    rebuilt_addr: int | None = None
    compared_size: int = 0
    matching_bytes: int | None = None
    raw_equal: bool | None = None
    status: str = "missing"
    detail: str = ""


@dataclass(frozen=True)
class LinkedFunctionComparisonSummary:
    rows: tuple[LinkedFunctionComparisonRow, ...]


@dataclass(frozen=True)
class ExeComparison:
    original_path: str
    rebuilt_path: str
    original_entry_point: int
    rebuilt_entry_point: int
    sections: tuple[SectionComparison, ...]
    byte_summaries: tuple[SectionByteComparison, ...]
    functions: FunctionComparisonSummary | None
    linked_functions: LinkedFunctionComparisonSummary | None = None


def section_by_name(image: PEImage) -> dict[str, Section]:
    return {section.name: section for section in image.sections}


def compare_sections(original: PEImage, rebuilt: PEImage) -> tuple[SectionComparison, ...]:
    original_sections = section_by_name(original)
    rebuilt_sections = section_by_name(rebuilt)
    names = list(original_sections)
    names.extend(name for name in rebuilt_sections if name not in original_sections)
    return tuple(
        SectionComparison(name, original_sections.get(name), rebuilt_sections.get(name))
        for name in names
    )


def compare_section_bytes(
    original: PEImage,
    rebuilt: PEImage,
    names: tuple[str, ...],
) -> tuple[SectionByteComparison, ...]:
    summaries: list[SectionByteComparison] = []
    for name in names:
        original_section = original.section_named(name)
        rebuilt_section = rebuilt.section_named(name)
        if original_section is None or rebuilt_section is None:
            continue

        original_data = original.read(original_section.start, original_section.rawsize)
        rebuilt_data = rebuilt.read(rebuilt_section.start, rebuilt_section.rawsize)
        if original_data is None or rebuilt_data is None:
            continue

        compared = min(len(original_data), len(rebuilt_data))
        matching = sum(1 for left, right in zip(original_data[:compared], rebuilt_data[:compared]) if left == right)
        summaries.append(SectionByteComparison(name, matching, compared))
    return tuple(summaries)


def target_function_map_dirs(target: ProjectTarget) -> tuple[str, ...]:
    map_dirs = []
    for source_dir in target.source_dirs:
        candidate = os.path.join(source_dir, "map")
        if os.path.isdir(candidate):
            map_dirs.append(candidate)
    return tuple(map_dirs)


def original_function_addresses_from_map_dirs(map_dirs: tuple[str, ...]) -> tuple[int, ...]:
    addresses: set[int] = set()
    for map_dir in map_dirs:
        for root, _, files in os.walk(map_dir):
            for filename in files:
                if not filename.endswith((".c", ".cpp", ".C")):
                    continue
                path = os.path.join(root, filename)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for match in FUNCTION_START_RE.finditer(f.read()):
                        addresses.add(int(match.group(1), 16))
    return tuple(sorted(addresses))


def original_function_addresses_from_sources(target: ProjectTarget) -> tuple[int, ...]:
    groups_by_source = load_source_groups(target.source_dirs, target.map_skip, target.source_excludes)
    addresses = {
        int(address, 16)
        for groups in groups_by_source.values()
        for group in groups
        for address in group.addresses
    }
    return tuple(sorted(addresses))


def original_function_addresses(target: ProjectTarget) -> tuple[int, ...]:
    addresses = original_function_addresses_from_map_dirs(target_function_map_dirs(target))
    if addresses:
        return addresses
    return original_function_addresses_from_sources(target)


def original_to_rebuilt_addresses(target: ProjectTarget) -> dict[int, int]:
    groups_by_source = load_source_groups(target.source_dirs, target.map_skip, target.source_excludes)
    mapped_groups, _, _ = map_source_groups(groups_by_source, target.map_path)
    mapping: dict[int, int] = {}
    for group in mapped_groups:
        for address in group.original_addrs:
            mapping[address] = group.rebuilt_addr
    return mapping


def bounded_function_size(
    address: int,
    starts: tuple[int, ...],
    section_end: int | None,
    max_bytes: int,
) -> int:
    end = section_end
    for candidate in starts:
        if candidate > address:
            end = candidate if end is None else min(end, candidate)
            break
    if end is None or end <= address:
        return 0
    return min(end - address, max_bytes)


def address_in_ranges(address: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= address <= end for start, end in ranges)


def compare_functions(
    target: ProjectTarget,
    original: PEImage,
    rebuilt: PEImage,
    max_function_bytes: int,
) -> FunctionComparisonSummary:
    original_addrs = original_function_addresses(target)
    address_map = original_to_rebuilt_addresses(target)
    rebuilt_starts = tuple(function_starts_from_map(parse_msvc_map_by_obj(target.map_path)))

    original_text = original.section_named(".text")
    rebuilt_text = rebuilt.section_named(".text")
    original_text_end = original_text.end if original_text else None
    rebuilt_text_end = rebuilt_text.end if rebuilt_text else None

    rows: list[FunctionComparisonRow] = []
    library_excluded = 0
    correct_address = 0
    shifted = 0
    missing = 0
    byte_identical = 0

    for index, original_addr in enumerate(original_addrs):
        if address_in_ranges(original_addr, target.library_ranges):
            library_excluded += 1
            continue

        if index + 1 < len(original_addrs):
            original_size = min(original_addrs[index + 1] - original_addr, max_function_bytes)
        else:
            original_size = bounded_function_size(original_addr, original_addrs, original_text_end, max_function_bytes)

        rebuilt_addr = address_map.get(original_addr)
        if rebuilt_addr is None:
            missing += 1
            rows.append(FunctionComparisonRow(original_addr, None, original_size, 0, 0.0, "missing"))
            continue

        if rebuilt_addr == original_addr:
            correct_address += 1
        else:
            shifted += 1

        rebuilt_size = bounded_function_size(rebuilt_addr, rebuilt_starts, rebuilt_text_end, max_function_bytes)
        if not rebuilt_size:
            rebuilt_size = original_size

        compare_size = min(original_size, rebuilt_size)
        original_data = original.read(original_addr, compare_size)
        rebuilt_data = rebuilt.read(rebuilt_addr, compare_size)
        matching = 0
        if original_data is not None and rebuilt_data is not None and compare_size:
            matching = sum(1 for left, right in zip(original_data, rebuilt_data) if left == right)
        raw_percent = matching / compare_size * 100 if compare_size else 0.0

        status = "ok" if rebuilt_addr == original_addr else "shifted"
        if raw_percent == 100.0 and original_size == rebuilt_size:
            status = "identical"
            byte_identical += 1

        rows.append(FunctionComparisonRow(
            original_addr,
            rebuilt_addr,
            original_size,
            rebuilt_size,
            raw_percent,
            status,
        ))

    return FunctionComparisonSummary(
        total=len(rows),
        library_excluded=library_excluded,
        mapped=len(rows) - missing,
        correct_address=correct_address,
        shifted=shifted,
        missing=missing,
        byte_identical=byte_identical,
        rows=tuple(rows),
    )


def compare_executable(
    target: ProjectTarget,
    options: ExeCompareOptions = ExeCompareOptions(),
    *,
    coff_modules: list[dict] | None = None,
) -> ExeComparison:
    functions = None
    linked_functions = None
    if target.kind == "i386-coff":
        from binary_comp.analyzers.coff_report import load_reference

        if not target.rebuilt_exe:
            raise ConfigError("i386-coff executable comparison needs rebuilt_exe")
        original, _ = load_reference(target, target.name)
        rebuilt = FileBackedPEImage(target.rebuilt_exe)
        if options.include_functions:
            if coff_modules is None:
                raise ConfigError("i386-coff function mapping needs validated COFF modules")
            if not target.map_path:
                raise ConfigError("i386-coff function mapping needs map")
            linked_functions = compare_linked_functions(target, original, rebuilt, coff_modules)
    else:
        original = PEImage(target.original_exe)
        rebuilt = PEImage(target.rebuilt_exe)
        if options.include_functions:
            functions = compare_functions(target, original, rebuilt, options.max_function_bytes)
    return ExeComparison(
        original_path=target.original_exe,
        rebuilt_path=target.rebuilt_exe,
        original_entry_point=original.entry_point,
        rebuilt_entry_point=rebuilt.entry_point,
        sections=compare_sections(original, rebuilt),
        byte_summaries=compare_section_bytes(original, rebuilt, options.byte_sections),
        functions=functions,
        linked_functions=linked_functions,
    )


def executable_window(image: FileBackedPEImage, address: int, size: int) -> bytes:
    section = image.section_for_va(address)
    if section is None or not section.flags & EXECUTABLE_FLAG:
        raise ValueError(f"VA {address:#x}+{size:#x} is not in an executable section")
    if address + size > section.start + section.rawsize:
        raise ValueError(f"VA {address:#x}+{size:#x} has no complete file-backed executable section")
    return image.read(address, size)


def compare_linked_functions(
    target: ProjectTarget,
    original: FileBackedPEImage,
    rebuilt: FileBackedPEImage,
    modules: list[dict],
) -> LinkedFunctionComparisonSummary:
    """Compare complete configured windows, without inferring linked extents."""
    from binary_comp.analyzers.coff_report import validate_counted_code

    validate_counted_code(modules, [target.name])
    if not Path(target.map_path).is_file():
        raise FileNotFoundError(f"MSVC MAP not found: {target.map_path}")
    symbols = parse_msvc_map_symbols(target.map_path)
    rows = []
    for module in modules:
        member = (module["name"] + ".obj").casefold()
        for function in module.get("functions", []):
            if target.name not in function:
                continue
            extent = function[target.name]
            row = LinkedFunctionComparisonRow(
                module["name"], function["label"], function["symbol"],
                int(extent["va"], 16), extent["size"],
            )
            original_data = executable_window(original, row.original_addr, row.configured_size)
            # Preserve full object provenance: two members with the same basename
            # do not become one identity, even if their linked VAs coincide.
            candidates = {}
            for symbol in symbols:
                if (symbol.symbol != row.symbol or symbol.object_file is None
                        or object_member_name(symbol.object_file).casefold() != member):
                    continue
                identity = (symbol.segment, symbol.va, symbol.is_function,
                            symbol.object_file.replace("\\", "/").casefold())
                candidates.setdefault(identity, symbol)
            if not candidates:
                rows.append(replace(row, detail=f"No MAP identity in {member}"))
                continue
            if len(candidates) != 1:
                rows.append(replace(row, status="ambiguous", detail=f"Multiple MAP identities in {member}"))
                continue
            symbol = next(iter(candidates.values()))
            row = replace(row, rebuilt_addr=symbol.va)
            if not symbol.is_function:
                rows.append(replace(row, status="not-function", detail="MAP entry has no f flag"))
                continue
            try:
                rebuilt_data = executable_window(rebuilt, symbol.va, row.configured_size)
            except ValueError as error:
                rows.append(replace(row, status="invalid-window", detail=str(error)))
                continue
            equal = original_data == rebuilt_data
            rows.append(replace(
                row, compared_size=row.configured_size,
                matching_bytes=sum(left == right for left, right in zip(original_data, rebuilt_data)),
                raw_equal=equal, status="raw-equal" if equal else "raw-different",
            ))
    return LinkedFunctionComparisonSummary(tuple(rows))


def format_linked_functions(summary: LinkedFunctionComparisonSummary) -> list[str]:
    rows = summary.rows
    total = len(rows)
    mapped = [row for row in rows if row.rebuilt_addr is not None and row.status != "not-function"]
    same_address = sum(row.rebuilt_addr == row.original_addr for row in mapped)
    lines = [
        "", "=== Configured linked RAW windows ===",
        f"  Configured functions: {total}",
        f"  Mapped function identities: {len(mapped)}/{total}",
        f"  At original address: {same_address}/{total}",
        f"  Shifted: {len(mapped) - same_address}/{total}",
        f"  Missing: {sum(row.status == 'missing' for row in rows)}/{total}",
        f"  Ambiguous: {sum(row.status == 'ambiguous' for row in rows)}/{total}",
        f"  Non-function MAP entries: {sum(row.status == 'not-function' for row in rows)}/{total}",
        f"  Raw-equal configured windows: {sum(row.raw_equal is True for row in rows)}/{total}",
        "  Full configured extents; exact raw operands; no relocation correction.",
        "  This diagnostic does not verify source/COFF equality, linked function extents,",
        "  behavior or whole-file equality.",
        "",
        f"  {'Original':>10} {'Rebuilt':>10} {'Size':>7} {'Matching':>15} {'Raw%':>6} Status / identity",
    ]
    for row in rows:
        rebuilt = f"0x{row.rebuilt_addr:08X}" if row.rebuilt_addr is not None else "-------"
        matching = (f"{row.matching_bytes}/{row.compared_size}"
                    if row.matching_bytes is not None else "--")
        percent = (f"{row.matching_bytes / row.compared_size * 100:.1f}%"
                   if row.compared_size else "--")
        lines.append(
            f"  0x{row.original_addr:08X} {rebuilt:>10} {row.configured_size:>7} "
            f"{matching:>15} {percent:>6} {row.status} "
            f"{row.module}:{row.label} [{row.symbol}]"
        )
        if row.detail:
            lines.append(f"    {row.detail}")
    return lines


def format_section(section: Section | None) -> tuple[str, str]:
    if section is None:
        return "(missing)", ""
    return f"0x{section.start:08X}", str(section.virtual_size)


def format_executable_comparison(comparison: ExeComparison) -> str:
    lines = [
        f"Comparing: {comparison.original_path}",
        f"     with: {comparison.rebuilt_path}",
        "",
        "=== PE Section Comparison ===",
        f"  {'Section':<10} {'Original VA':>14} {'Orig Size':>12} {'Rebuilt VA':>14} {'Rebuilt Size':>12} Match",
        f"  {'-' * 10} {'-' * 14} {'-' * 12} {'-' * 14} {'-' * 12} {'-' * 5}",
    ]

    for section in comparison.sections:
        original_va, original_size = format_section(section.original)
        rebuilt_va, rebuilt_size = format_section(section.rebuilt)
        match = "OK" if section.matches else "DIFF"
        lines.append(
            f"  {section.name:<10} {original_va:>14} {original_size:>12} "
            f" {rebuilt_va:>14} {rebuilt_size:>12} {match}"
        )

    entry_match = "OK" if comparison.original_entry_point == comparison.rebuilt_entry_point else "DIFF"
    lines.extend([
        "",
        (
            f"  Entry point: 0x{comparison.original_entry_point:08X} "
            f"vs 0x{comparison.rebuilt_entry_point:08X} {entry_match}"
        ),
    ])

    if comparison.functions is not None:
        functions = comparison.functions
        pct = functions.correct_address / functions.total * 100 if functions.total else 0.0
        lines.extend([
            "",
            "=== Function Address Mapping ===",
            f"  Total functions: {functions.total}",
            f"  Library excluded: {functions.library_excluded}",
            f"  Mapped in rebuilt: {functions.mapped}/{functions.total}",
            f"  At original address: {functions.correct_address}/{functions.total} ({pct:.1f}%)",
            f"  Shifted: {functions.shifted}/{functions.total}",
            f"  Missing: {functions.missing}/{functions.total}",
            f"  Byte-identical: {functions.byte_identical}/{functions.total}",
            "",
            f"  {'Address':>10} {'Rebuilt':>10} {'OSize':>6} {'RSize':>6} {'Raw%':>6} Status",
            f"  {'-' * 10} {'-' * 10} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 10}",
        ])
        for row in functions.rows:
            rebuilt = f"0x{row.rebuilt_addr:08X}" if row.rebuilt_addr is not None else "  -------"
            lines.append(
                f"  0x{row.original_addr:08X} {rebuilt:>10} "
                f"{row.original_size:>6} {row.rebuilt_size:>6} "
                f"{row.raw_match_percent:>5.1f}% {row.status}"
            )

        for row in functions.rows:
            if row.status in {"shifted", "missing"}:
                lines.append("")
                if row.rebuilt_addr is None:
                    lines.append(f"  First misalignment: 0x{row.original_addr:08X} (not found in rebuilt)")
                else:
                    lines.append(
                        f"  First misalignment: 0x{row.original_addr:08X} "
                        f"(expected) -> 0x{row.rebuilt_addr:08X} (actual)"
                    )
                break

    if comparison.linked_functions is not None:
        lines.extend(format_linked_functions(comparison.linked_functions))

    lines.extend(["", "=== Overall Summary ==="])
    total_matching = 0
    total_compared = 0
    for summary in comparison.byte_summaries:
        total_matching += summary.matching
        total_compared += summary.compared
        lines.append(
            f"  {summary.name:<8} {summary.matching:>8}/{summary.compared:<8} "
            f"bytes match ({summary.percent:.1f}%)"
        )
    if total_compared:
        total_percent = total_matching / total_compared * 100
        lines.append(
            f"  {'Total':<8} {total_matching:>8}/{total_compared:<8} "
            f"bytes match ({total_percent:.1f}%)"
        )
    return "\n".join(lines)
