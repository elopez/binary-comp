"""Linked raw data diagnostics for explicit, native-validated COFF contracts."""

from dataclasses import dataclass, replace
from pathlib import Path

from binary_comp.analyzers.coff_report import (
    _code_bindings, load_common_definitions, load_object, load_reference,
    ordinary_entries, region_context, register_external_bindings, validate_counted_code,
)
from binary_comp.analyzers.exe import linked_map_identity
from binary_comp.analyzers.coff_linked_graph import resolve_private_placements
from binary_comp.config import ConfigError
from binary_comp.core.coff import data_check_only_size, is_function_type, validate_data_binding
from binary_comp.core.mapfile import parse_msvc_map_symbols
from binary_comp.core.pe import FileBackedPEImage
from binary_comp.source.functions import object_member_name


@dataclass(frozen=True)
class LinkedDataRow:
    module: str
    symbol: str
    original_address: int
    size: int
    owner: str
    native_section: int | None = None
    native_offset: int | None = None
    contract_status: str = "unresolved"
    checked_fixups: int | None = None
    rebuilt_address: int | None = None
    map_object: str | None = None
    original_backing: str = "unmapped"
    rebuilt_backing: str = "unmapped"
    placement_equal: bool | None = None
    raw_equal: bool | None = None
    matching_bytes: int | None = None
    compared_size: int = 0
    status: str = "missing"
    detail: str = ""
    resolution: str = ""
    resolution_provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkedDataSummary:
    rows: tuple[LinkedDataRow, ...]

    @property
    def matches(self):
        return all(row.placement_equal is True and row.raw_equal is True for row in self.rows)


def data_window(image, address, size):
    """Classify one complete interval; virtual zeros never become raw bytes."""
    if not 0 <= address < address + size <= 0x100000000:
        return "invalid-window", None
    sections = [section for section in image.sections
                if section.start < address + size and address < section.end]
    if len(sections) != 1 or not sections[0].start <= address < address + size <= sections[0].end:
        return "invalid-window", None
    section = sections[0]
    raw_end = section.start + section.rawsize
    if address >= raw_end:
        return "virtual-zero-fill", None
    if address + size > raw_end:
        return "mixed-file-zero-fill", None
    return "file-backed", image.read(address, size)


def _compare_window(row, original, rebuilt):
    _, original_data = data_window(original, row.original_address, row.size)
    backing, rebuilt_data = data_window(rebuilt, row.rebuilt_address, row.size)
    row = replace(row, rebuilt_backing=backing)
    if row.contract_status != "validated":
        return replace(row, status=row.contract_status)
    if original_data is None or rebuilt_data is None:
        return replace(row, status="non-file-backed-window")
    equal = original_data == rebuilt_data
    return replace(row, raw_equal=equal, compared_size=row.size,
                   matching_bytes=sum(a == b for a, b in zip(original_data, rebuilt_data)),
                   status="raw-equal" if equal else "raw-different")


def compare_linked_data(target, modules, *, allow_empty=False):
    """Check strict object contracts separately from exact linked raw windows."""
    if not target.rebuilt_exe or not target.map_path:
        raise ConfigError("i386-coff data comparison needs rebuilt_exe and map")
    if not Path(target.map_path).is_file():
        raise FileNotFoundError(f"MSVC MAP not found: {target.map_path}")
    validate_counted_code(modules, [target.name])
    original, _ = load_reference(target, target.name)
    rebuilt = FileBackedPEImage(target.rebuilt_exe)
    symbols = parse_msvc_map_symbols(target.map_path)
    rows, externals = [], {}
    for module in modules:
        bindings = _code_bindings(module.get("bindings", {}).get(target.name, {}),
                                  ordinary_entries(module), target.name)
        selected = [(name, binding) for name, binding in bindings.items()
                    if ("data_hex" in binding or "data_size" in binding)
                    and data_check_only_size(name, binding) is None]
        if not selected:
            continue
        module_rows, validated_data = [], {}
        coff = load_object(module, target)
        providers, _ = load_common_definitions(module, target, target.name)
        register_external_bindings(externals, coff, bindings, target.name, module["name"])
        for name, provider in providers.items():
            register_external_bindings(externals, provider, bindings, target.name,
                                       module["common_definitions"][target.name][name])
        for name, binding in selected:
            if ("data_hex" in binding) == ("data_size" in binding):
                raise ConfigError(f"Data contract needs exactly one complete extent: {name}")
            size = len(bytes.fromhex(binding["data_hex"])) if "data_hex" in binding else binding["data_size"]
            if type(size) is not int or size <= 0:
                raise ConfigError(f"Invalid complete data extent: {name}")
            owner = module["name"] + ".obj"
            row = LinkedDataRow(module["name"], name, int(binding["va"], 16), size, owner)
            backing, _ = data_window(original, row.original_address, size)
            row = replace(row, original_backing=backing)
            definition = None
            try:
                common = coff.common_size(name)
                native = coff
                if common is not None:
                    if name not in providers:
                        module_rows.append(replace(row, contract_status="common-request", status="common-request",
                                                   detail="Sectionless COMMON request does not prove linked allocation"))
                        continue
                    native = providers[name]
                    owner = module["common_definitions"][target.name][name]
                    row = replace(row, owner=owner)
                if native.named[name]["section"] <= 0:
                    raise KeyError(name)
                definition, section = native.definition(name)
                storage = (2, 3, 6) if binding.get("inline_data") is True else (2, 3)
                if (definition["storage"] not in storage or definition["aux_count"]
                        or is_function_type(definition["type"]) or "weak_fallback" in definition):
                    raise ValueError("No strong native data definition")
                if not (binding.get("inline_data") is True or binding.get("executable_data") is True):
                    if (not section["flags"] & 0xC0
                            or section["flags"] & (0x20 | 0x200 | 0x800 | 0x02000000 | 0x20000000)
                            or ((definition["type"] & 0xF) == 1
                                and (definition["type"] & 0x30) != 0x10)):
                        raise ValueError("No native ordinary data contribution")
                row = replace(row, native_section=definition["section"], native_offset=definition["value"])
                context = region_context(coff, module, target.name, bindings, original, providers)
                _, fixups = validate_data_binding(coff, name, bindings, original,
                                                  common_definitions=providers, code_regions=context)
                row = replace(row, contract_status="validated", checked_fixups=fixups)
                if native is coff:
                    validated_data[name] = size, fixups
            except KeyError:
                row = replace(row, detail="No owned native data definition")
            except ValueError as error:
                row = replace(row, contract_status="object-contract-invalid", detail=str(error))

            matching = [symbol for symbol in symbols if symbol.symbol == name and symbol.object_file is not None]
            candidates = {linked_map_identity(symbol): symbol for symbol in matching
                          if object_member_name(symbol.object_file).casefold() == owner.casefold()}
            if not candidates and row.contract_status == "validated" and definition["storage"] == 2:
                candidates = {linked_map_identity(symbol): symbol for symbol in matching
                              if symbol.visibility == "public" and not symbol.is_function}
            if len(candidates) != 1:
                module_rows.append(replace(row, status="ambiguous" if candidates else "missing"))
                continue
            mapped = next(iter(candidates.values()))
            row = replace(row, rebuilt_address=mapped.va,
                          map_object=mapped.object_file,
                          placement_equal=mapped.va == row.original_address, resolution="map")
            if mapped.is_function:
                module_rows.append(replace(row, status="not-data"))
                continue
            module_rows.append(_compare_window(row, original, rebuilt))
        if any(row.status == "missing" and row.contract_status == "validated"
               and row.symbol in validated_data for row in module_rows):
            context = region_context(coff, module, target.name, bindings, original, providers)
            placements = resolve_private_placements(
                coff, module, target.name, bindings, original, rebuilt, symbols,
                validated_data, providers=providers, regions=context, read_window=data_window)
            for index, row in enumerate(module_rows):
                proof = placements.get(row.symbol)
                if row.status != "missing" or row.contract_status != "validated" or proof is None:
                    continue
                if proof.address is None:
                    module_rows[index] = replace(row, detail=proof.detail)
                else:
                    row = replace(row, rebuilt_address=proof.address,
                                  placement_equal=proof.address == row.original_address,
                                  resolution="relocation-graph",
                                  resolution_provenance=proof.provenance)
                    module_rows[index] = _compare_window(row, original, rebuilt)
        rows.extend(module_rows)
    if not rows and not allow_empty:
        raise ConfigError("No complete owned COFF data contracts selected")
    return LinkedDataSummary(tuple(rows))


def format_linked_data(summary, *, verbose=False):
    rows = summary.rows
    lines = ["Configured linked RAW data windows",
             f"  Configured data contracts: {len(rows)}",
             f"  Validated object contracts: {sum(r.contract_status == 'validated' for r in rows)}/{len(rows)}",
             f"  At original address: {sum(r.placement_equal is True for r in rows)}/{len(rows)}",
             f"  Raw-equal file-backed windows: {sum(r.raw_equal is True for r in rows)}/{len(rows)}",
             "  Exact raw bytes; no relocation correction, scanning or inferred extents.",
             "  Object contracts use the strict data validator; linked windows are a separate diagnostic.",
             "  Virtual zeros do not prove allocation extent or raw equality.",
             "  No claim of linked allocation extent, behavior or whole-file equality.", ""]
    if not rows:
        lines.append("  No configured complete data contracts for this target.")
    for row in rows:
        rebuilt = f"0x{row.rebuilt_address:08X}" if row.rebuilt_address is not None else "-------"
        matching = f"{row.matching_bytes}/{row.compared_size}" if row.matching_bytes is not None else "--"
        placement = "same" if row.placement_equal is True else "shifted" if row.placement_equal is False else "unresolved"
        offset = f"+{row.native_offset:#x}" if row.native_offset is not None else ""
        lines.append(f"  0x{row.original_address:08X} {rebuilt} size={row.size} raw={matching} "
                     f"placement={placement} {row.status} {row.module}:{row.symbol}")
        lines.append(f"    owner={row.owner} section={row.native_section or '--'} offset={offset or '--'} "
                     f"object={row.contract_status} "
                     f"fixups={row.checked_fixups if row.checked_fixups is not None else '--'} "
                     f"backing={row.original_backing}/{row.rebuilt_backing}")
        if row.map_object:
            lines.append(f"    MAP identity: {row.map_object}")
        if row.resolution == "relocation-graph":
            lines.append("    Placement: validated actual COFF relocation graph")
        if verbose:
            lines.extend(f"    {line}" for line in row.resolution_provenance)
        if verbose and row.detail:
            lines.append(f"    {row.detail}")
    return "\n".join(lines)
