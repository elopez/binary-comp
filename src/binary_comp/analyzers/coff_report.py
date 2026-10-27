"""Strict i386 COFF object comparison and native binary-comp reports.

Objects are supplied by the caller's build. All original targets, complete
extents and data definitions come from an explicit manifest; no linker MAP,
source annotations or synthetic executable is needed.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re

from binary_comp.config import ConfigError, load_json, load_project_target
from binary_comp.core.coff import (COFF, CodeRegions, BodyExtentError, apply_fixups, code_window, compare_code_region,
                                   compare_function, is_function_type, validate_common_definitions,
                                   validate_data_binding)
from binary_comp.core.pe import FileBackedPEImage

PE = FileBackedPEImage
CODE_ROLES = (("functions", "function"), ("code_labels", "label"), ("code_regions", "region"))
CODE_KEYS = tuple(key for key, _ in CODE_ROLES)


def region_contracts(module):
    """Filtering retains declared region dependencies without counting their rows."""
    selected = module.get("code_regions", [])
    return selected + [entry for entry in module.get("_code_region_dependencies", [])
                       if entry not in selected]


def ordinary_entries(module):
    return [f for key in ("functions", "code_labels") for f in module.get(key, [])]


def region_context(coff, module, version, bindings, reference, common_definitions):
    contracts = region_contracts(module)
    if not any(version in entry for entry in contracts):
        return None
    context = CodeRegions(coff, contracts, version, bindings, reference,
                          common_definitions=common_definitions)
    selected = ordinary_entries(module)
    owners = selected + [entry for entry in module.get("_ordinary_code_owners", [])
                         if entry not in selected]
    for entry in owners:
        if version in entry and entry["symbol"] in coff.named:
            _, contribution = coff.definition(entry["symbol"])
            if id(contribution) in context.sections:
                raise ValueError("Duplicate COFF code contribution ownership")
    return context


def load_object(module, target):
    """Load one caller-built object; compilation remains outside this backend."""
    return COFF(object_path(target, module["name"] + ".obj"))


def object_directories(target):
    """Accept parsed targets and direct API dictionaries without root precedence."""
    if isinstance(target, dict):
        if "object_dir" in target and "object_dirs" in target:
            raise ConfigError("object_dir and object_dirs are mutually exclusive")
        if "object_dirs" in target:
            directories = target["object_dirs"]
            if (not isinstance(directories, list) or not directories
                    or any(not isinstance(path, str) or not path for path in directories)):
                raise ConfigError("object_dirs must be a nonempty list of directory strings")
            return tuple(Path(path) for path in directories)
    elif getattr(target, "object_dirs", ()):
        if target.object_dir is not None:
            raise ConfigError("object_dir and object_dirs are mutually exclusive")
        return tuple(Path(path) for path in target.object_dirs)
    return (Path(target_value(target, "object_dir")),)


def object_path(target, filename):
    """Require exactly one basename match, including explicit COMMON providers."""
    directories = object_directories(target)
    matches = [directory / filename for directory in directories
               if (directory / filename).is_file()]
    if not matches:
        raise FileNotFoundError(f"COFF object not found: {filename}; searched "
                                + ", ".join(str(path) for path in directories))
    if len(matches) != 1:
        raise ConfigError(f"Ambiguous COFF object: {filename}; matches "
                          + ", ".join(str(path) for path in matches))
    return matches[0]


def _common_definition_records(module):
    records = module.get("common_definitions", {})
    if not isinstance(records, dict):
        raise ConfigError("common_definitions must map target names to provider objects")
    for version, names in records.items():
        _component(version, "COMMON provider target")
        if not isinstance(names, dict):
            raise ConfigError("COMMON provider targets must map symbols to object basenames")
        for name, filename in names.items():
            if not isinstance(name, str) or not name:
                raise ConfigError("COMMON provider needs a nonempty native symbol")
            _component(filename, "COMMON provider object")
            if not filename.lower().endswith(".obj") or ":" in filename:
                raise ConfigError("COMMON provider object must be an .obj basename")
    return records


def load_common_definitions(module, target, version):
    """Load only explicit providers from the configured object directories."""
    names = _common_definition_records(module).get(version, {})
    objects = {}
    for filename in names.values():
        if filename not in objects:
            objects[filename] = COFF(object_path(target, filename))
    return {name: objects[filename] for name, filename in names.items()}, {
        f"{version}/{filename}": hashlib.sha256(coff.data).hexdigest()
        for filename, coff in objects.items()}


def target_value(target, name):
    value = target.get(name) if isinstance(target, dict) else getattr(target, name, None)
    if not value:
        raise ValueError(f"COFF target needs {name}")
    return value


def load_reference(target, version):
    """Check the configured original digest before reading any comparison window."""
    path = Path(target_value(target, "original_exe"))
    expected_hash = target_value(target, "original_sha256")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError(f"Target {version} needs an original_sha256 digest")
    if actual_hash != expected_hash.lower():
        raise ValueError(f"Original SHA-256 mismatch for {version}: {path}")
    return PE(path), actual_hash


def source_hashes(modules, source_root):
    """Source labels are optional; objects and references are always hashed."""
    if source_root is None:
        return {}
    result = {}
    for module in modules:
        source = module.get("source")
        if source and (Path(source_root) / source).is_file():
            result[source] = hashlib.sha256((Path(source_root) / source).read_bytes()).hexdigest()
    return result


def manifest_hashes(manifest_path, modules):
    """Keep the single-file digest and retain every composed manifest input."""
    if manifest_path is None:
        return {"modules_sha256": hashlib.sha256(
            json.dumps(modules, sort_keys=True).encode()).hexdigest()}
    paths = ((Path(manifest_path),) if isinstance(manifest_path, (str, Path))
             else tuple(Path(path) for path in manifest_path))
    digests = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if len(paths) == 1:
        return {"modules_sha256": digests[str(paths[0])]}
    return {"modules_sha256": hashlib.sha256(
                json.dumps([digests[str(path)] for path in paths]).encode()).hexdigest(),
            "module_manifests_sha256": digests}


def _component(value, label):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value):
        raise ConfigError(f"{label} must be a nonempty path component")
    return value


def _code_bindings(bindings, entries, version):
    """Add counted code addresses without hiding an explicit contradiction."""
    result = dict(bindings)
    for entry in entries:
        if version not in entry:
            continue
        explicit = result.get(entry["symbol"])
        if explicit is not None and int(explicit["va"], 16) != int(entry[version]["va"], 16):
            raise ConfigError(f"Conflicting configured code address for {entry['symbol']}")
        result[entry["symbol"]] = {**(explicit or {}), **entry[version]}
    return result


def load_selection(config_path, target_name, module_names=None, function_name=None):
    """Load explicit modules and arbitrary target names without source parsing."""
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    target_configs = config.get("targets", {})
    if not isinstance(target_configs, dict):
        raise ConfigError("targets must be an object")
    names = ([name for name, item in target_configs.items()
              if isinstance(item, dict) and item.get("kind") == "i386-coff"]
             if target_name == "all" else [target_name])
    if not names:
        raise ConfigError("No i386-coff targets configured")
    targets = {}
    for name in names:
        _component(name, "COFF target name")
        _, target = load_project_target(config_path, name)
        if target.kind != "i386-coff":
            raise ConfigError(f"Target {name} is not i386-coff")
        object_directories(target)
        digest = target_value(target, "original_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ConfigError(f"Target {name} needs an original_sha256 digest")
        targets[name] = target
    section = config.get("coff")
    manifests = section.get("modules") if isinstance(section, dict) else None
    if isinstance(manifests, str) and manifests:
        manifests = [manifests]
    if (not isinstance(manifests, list) or not manifests
            or any(not isinstance(path, str) or not path for path in manifests)):
        raise ConfigError("coff.modules must name a manifest or a nonempty list of manifests")
    paths = tuple(config_path.parent / path for path in manifests)
    modules = []
    for path in paths:
        entries = json.loads(path.read_text())
        if not isinstance(entries, list):
            raise ConfigError(f"COFF module manifest must be a list: {path}")
        modules.extend(entries)
    seen = set()
    for module in modules:
        if not isinstance(module, dict):
            raise ConfigError("COFF modules must be objects")
        if {"_code_region_dependencies", "_ordinary_code_owners"} & module.keys():
            raise ConfigError("COFF module uses reserved selection metadata")
        name = _component(module.get("name"), "COFF module name")
        if name in seen:
            raise ConfigError(f"Duplicate COFF module: {name}")
        seen.add(name)
        if "source" in module and not isinstance(module["source"], str):
            raise ConfigError(f"{name}.source must be a path label")
        providers = _common_definition_records(module)
        if providers.keys() - target_configs.keys():
            raise ConfigError(f"Unknown COMMON provider target in {name}")
        for key in CODE_KEYS:
            entries = module.get(key, [])
            if not isinstance(entries, list):
                raise ConfigError(f"{name}.{key} must be a list")
            for entry in entries:
                if (not isinstance(entry, dict) or not isinstance(entry.get("label"), str)
                        or not isinstance(entry.get("symbol"), str)):
                    raise ConfigError(f"Invalid COFF code entry in {name}.{key}")
                if key == "code_regions":
                    if not isinstance(entry.get("entries"), list) or not entry["entries"]:
                        raise ConfigError("Code region needs an exhaustive typed-entry list")
                    anchor_offset = entry.get("anchor_offset", 0)
                    if type(anchor_offset) is not int or anchor_offset < 0:
                        raise ConfigError("Invalid code region anchor_offset")
                elif "anchor_offset" in entry:
                    raise ConfigError("anchor_offset is only supported for code_regions")
                for target_name in targets:
                    if target_name not in entry:
                        continue
                    binding = entry[target_name]
                    if (not isinstance(binding, dict) or type(binding.get("size")) is not int
                            or binding["size"] <= 0 or not isinstance(binding.get("va"), str)):
                        raise ConfigError(f"Invalid complete COFF extent for {entry['label']}")
                    int(binding["va"], 16)
                    if key == "code_regions" and anchor_offset >= binding["size"]:
                        raise ConfigError("Code region anchor_offset outside complete extent")
        bindings = module.get("bindings", {})
        if not isinstance(bindings, dict) or any(not isinstance(b, dict) for b in bindings.values()):
            raise ConfigError(f"{name}.bindings must map target names to binding objects")
        for target_name in targets:
            for symbol, binding in bindings.get(target_name, {}).items():
                if (not isinstance(symbol, str) or not isinstance(binding, dict)
                        or not isinstance(binding.get("va"), str)):
                    raise ConfigError(f"Invalid COFF address binding in {name}")
                int(binding["va"], 16)
                if "data_hex" in binding and not isinstance(binding["data_hex"], str):
                    raise ConfigError(f"Invalid COFF data_hex for {symbol}")
                if "data_size" in binding and (type(binding["data_size"]) is not int
                                                or binding["data_size"] <= 0):
                    raise ConfigError(f"Invalid COFF data_size for {symbol}")
            _code_bindings(bindings.get(target_name, {}),
                           ordinary_entries(module), target_name)
    if module_names:
        if set(module_names) - seen:
            raise ConfigError("Unknown COFF module: " + ", ".join(sorted(set(module_names) - seen)))
        modules = [m for m in modules if m["name"] in module_names]
    if function_name is not None:
        if any(module.get("code_regions") for module in modules):
            validate_counted_code(modules, list(targets))
        selected = []
        for module in modules:
            entries = [(key, f) for key in CODE_KEYS
                       for f in module.get(key, [])
                       if function_name in (f["label"], f["symbol"])
                       and any(name in f for name in targets)]
            if not entries:
                continue
            narrowed = dict(module)
            narrowed["_code_region_dependencies"] = region_contracts(module)
            narrowed["_ordinary_code_owners"] = ordinary_entries(module)
            # All function addresses remain available to code and data fixups.
            narrowed["bindings"] = {
                name: _code_bindings(module.get("bindings", {}).get(name, {}),
                                     ordinary_entries(module), name)
                for name in targets}
            for key in CODE_KEYS:
                narrowed[key] = [f for role, f in entries if role == key]
            selected.append(narrowed)
        modules = selected
        if not modules:
            raise ConfigError(f"COFF function not found: {function_name}")
    return modules, targets, paths[0] if len(paths) == 1 else paths


def run_config(config_path, target_name, output, *, module_names=None,
               function_name=None, file_filter=None, build=False, show_reviews=False):
    """Run the same strict backend from the compare and report CLI commands."""
    from binary_comp.analyzers.function_compare import maybe_build

    # Invalidate summaries even when selection or configuration fails.
    output = Path(output)
    for filename in ("report.json", "report.txt"):
        (output / filename).unlink(missing_ok=True)
    modules, targets, manifest = load_selection(config_path, target_name, module_names, function_name)
    if file_filter:
        modules = [m for m in modules if file_filter in m["name"]
                   or file_filter in m.get("source", "")
                   or any(file_filter in f["label"] for key in CODE_KEYS
                          for f in m.get(key, []))]
    # Build and input errors must not leave a previous green summary behind.
    for target in targets.values():
        maybe_build(target, build)
    section = load_json(config_path)["coff"]
    source_root = section.get("source_root", ".")
    if not isinstance(source_root, str):
        raise ConfigError("coff.source_root must be a path string")
    source_root = Path(config_path).resolve().parent / source_root
    failures = report(modules, targets, list(targets), output,
                      manifest_path=manifest, source_root=source_root)
    if show_reviews:
        result = json.loads((output / "report.json").read_text())
        for row in result["code"]:
            if "review" in row:
                print((output / row["review"]).read_text())
    return int(bool(failures))


def byte_metrics(original, rebuilt):
    from binary_comp.core.binary import compare_binary
    result = compare_binary(original, rebuilt)
    return {**asdict(result), "positional_identity": result.positional_identity,
            "exact": result.exact}


def compare_windows(label, address, original, raw, relocated):
    """All bytes participate in equality; the independent mnemonic score has no gate."""
    from binary_comp.analyzers.function_compare import (
        DisassemblyResult, FunctionComparison, format_comparison,
        instruction_mnemonics, mnemonic_similarity,
    )
    from binary_comp.core.binary import compare_binary, format_binary_comparison
    from binary_comp.core.disasm import disassemble_raw

    rebuilt_instructions = disassemble_raw(relocated, address, bits=32)
    original_instructions = disassemble_raw(original, address, bits=32)
    decoded = {"rebuilt": sum(i.size for i in rebuilt_instructions),
               "original": sum(i.size for i in original_instructions)}
    complete = decoded == {"rebuilt": len(relocated), "original": len(original)}
    similarity = mnemonic_similarity(instruction_mnemonics(rebuilt_instructions),
                                     instruction_mnemonics(original_instructions)) if complete else None
    result = {"raw_bytes": byte_metrics(original, raw),
              "relocated_bytes": byte_metrics(original, relocated),
              "mnemonic_similarity": similarity, "decoded_bytes": decoded}
    review = [format_binary_comparison(compare_binary(original, raw),
                                      title="Original vs raw COFF body (no fixups)"),
              format_binary_comparison(compare_binary(original, relocated),
                                      title="Original vs explicitly relocated body (no masks)")]
    if complete:
        comparison = FunctionComparison(label, address, address, similarity,
                                        DisassemblyResult(rebuilt_instructions, []),
                                        DisassemblyResult(original_instructions, []))
        review += ["Left: rebuilt after explicit fixups. Right: original.",
                   format_comparison(comparison)]
    else:
        # Never show 100% from a prefix that happens to decode successfully.
        result["metric_error"] = "Linear decoding did not cover both complete byte windows"
        review.append(result["metric_error"])
    review.append("Mnemonic similarity ignores operands and encodings; it does not establish equality.")
    return result, "\n\n".join(review) + "\n"


def diagnostic_bytes(coff, section, function, version, bindings, reference, *,
                     common_definitions=None, code_regions=None):
    """Relocate a rejected complete code window without disassembly or report output."""
    target = function[version]
    address, size = int(target["va"], 16), target["size"]
    original = reference.read(address, size)
    window = code_window(coff, section, address, size, bindings, reference, diagnostic=True,
                         common_definitions=common_definitions, code_regions=code_regions)
    bindings = window["bindings"]
    code_regions = window.get("code_regions", code_regions)
    # Validate even unreferenced configured data before publishing diagnostics.
    # A length failure in the strict gate occurs before its relocation checks.
    for name, binding in bindings.items():
        if "data_hex" in binding or "data_size" in binding:
            validate_data_binding(coff, name, bindings, reference,
                                  common_definitions=common_definitions, code_regions=code_regions)
    raw = window["data"]
    rebuilt, fixups = apply_fixups(coff, raw, window["relocations"], address,
                                  bindings, reference, common_definitions=common_definitions,
                                  code_regions=window.get("code_regions", code_regions))
    return window, original, rebuilt, fixups


def diagnostic_window(coff, section, function, version, bindings, reference, *,
                      common_definitions=None, code_regions=None):
    """Review a rejected code extent, keeping explicitly validated native data separate."""
    window, original, rebuilt, fixups = diagnostic_bytes(
        coff, section, function, version, bindings, reference,
        common_definitions=common_definitions, code_regions=code_regions)
    address, size = int(function[version]["va"], 16), function[version]["size"]
    raw = window["data"]
    metrics, review = compare_windows(function["label"], address, original, raw, rebuilt)
    inline = "inline_data_bytes" in window
    description = ("complete COFF code prefix before validated inline data" if inline
                   else "complete COFF function section")
    note = (f"Diagnostic only: {description} ({len(raw)} bytes), "
            f"including any compiler padding before native data, versus bounded original body ({size} bytes). "
            "The strict matcher rejected this extent; these metrics do not accept it.")
    metrics.update(diagnostic_only=True,
                   comparison_window="coff_code_before_inline_data" if inline else "complete_coff_function_section",
                   coff_window_size=len(raw), checked_fixups=fixups)
    if inline:
        metrics.update(inline_data_bytes=window["inline_data_bytes"],
                       inline_alignment_bytes=window["inline_alignment_bytes"],
                       inline_data_symbols=window["inline_data_symbols"])
    return metrics, note + "\n\n" + review


def score_config(config_path, target_name, function_name, object_path, *, module_names=None):
    """Score one external candidate with zero reserved for the complete strict gate.

    Valid code differences return positional byte differences after explicit
    fixups. Rejected extents use the diagnostic window and a positive minimum;
    invalid configuration, native roles, data or fixups remain errors.
    """
    from binary_comp.core.binary import compare_binary

    if target_name == "all" or (module_names is not None and len(module_names) != 1):
        raise ConfigError("COFF scoring requires exactly one target and module")
    modules, targets, _ = load_selection(config_path, target_name, module_names, function_name)
    if len(targets) != 1 or len(modules) != 1:
        raise ConfigError("COFF scoring requires exactly one target and module")
    module = modules[0]
    functions = [f for f in module.get("functions", []) if target_name in f]
    labels = [f for f in module.get("code_labels", []) if target_name in f]
    if len(functions) != 1 or labels or any(target_name in r for r in module.get("code_regions", [])):
        raise ConfigError("COFF scoring requires exactly one native function")
    validate_counted_code(modules, [target_name])
    reference, _ = load_reference(targets[target_name], target_name)
    coff = COFF(object_path)
    common_definitions, _ = load_common_definitions(module, targets[target_name], target_name)
    function = functions[0]
    validate_native_role(coff, function["symbol"], "function")
    bindings = _code_bindings(module.get("bindings", {}).get(target_name, {}),
                              functions, target_name)
    code_regions = region_context(coff, module, target_name, bindings, reference, common_definitions)
    validate_common_definitions(coff, bindings, reference, common_definitions, code_regions=code_regions)
    try:
        # Unreferenced configured definitions must pass the same data gate too.
        for name, binding in bindings.items():
            if "data_hex" in binding or "data_size" in binding:
                validate_data_binding(coff, name, bindings, reference,
                                      common_definitions=common_definitions, code_regions=code_regions)
        try:
            _, rebuilt, original, _ = compare_function(
                coff, function, target_name, bindings, reference,
                common_definitions=common_definitions, code_regions=code_regions)
        except BodyExtentError:
            _, original, rebuilt, _ = diagnostic_bytes(
                coff, coff.function(function["symbol"]), function, target_name, bindings, reference,
                common_definitions=common_definitions, code_regions=code_regions)
            return max(1, compare_binary(original, rebuilt).differing_positions)
    except KeyError as error:
        raise ValueError(f"Missing configured COFF symbol: {error.args[0]}") from error
    return compare_binary(original, rebuilt).differing_positions


def validate_counted_code(modules, versions):
    """Disjoint original owners across code roles, including region dependencies."""
    for version in versions:
        symbols = set()
        windows, regions, data = [], [], []
        for module in modules:
            entries = ordinary_entries(module) + region_contracts(module)
            for entry in entries:
                if version not in entry:
                    continue
                symbol = entry["symbol"]
                identity = module["name"], symbol
                if identity in symbols:
                    raise ValueError(f"Duplicate counted COFF symbol for {version}: {symbol}")
                symbols.add(identity)
                binding = entry[version]
                start, size = int(binding["va"], 16), binding["size"]
                if type(size) is not int or size <= 0 or not 0 <= start < start + size <= 0x100000000:
                    raise ValueError(f"Invalid counted COFF window for {version}: {symbol}")
                window = start, start + size, module["name"], symbol
                windows.append(window)
                if entry in region_contracts(module):
                    regions.append(window)
            for name, binding in module.get("bindings", {}).get(version, {}).items():
                if "data_hex" in binding or "data_size" in binding:
                    size = (len(bytes.fromhex(binding["data_hex"])) if "data_hex" in binding
                            else binding["data_size"])
                    start = int(binding["va"], 16)
                    if type(size) is not int or size <= 0 or not 0 <= start < start + size <= 0x100000000:
                        raise ValueError("Invalid complete COFF data range")
                    data.append((start, start + size, module["name"], name))
        windows.sort()
        for left, right in zip(windows, windows[1:]):
            if left[1] > right[0]:
                raise ValueError(f"Overlapping original COFF code windows for {version}: "
                                 f"{left[2]}:{left[3]} and {right[2]}:{right[3]}")
        for region in regions:
            for definition in data:
                if region[0] < definition[1] and definition[0] < region[1]:
                    raise ValueError("Code region overlaps configured complete data")


def validate_native_role(coff, symbol, kind):
    """Use native symbol records, never disassembly, to classify counted code."""
    try:
        definition, _ = coff.definition(symbol)
    except KeyError as error:
        raise ValueError(f"Counted COFF symbol is not defined: {symbol}") from error
    if kind == "function":
        if not is_function_type(definition["type"]):
            raise ValueError(f"Counted COFF function needs native function type: {symbol}")
    elif kind == "label":
        if definition["storage"] != 6 or definition["type"] != 0:
            raise ValueError(f"Counted COFF code label needs native storage 6/type 0: {symbol}")
        # The core also excludes local case labels inside a /Gy function section.
        coff.function(symbol)


def register_external_bindings(seen, coff, bindings, version, owner):
    """Keep selected objects' native external address assertions consistent.

    Weak names and absolute/debug records do not identify ordinary external
    addresses. A resolved strong fallback is checked under its own native name.
    """
    for name, binding in bindings.items():
        symbol = coff.named.get(name)
        if symbol is None or symbol["storage"] != 2 or symbol["section"] < 0:
            continue
        if symbol["section"] > len(coff.sections):
            raise ValueError(f"Invalid native external section for {owner}:{name}")
        address = int(binding["va"], 16)
        if not 0 <= address <= 0xFFFFFFFF:
            raise ValueError(f"External binding outside i386 address domain: {owner}:{name}")
        previous = seen.get(name)
        if previous is not None and previous[0] != address:
            raise ValueError(f"Conflicting external binding for {version} {name}: "
                             f"{previous[1]}={previous[0]:#x}, {owner}={address:#x}")
        seen[name] = address, owner


def report(modules, targets, versions, output, *, manifest_path=None, source_root=None):
    rows, data_rows = [], []
    output = Path(output)
    object_hashes = {}
    provider_hashes = {}
    reference_hashes = {}
    output.mkdir(parents=True, exist_ok=True)
    # A failed compilation/reference check must not leave an old green summary.
    for filename in ("report.json", "report.txt"):
        (output / filename).unlink(missing_ok=True)
    module_inputs = manifest_hashes(manifest_path, modules)
    validate_counted_code(modules, versions)
    for version in versions:
        external_bindings = {}
        target_config = targets[version]
        reference, actual_hash = load_reference(target_config, version)
        reference_hashes[version] = actual_hash
        for module in modules:
            entries = [(kind, f) for key, kind in CODE_ROLES
                       for f in module.get(key, []) if version in f]
            if (not entries and not module.get("bindings", {}).get(version)
                    and not _common_definition_records(module).get(version)):
                continue
            coff = load_object(module, target_config)
            common_definitions, hashes = load_common_definitions(module, target_config, version)
            provider_hashes.update(hashes)
            for kind, function in entries:
                validate_native_role(coff, function["symbol"], kind)
            object_hashes[f"{version}/{module['name']}"] = hashlib.sha256(coff.data).hexdigest()
            bindings = _code_bindings(module.get("bindings", {}).get(version, {}),
                                      ordinary_entries(module), version)
            code_regions = region_context(coff, module, version, bindings, reference, common_definitions)
            validate_common_definitions(coff, bindings, reference, common_definitions,
                                        code_regions=code_regions)
            register_external_bindings(external_bindings, coff, bindings, version, module["name"])
            for name, provider in common_definitions.items():
                filename = module["common_definitions"][version][name]
                register_external_bindings(external_bindings, provider, bindings, version,
                                           f"{module['name']} provider {filename}")
            for kind, function in entries:
                target = function[version]
                row = {"version": version, "module": module["name"], "kind": kind,
                       "label": function["label"], "symbol": function["symbol"],
                       "original_va": target["va"], "body_size": target["size"]}
                relative = Path(version) / module["name"] / f"{int(target['va'], 16):08x}-{kind}.txt"
                destination = output / relative
                destination.unlink(missing_ok=True)
                section = None
                try:
                    if kind == "region":
                        section = coff.code_region(function["symbol"], function["entries"],
                                                   anchor_offset=function.get("anchor_offset", 0))
                        _, rebuilt, original, fixups = compare_code_region(
                            coff, function, version, bindings, reference,
                            common_definitions=common_definitions, code_regions=code_regions)
                        window = {"data": section["data"], "excluded_nop_padding": 0}
                        row["anchor_offset"] = function.get("anchor_offset", 0)
                        row["entries"] = [{**entry, "original_va": hex(int(target["va"], 16) + entry["offset"])}
                                          for entry in function["entries"]]
                    else:
                        section = coff.function(function["symbol"])
                        _, rebuilt, original, fixups = compare_function(
                            coff, function, version, bindings, reference,
                            common_definitions=common_definitions, code_regions=code_regions)
                        window = code_window(coff, section, int(target["va"], 16), len(original),
                                             bindings, reference,
                                             common_definitions=common_definitions, code_regions=code_regions)
                    metrics, review = compare_windows(function["label"], int(target["va"], 16),
                                                      original, window["data"], rebuilt)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(review)
                    row.update(metrics, checked_fixups=fixups,
                               excluded_nop_padding=window["excluded_nop_padding"],
                               review=str(relative))
                    if "inline_data_bytes" in window:
                        row.update(inline_data_bytes=window["inline_data_bytes"],
                                   inline_alignment_bytes=window["inline_alignment_bytes"],
                                   inline_data_symbols=window["inline_data_symbols"])
                except (ValueError, KeyError) as exc:
                    row["error"] = str(exc)
                    if (kind == "function" and section is not None
                            and isinstance(exc, BodyExtentError)):
                        try:
                            metrics, review = diagnostic_window(
                                coff, section, function, version, bindings, reference,
                                common_definitions=common_definitions, code_regions=code_regions)
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_text("Strict matcher error: " + str(exc) + "\n\n" + review)
                            row.update(metrics, review=str(relative))
                        except (ValueError, KeyError) as diagnostic_exc:
                            row["diagnostic_error"] = str(diagnostic_exc)
                rows.append(row)
            for name, binding in bindings.items():
                if "data_hex" not in binding and "data_size" not in binding:
                    continue
                row = {"version": version, "module": module["name"], "symbol": name}
                try:
                    actual, fixups = validate_data_binding(coff, name, bindings, reference,
                                                          common_definitions=common_definitions, code_regions=code_regions)
                    row.update(exact=True, size=len(actual), checked_fixups=fixups)
                    if coff.common_size(name) is not None:
                        row.update(storage="common", object_file_bytes=0)
                        if name in common_definitions:
                            _, provider_section = common_definitions[name].definition(name)
                            row.update(storage="common_overridden",
                                       provider_object=module["common_definitions"][version][name],
                                       provider_file_bytes=(0 if provider_section["flags"] & 0x80
                                                            else len(actual)))
                except (ValueError, KeyError) as exc:
                    row.update(exact=False, error=str(exc))
                data_rows.append(row)
    if not rows and not data_rows:
        raise ValueError("No configured code or data selected")
    result = {
        "schema": 1,
        "tool": {"name": "binary-comp", "capstone": metadata.version("capstone")},
        "scope": "Configured complete COFF functions, compiler labels and separately counted complete code regions",
        "semantics": {
            "raw_bytes": "All original bytes vs COFF body before fixups; no masks",
            "relocated_bytes": "All original bytes vs body after explicit checked COFF fixups; no masks",
            "diagnostic_only": "Rejected extents compare the COFF section, or its code prefix before validated inline data, including padding; the strict gate error remains a failure",
            "mnemonic_similarity": "binary-comp normalized linear mnemonic Levenshtein score; ignores all operands",
            "data": "Complete configured COFF definitions, including native inline tables and sectionless COMMON contributions with logical zeros or explicitly selected strong providers, with explicit fixups; COMMON checks one object's request, not final linker allocation; excluded from code metrics",
            "external_bindings": "Native storage-2 external address bindings agree by name within each selected target across selected modules and explicit COMMON providers; excludes static/local, weak, absolute and debug records; no claim about unselected objects or whole-link resolution",
            "regions": "Complete multi-entry contributions including padding; entries are metadata, not independent function counts",
            "whole_executable_equality": "Not tested: no rebuilt linked executable exists",
        },
        "inputs": {
            **module_inputs,
            "sources_sha256": source_hashes(modules, source_root),
            "analysis_sha256": reference_hashes,
            "objects_sha256": object_hashes,
            "provider_objects_sha256": provider_hashes,
        },
        "code": rows, "data": data_rows,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["binary-comp COFF-stage report; mnemonic similarity is not byte equality."]
    for row in rows:
        details = ["ERROR " + row["error"]] if "error" in row else []
        if "relocated_bytes" in row:
            similarity = row["mnemonic_similarity"]
            score = f"{similarity:.2f}%" if similarity is not None else "unavailable"
            if row.get("diagnostic_only"):
                extent = "code prefix before inline data" if "inline_data_bytes" in row else "complete COFF section"
                details.append(f"diagnostic {extent}={row['coff_window_size']} bytes "
                               "(any compiler padding included)")
            details.append(f"raw={row['raw_bytes']['exact']} relocated={row['relocated_bytes']['exact']} "
                           f"mnemonics={score}")
            if "metric_error" in row:
                details.append(row["metric_error"])
        if "diagnostic_error" in row:
            details.append("diagnostic unavailable: " + row["diagnostic_error"])
        detail = "; ".join(details)
        lines.append(f"{row['version']} {row['kind']} {row['original_va']} {row['label']}: {detail}")
    for kind in ("function", "label", "region"):
        selected = [r for r in rows if r["kind"] == kind]
        raw = sum("error" not in r and r.get("raw_bytes", {}).get("exact", False) for r in selected)
        exact = sum("error" not in r and r.get("relocated_bytes", {}).get("exact", False) for r in selected)
        scores = [r["mnemonic_similarity"] for r in selected if r.get("mnemonic_similarity") is not None]
        average = f"{sum(scores) / len(scores):.2f}% over {len(scores)} rows" if scores else "unavailable"
        lines.append(f"{kind}: {raw}/{len(selected)} raw-byte equal; {exact}/{len(selected)} "
                     f"relocated-byte equal; mean mnemonic similarity {average}.")
        accepted = [r for r in selected if "error" not in r
                    and r.get("relocated_bytes", {}).get("exact", False)]
        lines.append(f"  {sum(r['body_size'] for r in accepted)} exact bytes; "
                     f"{sum(r['checked_fixups'] for r in accepted)} checked code fixups.")
    for row in data_rows:
        if not row["exact"]:
            lines.append(f"FAIL DATA {row['version']} {row['symbol']}: {row['error']}")
    accepted_data = [r for r in data_rows if r["exact"]]
    lines.append(f"Data: {len(accepted_data)}/{len(data_rows)} complete definitions exact; "
                 f"{sum(r['size'] for r in accepted_data)} bytes; "
                 f"{sum(r['checked_fixups'] for r in accepted_data)} checked data fixups.")
    lines.append("Scope excludes unknown/protected code. Whole executable equality and behavior are not tested.")
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"JSON and per-body binary-comp reviews: {output}")
    return sum("error" in r or "metric_error" in r or not r.get("relocated_bytes", {}).get("exact", False)
               for r in rows) + sum(not r["exact"] for r in data_rows)
