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
from binary_comp.core.coff import (COFF, BodyExtentError, apply_fixups, code_window,
                                   compare_function, is_function_type, validate_data_binding)
from binary_comp.core.pe import FileBackedPEImage

PE = FileBackedPEImage


def load_object(module, target):
    """Load one caller-built object; compilation remains outside this backend."""
    return COFF(Path(target_value(target, "object_dir")) / (module["name"] + ".obj"))


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
        result[entry["symbol"]] = dict(entry[version])
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
        target_value(target, "object_dir")
        digest = target_value(target, "original_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ConfigError(f"Target {name} needs an original_sha256 digest")
        targets[name] = target
    section = config.get("coff")
    if not isinstance(section, dict) or not isinstance(section.get("modules"), str):
        raise ConfigError("coff.modules must name a module manifest")
    manifest = config_path.parent / section["modules"]
    modules = json.loads(manifest.read_text())
    if not isinstance(modules, list):
        raise ConfigError("COFF module manifest must be a list")
    seen = set()
    for module in modules:
        if not isinstance(module, dict):
            raise ConfigError("COFF modules must be objects")
        name = _component(module.get("name"), "COFF module name")
        if name in seen:
            raise ConfigError(f"Duplicate COFF module: {name}")
        seen.add(name)
        if "source" in module and not isinstance(module["source"], str):
            raise ConfigError(f"{name}.source must be a path label")
        for key in ("functions", "code_labels"):
            entries = module.get(key, [])
            if not isinstance(entries, list):
                raise ConfigError(f"{name}.{key} must be a list")
            for entry in entries:
                if (not isinstance(entry, dict) or not isinstance(entry.get("label"), str)
                        or not isinstance(entry.get("symbol"), str)):
                    raise ConfigError(f"Invalid COFF code entry in {name}.{key}")
                for target_name in targets:
                    if target_name not in entry:
                        continue
                    binding = entry[target_name]
                    if (not isinstance(binding, dict) or type(binding.get("size")) is not int
                            or binding["size"] <= 0 or not isinstance(binding.get("va"), str)):
                        raise ConfigError(f"Invalid complete COFF extent for {entry['label']}")
                    int(binding["va"], 16)
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
                           [entry for key in ("functions", "code_labels")
                            for entry in module.get(key, [])], target_name)
    if module_names:
        if set(module_names) - seen:
            raise ConfigError("Unknown COFF module: " + ", ".join(sorted(set(module_names) - seen)))
        modules = [m for m in modules if m["name"] in module_names]
    if function_name is not None:
        selected = []
        for module in modules:
            entries = [(key, f) for key in ("functions", "code_labels")
                       for f in module.get(key, [])
                       if function_name in (f["label"], f["symbol"])
                       and any(name in f for name in targets)]
            if not entries:
                continue
            narrowed = dict(module)
            # All function addresses remain available to code and data fixups.
            narrowed["bindings"] = {
                name: _code_bindings(module.get("bindings", {}).get(name, {}),
                                     [f for key in ("functions", "code_labels")
                                      for f in module.get(key, [])], name)
                for name in targets}
            for key in ("functions", "code_labels"):
                narrowed[key] = [f for role, f in entries if role == key]
            selected.append(narrowed)
        modules = selected
        if not modules:
            raise ConfigError(f"COFF function not found: {function_name}")
    return modules, targets, manifest


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
                   or any(file_filter in f["label"] for key in ("functions", "code_labels")
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


def diagnostic_bytes(coff, section, function, version, bindings, reference):
    """Relocate a rejected complete code window without disassembly or report output."""
    target = function[version]
    address, size = int(target["va"], 16), target["size"]
    original = reference.read(address, size)
    window = code_window(coff, section, address, size, bindings, reference, diagnostic=True)
    bindings = window["bindings"]
    # Validate even unreferenced configured data before publishing diagnostics.
    # A length failure in the strict gate occurs before its relocation checks.
    for name, binding in bindings.items():
        if "data_hex" in binding or "data_size" in binding:
            validate_data_binding(coff, name, bindings, reference)
    raw = window["data"]
    rebuilt, fixups = apply_fixups(coff, raw, window["relocations"], address,
                                  bindings, reference)
    return window, original, rebuilt, fixups


def diagnostic_window(coff, section, function, version, bindings, reference):
    """Review a rejected code extent, keeping explicitly validated native data separate."""
    window, original, rebuilt, fixups = diagnostic_bytes(
        coff, section, function, version, bindings, reference)
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
    if len(functions) != 1 or labels:
        raise ConfigError("COFF scoring requires exactly one native function")
    validate_counted_code(modules, [target_name])
    reference, _ = load_reference(targets[target_name], target_name)
    coff = COFF(object_path)
    function = functions[0]
    validate_native_role(coff, function["symbol"], "function")
    bindings = _code_bindings(module.get("bindings", {}).get(target_name, {}),
                              functions, target_name)
    try:
        # Unreferenced configured definitions must pass the same data gate too.
        for name, binding in bindings.items():
            if "data_hex" in binding or "data_size" in binding:
                validate_data_binding(coff, name, bindings, reference)
        try:
            _, rebuilt, original, _ = compare_function(
                coff, function, target_name, bindings, reference)
        except BodyExtentError:
            _, original, rebuilt, _ = diagnostic_bytes(
                coff, coff.function(function["symbol"]), function, target_name, bindings, reference)
            return max(1, compare_binary(original, rebuilt).differing_positions)
    except KeyError as error:
        raise ValueError(f"Missing configured COFF symbol: {error.args[0]}") from error
    return compare_binary(original, rebuilt).differing_positions


def validate_counted_code(modules, versions):
    """Count each original code window once; aliases belong in bindings."""
    for version in versions:
        symbols = set()
        windows = []
        for module in modules:
            for key in ("functions", "code_labels"):
                for entry in module.get(key, []):
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
                    windows.append((start, start + size, module["name"], symbol))
        windows.sort()
        for left, right in zip(windows, windows[1:]):
            if left[1] > right[0]:
                raise ValueError(f"Overlapping original COFF code windows for {version}: "
                                 f"{left[2]}:{left[3]} and {right[2]}:{right[3]}")


def validate_native_role(coff, symbol, kind):
    """Use native symbol records, never disassembly, to classify counted code."""
    try:
        definition, _ = coff.definition(symbol)
    except KeyError as error:
        raise ValueError(f"Counted COFF symbol is not defined: {symbol}") from error
    if kind == "function":
        if not is_function_type(definition["type"]):
            raise ValueError(f"Counted COFF function needs native function type: {symbol}")
    else:
        if definition["storage"] != 6 or definition["type"] != 0:
            raise ValueError(f"Counted COFF code label needs native storage 6/type 0: {symbol}")
        # The core also excludes local case labels inside a /Gy function section.
        coff.function(symbol)


def report(modules, targets, versions, output, *, manifest_path=None, source_root=None):
    rows, data_rows = [], []
    output = Path(output)
    object_hashes = {}
    reference_hashes = {}
    output.mkdir(parents=True, exist_ok=True)
    # A failed compilation/reference check must not leave an old green summary.
    for filename in ("report.json", "report.txt"):
        (output / filename).unlink(missing_ok=True)
    validate_counted_code(modules, versions)
    for version in versions:
        target_config = targets[version]
        reference, actual_hash = load_reference(target_config, version)
        reference_hashes[version] = actual_hash
        for module in modules:
            entries = [(kind, f) for key, kind in (("functions", "function"), ("code_labels", "label"))
                       for f in module.get(key, []) if version in f]
            if not entries and not module.get("bindings", {}).get(version):
                continue
            coff = load_object(module, target_config)
            for kind, function in entries:
                validate_native_role(coff, function["symbol"], kind)
            object_hashes[f"{version}/{module['name']}"] = hashlib.sha256(coff.data).hexdigest()
            bindings = _code_bindings(module.get("bindings", {}).get(version, {}),
                                      [f for _, f in entries], version)
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
                    section = coff.function(function["symbol"])
                    _, rebuilt, original, fixups = compare_function(
                        coff, function, version, bindings, reference)
                    window = code_window(coff, section, int(target["va"], 16), len(original),
                                         bindings, reference)
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
                                coff, section, function, version, bindings, reference)
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
                    actual, fixups = validate_data_binding(coff, name, bindings, reference)
                    row.update(exact=True, size=len(actual), checked_fixups=fixups)
                    if coff.common_size(name) is not None:
                        row.update(storage="common", object_file_bytes=0)
                except (ValueError, KeyError) as exc:
                    row.update(exact=False, error=str(exc))
                data_rows.append(row)
    if not rows and not data_rows:
        raise ValueError("No configured code or data selected")
    result = {
        "schema": 1,
        "tool": {"name": "binary-comp", "capstone": metadata.version("capstone")},
        "scope": "Configured complete COFF function bodies and separate compiler code labels only",
        "semantics": {
            "raw_bytes": "All original bytes vs COFF body before fixups; no masks",
            "relocated_bytes": "All original bytes vs body after explicit checked COFF fixups; no masks",
            "diagnostic_only": "Rejected extents compare the COFF section, or its code prefix before validated inline data, including padding; the strict gate error remains a failure",
            "mnemonic_similarity": "binary-comp normalized linear mnemonic Levenshtein score; ignores all operands",
            "data": "Complete configured COFF definitions, including native inline tables and sectionless zero-initialized COMMON contributions, with explicit fixups; COMMON checks one object's request, not final linker allocation; excluded from code metrics",
            "whole_executable_equality": "Not tested: no rebuilt linked executable exists",
        },
        "inputs": {
            "modules_sha256": hashlib.sha256(Path(manifest_path).read_bytes() if manifest_path
                                              else json.dumps(modules, sort_keys=True).encode()).hexdigest(),
            "sources_sha256": source_hashes(modules, source_root),
            "analysis_sha256": reference_hashes,
            "objects_sha256": object_hashes,
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
    for kind in ("function", "label"):
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
