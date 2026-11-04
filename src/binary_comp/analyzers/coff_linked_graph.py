"""Anchored, object-qualified placement proofs from actual i386 COFF edges.

Original addresses validate native contracts only. Actual addresses start at
exact same-owner PUBLIC MAP records; private candidates stay tentative until
every source in their anchored component replays its complete linked window.
"""

from dataclasses import dataclass, field
import hashlib
import struct

from binary_comp.analyzers.coff_report import (
    object_path, region_contracts, validate_native_role,
)
from binary_comp.analyzers.exe import linked_map_identity
from binary_comp.core.coff import COFF, code_window, compare_code_region, compare_function, is_function_type
from binary_comp.source.functions import object_member_name


@dataclass(frozen=True)
class GraphPlacement:
    address: int | None = None
    provenance: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class _AbsoluteTerminal:
    value: int | None = None
    provenance: str = ""
    detail: str = ""


def load_absolute_definitions(target):
    """Read explicitly selected absolute scalar definitions, never bindings.

    Provider failures stay per-terminal diagnostics. All configured provider
    objects participate in detecting conflicting definitions of selected names.
    No section bytes, allocation, address aliases or weak fallbacks are used.
    """
    objects = {}
    for _, filename in target.absolute_definitions:
        if filename in objects:
            continue
        try:
            path = object_path(target, filename)
            objects[filename] = (COFF(path), path, "")
        except (OSError, ValueError, RuntimeError) as error:
            objects[filename] = (None, None, str(error))

    def scalar(record):
        return (record["storage"] == 2 and record["section"] == -1
                and record["type"] == 0 and record["aux_count"] == 0
                and "weak_fallback" not in record)

    result = {}
    for name, filename in target.absolute_definitions:
        try:
            coff, path, failure = objects[filename]
            if failure:
                raise ValueError(failure)
            records = [(index, record) for index, record in coff.symbols.items() if record["name"] == name]
            if len(records) != 1 or not scalar(records[0][1]):
                raise ValueError("Provider needs exactly one strong native absolute type-0 scalar record")
            index, native = records[0]
            value = native["value"]
            for other, _, error in objects.values():
                if other is None:
                    continue
                matching = [record for record in other.symbols.values() if record["name"] == name]
                definitions = [record for record in matching if record["section"] != 0 or record["value"]]
                if definitions and (len(matching) != 1 or not scalar(definitions[0])
                                    or definitions[0]["value"] != value):
                    raise ValueError("Conflicting configured absolute provider definitions")
            result[name] = _AbsoluteTerminal(
                value, f"ABS provider {path} SHA256={hashlib.sha256(coff.data).hexdigest()} "
                f"native index={index} storage=2 section=-1 type=0 value={value:#010x}")
        except ValueError as error:
            result[name] = _AbsoluteTerminal(detail=f"Invalid absolute provider for {name}: {error}")
    return result


@dataclass
class _Node:
    name: str
    kind: str
    original: int
    section: int
    start: int
    raw: bytes
    relocations: list
    fixups: int
    zero_fill: bool = False
    # Region entry symbols identify offsets in one complete contribution.
    symbols: dict = field(default_factory=dict)


@dataclass
class _Closure:
    addresses: dict
    private: set = field(default_factory=set)
    provenance: list = field(default_factory=list)
    error: str = ""


def _map_candidates(symbols, name, owner, *, public=False):
    selected = [record for record in symbols if record.symbol == name
                and record.object_file is not None
                and (not public or record.visibility == "public")]
    owned = [record for record in selected
             if object_member_name(record.object_file).casefold() == owner.casefold()]
    candidates = owned if owned else selected if public else []
    return {linked_map_identity(record): record for record in candidates}


def resolve_private_placements(coff, module, version, bindings, reference, rebuilt,
                               symbols, validated_data, *, providers, regions,
                               read_window, absolute_definitions=None):
    """Return diagnostics for complete owned private contracts, never aliases.

    ``validated_data`` contains only data definitions already accepted by the
    strict native data path, as ``name: (size, checked_fixups)``. Native code is
    independently accepted by the existing complete function/region validators.
    Known PUBLIC targets are fixed terminals, not source ownership transfers.
    """
    owner = module["name"] + ".obj"
    nodes, by_symbol, failures = {}, {}, {}
    absolute_definitions = absolute_definitions or {}

    def add(node):
        if node.name in nodes:
            raise ValueError("Duplicate complete native graph owner")
        nodes[node.name] = node
        for name, delta in node.symbols.items():
            if name in by_symbol:
                raise ValueError("Overlapping complete native graph contracts")
            by_symbol[name] = node, delta

    for name, (size, fixups) in validated_data.items():
        definition, section = coff.definition(name)
        start, end = definition["value"], definition["value"] + size
        relocations = [(offset - start, index, kind)
                       for offset, index, kind in section["relocations"]
                       if offset < end and offset + 4 > start]
        add(_Node(name, "data", int(bindings[name]["va"], 16), definition["section"],
                  start, section["data"][start:end], relocations, fixups,
                  bool(section["flags"] & 0x80), {name: 0}))

    # Retained layout owners do not preserve the declared function/label role;
    # they cannot silently become graph sources based on their native storage.
    entries = [(entry, kind) for key, kind in (("functions", "function"), ("code_labels", "label"))
               for entry in module.get(key, [])]
    for entry, kind in entries:
        if version not in entry:
            continue
        name = entry["symbol"]
        try:
            native = coff.named[name]
            validate_native_role(coff, name, kind)
            exact, _, _, fixups = compare_function(
                coff, entry, version, bindings, reference,
                common_definitions=providers, code_regions=regions)
            if not exact:
                raise ValueError("Complete native code contract differs from original")
            address, size = int(entry[version]["va"], 16), entry[version]["size"]
            window = code_window(coff, coff.function(name), address, size, bindings,
                                 reference, common_definitions=providers, code_regions=regions)
            add(_Node(name, kind, address, native["section"], native["value"],
                      window["data"], window["relocations"], fixups, symbols={name: 0}))
        except (KeyError, ValueError) as error:
            failures[name] = str(error)

    for entry in region_contracts(module):
        if version not in entry:
            continue
        name = entry["symbol"]
        try:
            exact, _, _, fixups = compare_code_region(
                coff, entry, version, bindings, reference,
                common_definitions=providers, code_regions=regions)
            if not exact:
                raise ValueError("Complete native region contract differs from original")
            native, section = coff.definition(name)
            add(_Node(name, "region", int(entry[version]["va"], 16), native["section"],
                      0, section["data"], section["relocations"], fixups,
                      symbols={record["symbol"]: record["offset"] for record in entry["entries"]}))
        except (KeyError, ValueError) as error:
            for record in entry["entries"]:
                failures[record["symbol"]] = str(error)

    def unique_map(name, native):
        # Owned identities take precedence, as in the direct linked diagnostic.
        candidates = _map_candidates(symbols, name, owner)
        dlls = [record for record in symbols if record.symbol == name
                and record.visibility == "public" and record.dll_contributor is not None]
        if dlls and native["storage"] == 2:
            if (native["section"] != 0 or native["value"] or native["aux_count"]
                    or native["type"] != 0 and not is_function_type(native["type"])
                    or sum(record["name"] == name for record in coff.symbols.values()) != 1):
                raise ValueError(f"DLL terminal needs one strong unaliased undefined external: {name}")
            # DLL terminals have no local ownership precedence. Every qualified
            # PUBLIC object/DLL identity for this exact extern must be unique.
            public = [record for record in symbols if record.symbol == name and record.visibility == "public"]
            if any(record.object_file is None and record.dll_contributor is None for record in public):
                raise ValueError(f"Ambiguous actual MAP target with unqualified PUBLIC provenance: {name}")
            candidates = {linked_map_identity(record): record for record in public}
        if not candidates and native["storage"] == 2:
            candidates = _map_candidates(symbols, name, owner, public=True)
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous actual MAP target: {name}")
        return next(iter(candidates.values())) if candidates else None

    def native_target(index):
        # resolve_symbol intentionally supports weak/name fallbacks for native
        # contracts. They cannot establish an actual private graph identity.
        native = coff.symbols.get(index)
        if (native is None or native["storage"] not in (2, 3, 6)
                or "weak_fallback" in native
                or not 0 <= native["section"] <= len(coff.sections)):
            raise ValueError(f"No strong native relocation target at symbol index {index}")
        if native["section"] == 0:
            if native["storage"] != 2 or native["value"] or native["aux_count"]:
                raise ValueError(f"No defined private or external target: {native['name']}")
        elif coff.named.get(native["name"]) is not native:
            raise ValueError(f"Relocation does not name its actual native definition: {native['name']}")
        return native

    def source_bytes(node, address):
        if not 0 <= address < address + len(node.raw) <= 0x100000000:
            raise ValueError(f"Actual source window outside i386 address domain: {node.name}")
        backing, payload = read_window(rebuilt, address, len(node.raw))
        original_backing, _ = read_window(reference, node.original, len(node.raw))
        if backing != "file-backed" or original_backing != "file-backed":
            raise ValueError(f"Complete source window is not file-backed: {node.name}")
        if node.kind != "data" and (
                not rebuilt.section_for_va(address).flags & 0x20000000
                or not reference.section_for_va(node.original).flags & 0x20000000):
            raise ValueError(f"Complete code source is not executable: {node.name}")
        return payload

    def map_target_role(name, native, mapped, kind):
        # i386 import-pointer names carry function type in COFF, while the
        # qualified DLL MAP row identifies a scalar slot. This ABI prefix
        # classifies only the role; the address always comes from that MAP row.
        import_slot = (mapped.dll_contributor is not None and native["section"] == 0
                       and is_function_type(native["type"])
                       and name.startswith("__imp_") and len(name) > len("__imp_"))
        if import_slot:
            if kind != 6 or mapped.is_function or read_window(rebuilt, mapped.va, 4)[0] != "file-backed":
                raise ValueError(f"DLL import-slot role needs DIR32 and a file-backed non-function 4-byte terminal: {name}")
        if import_slot:
            expected = False
        elif name in by_symbol:
            expected = by_symbol[name][0].kind != "data"
        elif native["section"] > 0 or is_function_type(native["type"]):
            expected = is_function_type(native["type"])
        else:
            expected = None
        if expected is not None and mapped.is_function != expected:
            raise ValueError(f"Actual MAP target role differs: {name}")
        return import_slot

    def validate_windows(addresses):
        used = [(nodes[name], placement) for name, placement in addresses.items()]
        for position, (left, left_va) in enumerate(used):
            for right, right_va in used[position + 1:]:
                if left.section == right.section:
                    if (left.start < right.start + len(right.raw)
                            and right.start < left.start + len(left.raw)):
                        raise ValueError("Overlapping native source contracts in actual graph")
                    if left_va - left.start != right_va - right.start:
                        raise ValueError("Actual same-section placements differ from native offsets")
                if (left_va < right_va + len(right.raw)
                        and right_va < left_va + len(left.raw)):
                    raise ValueError("Overlapping actual graph contract windows")

    def private_dependencies(root):
        # Component membership must not depend on relocation table order: an
        # earlier failed actual field cannot hide a later shared private edge.
        # This walk supplies rejection dependencies only, never addresses/proof.
        pending, visited, private = [root], set(), set()
        while pending:
            node = pending.pop()
            if node.name in visited or node.zero_fill:
                continue
            visited.add(node.name)
            for offset, index, kind in node.relocations:
                if kind not in (6, 20) or not 0 <= offset <= len(node.raw) - 4:
                    continue
                try:
                    native = native_target(index)
                except ValueError:
                    continue
                name = native["name"]
                try:
                    mapped = unique_map(name, native)
                    if mapped is not None and native["storage"] == 2 and mapped.visibility == "public":
                        map_target_role(name, native, mapped, kind)
                except ValueError:
                    mapped = None
                if (native["storage"] == 2 and mapped is not None
                        and mapped.visibility == "public"):
                    continue
                if native["section"] == 0 or name not in by_symbol:
                    continue
                child, delta = by_symbol[name]
                if child.section == native["section"] and child.start + delta == native["value"]:
                    private.add(child.name)
                    pending.append(child)
        return private

    def follow(root, address, mapped):
        closure = _Closure({root.name: address}, private=private_dependencies(root))
        closure.provenance.append(
            f"PUBLIC MAP root {owner}:{mapped.symbol} at {mapped.va:#010x} "
            f"from {mapped.object_file}; complete {root.kind} window at {address:#010x}")
        pending, visited = [root], set()
        try:
            while pending:
                node = pending.pop()
                if node.name in visited:
                    continue
                visited.add(node.name)
                source = closure.addresses[node.name]
                # BSS may be a target, but has no raw object payload to replay.
                if node.zero_fill:
                    backing, _ = read_window(rebuilt, source, len(node.raw))
                    if backing == "invalid-window":
                        raise ValueError(f"Invalid complete actual BSS target window: {node.name}")
                    continue
                actual = source_bytes(node, source)
                replay, occupied = bytearray(node.raw), set()
                closure.provenance.append(
                    f"Source proof {owner}:{node.name}: complete native {node.kind} "
                    f"contract at {node.original:#010x}, size={len(node.raw)}, "
                    f"fixups={node.fixups}; actual full-window replay at {source:#010x}")
                for offset, index, kind in node.relocations:
                    if (kind not in (6, 20) or offset < 0 or offset + 4 > len(replay)
                            or occupied.intersection(range(offset, offset + 4))):
                        raise ValueError(f"Unsupported, overlapping or out-of-bounds actual relocation: {node.name}")
                    occupied.update(range(offset, offset + 4))
                    native = native_target(index)
                    name = native["name"]
                    addend = struct.unpack_from("<I", node.raw, offset)[0]
                    field = struct.unpack_from("<I", actual, offset)[0]
                    if kind == 20:
                        displacement = struct.unpack_from("<i", actual, offset)[0]
                        inferred = (source + offset + 4 + displacement - addend) & 0xFFFFFFFF
                    else:
                        inferred = (field - addend) & 0xFFFFFFFF
                    absolute = absolute_definitions.get(name)
                    target_map = unique_map(name, native) if absolute is None else None
                    if absolute is not None:
                        if absolute.detail:
                            raise ValueError(absolute.detail)
                        if (native["section"] != 0 or native["storage"] != 2
                                or native["type"] != 0 or native["value"] or native["aux_count"]
                                or sum(record["name"] == name for record in coff.symbols.values()) != 1):
                            raise ValueError(f"Absolute scalar needs one unaliased type-0 external reference: {name}")
                        if (inferred != absolute.value or any(
                                record.va != absolute.value or record.is_function
                                for record in symbols if record.symbol == name)):
                            raise ValueError(f"Actual relocation or MAP conflicts with absolute provider: {name}")
                        target = absolute.value
                        closure.provenance.append(
                            f"{owner}:{node.name}+{offset:#x} {'REL32' if kind == 20 else 'DIR32'} "
                            f"index={index} addend={addend:#x} -> ABS {name} at {target:#010x}; "
                            f"{absolute.provenance}")
                    elif target_map is not None:
                        if inferred != target_map.va:
                            raise ValueError(f"Actual relocation conflicts with MAP target: {name}")
                        import_slot = map_target_role(name, native, target_map, kind)
                        if target_map.visibility == "public" and native["storage"] == 2:
                            # Exact PUBLIC addresses are terminals. A foreign
                            # selected owner never authenticates a local source.
                            target = target_map.va
                            provenance = target_map.object_file or (
                                ("DLL import-slot " if import_slot else "DLL ") + target_map.dll_contributor)
                            closure.provenance.append(
                                f"{owner}:{node.name}+{offset:#x} {'REL32' if kind == 20 else 'DIR32'} "
                                f"index={index} addend={addend:#x} -> PUBLIC {name} "
                                f"at {target:#010x} from {provenance}")
                        else:
                            target = None
                    else:
                        target = None
                    if target is None:
                        if native["section"] == 0:
                            raise ValueError(f"External target needs a unique actual PUBLIC MAP identity: {name}")
                        if name not in by_symbol:
                            reason = failures.get(name, "No complete validated native contract")
                            raise ValueError(f"Cannot traverse private target {name}: {reason}")
                        child, delta = by_symbol[name]
                        if (child.section != native["section"]
                                or child.start + delta != native["value"]):
                            raise ValueError(f"Private target differs from its exact native contract: {name}")
                        target = inferred
                        base = target - delta
                        previous = closure.addresses.get(child.name)
                        if previous is not None and previous != base:
                            raise ValueError(f"Conflicting actual private paths: {name}")
                        closure.addresses[child.name] = base
                        closure.private.add(child.name)
                        pending.append(child)
                        closure.provenance.append(
                            f"{owner}:{node.name}+{offset:#x} {'REL32' if kind == 20 else 'DIR32'} "
                            f"index={index} addend={addend:#x} -> private {name} "
                            f"at {target:#010x}; complete {child.kind} owner {child.name}")
                    value = target + addend - (source + offset + 4 if kind == 20 else 0)
                    struct.pack_into("<I", replay, offset, value & 0xFFFFFFFF)
                if bytes(replay) != actual:
                    raise ValueError(f"Complete actual source replay differs: {node.name}")
            validate_windows(closure.addresses)
        except (KeyError, ValueError) as error:
            closure.error = str(error)
        return closure

    closures = []
    for node in nodes.values():
        if node.zero_fill:
            continue
        for name, delta in node.symbols.items():
            native = coff.named[name]
            if native["storage"] != 2:
                continue
            candidates = _map_candidates(symbols, name, owner)
            if len(candidates) != 1:
                continue
            mapped = next(iter(candidates.values()))
            if (mapped.visibility != "public"
                    or mapped.is_function != (node.kind != "data")):
                continue
            closures.append(follow(node, mapped.va - delta, mapped))

    # Union only owned native private dependencies. Fixed PUBLIC target
    # addresses are terminals, so unrelated roots sharing an extern stay apart.
    components = []
    for closure in closures:
        touching = [group for group in components
                    if any(closure.private & other.private for other in group)]
        group = [closure]
        for previous in touching:
            components.remove(previous)
            group.extend(previous)
        components.append(group)
    results = {}
    for group in components:
        candidates, error = {}, next((item.error for item in group if item.error), "")
        private = set().union(*(item.private for item in group))
        provenance = tuple(dict.fromkeys(line for item in group for line in item.provenance))
        for closure in group:
            for name in closure.private:
                if name not in closure.addresses:
                    continue  # A failed source supplies rejection dependency only.
                address = closure.addresses[name]
                if name in candidates and candidates[name] != address:
                    error = f"Conflicting actual addresses from PUBLIC MAP roots: {name}"
                candidates[name] = address
        if not error:
            addresses = {name: address for closure in group for name, address in closure.addresses.items()}
            try:
                validate_windows(addresses)
            except ValueError as failure:
                error = str(failure)
        for name in private:
            node = nodes[name]
            for symbol, delta in node.symbols.items():
                results[symbol] = GraphPlacement(
                    None if error else candidates[name] + delta,
                    () if error else provenance, error)
    return results
