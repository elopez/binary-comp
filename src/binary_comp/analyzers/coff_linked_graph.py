"""Anchored, object-qualified placement proofs from actual i386 COFF edges.

Original addresses validate native contracts only. Actual addresses start at
exact same-owner PUBLIC MAP records; private candidates stay tentative until
every source in their anchored component replays its complete linked window.
"""

from dataclasses import dataclass, field
import struct

from binary_comp.analyzers.coff_report import (
    region_contracts, validate_native_role,
)
from binary_comp.analyzers.exe import linked_map_identity
from binary_comp.core.coff import code_window, compare_code_region, compare_function, is_function_type
from binary_comp.source.functions import object_member_name


@dataclass(frozen=True)
class GraphPlacement:
    address: int | None = None
    provenance: tuple[str, ...] = ()
    detail: str = ""


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
                               read_window):
    """Return diagnostics for complete owned private contracts, never aliases.

    ``validated_data`` contains only data definitions already accepted by the
    strict native data path, as ``name: (size, checked_fixups)``. Native code is
    independently accepted by the existing complete function/region validators.
    Known PUBLIC targets are fixed terminals, not source ownership transfers.
    """
    owner = module["name"] + ".obj"
    nodes, by_symbol, failures = {}, {}, {}

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
                    target_map = unique_map(name, native)
                    if target_map is not None:
                        if inferred != target_map.va:
                            raise ValueError(f"Actual relocation conflicts with MAP target: {name}")
                        expected_function = (by_symbol[name][0].kind != "data" if name in by_symbol
                                             else is_function_type(native["type"])
                                             if native["section"] > 0 or is_function_type(native["type"]) else None)
                        if expected_function is not None and target_map.is_function != expected_function:
                            raise ValueError(f"Actual MAP target role differs: {name}")
                        if target_map.visibility == "public" and native["storage"] == 2:
                            # Exact PUBLIC addresses are terminals. A foreign
                            # selected owner never authenticates a local source.
                            target = target_map.va
                            closure.provenance.append(
                                f"{owner}:{node.name}+{offset:#x} {'REL32' if kind == 20 else 'DIR32'} "
                                f"index={index} addend={addend:#x} -> PUBLIC {name} "
                                f"at {target:#010x} from {target_map.object_file}")
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
