"""Tree-sitter backed C++ source inventory."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from string import hexdigits

from .preprocessor import blank_inactive_conditionals


CALLING_CONVENTIONS = (
    b"__cdecl",
    b"__fastcall",
    b"__stdcall",
    b"__thiscall",
    b"__pascal",
    b"__far",
    b"__near",
    b"CALLBACK",
    b"WINAPI",
)

ASM_EXCLUSION_MARKER_RE = re.compile(r"/\*\s*ASM\s*\*/")
STUB_EXCLUSION_MARKER_RE = re.compile(r"/\*\s*STUB\s*\*/")
INLINE_ASSEMBLY_RE = re.compile(rb"(?<![A-Za-z0-9_])__?asm(?![A-Za-z0-9_])")
COMMENT_OR_LITERAL_RE = re.compile(
    rb"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.S,
)
BLANKABLE_RE = re.compile(rb"[^\r\n]")


@dataclass(frozen=True)
class SourceFunction:
    address: str
    name: str
    line: int


@dataclass(frozen=True)
class SourceFunctionGroup:
    addresses: tuple[str, ...]
    name: str
    line: int


@dataclass(frozen=True)
class SourceFunctionMarker:
    address: str
    name: str
    line: int
    function_start: int


@dataclass(frozen=True)
class SourceCommentMarker:
    address: str
    line: int


def make_cpp_parser():
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_cpp
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "tree-sitter C++ support is required. Install with: "
            "python3 -m pip install binary-comp[cpp]"
        ) from exc

    parser = Parser()
    language = Language(tree_sitter_cpp.language())
    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)
    return parser


def node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def sanitize_source(source: bytes) -> bytes:
    sanitized = bytearray(source)
    for keyword in CALLING_CONVENTIONS:
        start = 0
        while True:
            index = source.find(keyword, start)
            if index < 0:
                break
            sanitized[index:index + len(keyword)] = b" " * len(keyword)
            start = index + len(keyword)
    return blank_inline_assembly(blank_inactive_conditionals(bytes(sanitized)))


def blank_inline_assembly(source: bytes) -> bytes:
    """Blank MSVC ``__asm`` blocks and statements, which tree-sitter cannot parse.

    One unparsable block turns the rest of the file into an error node and
    drops every function whose body is inline assembly.  Only the assembly is
    blanked, keeping offsets and line breaks; the definition around it stays.
    """
    if b"asm" not in source:
        return source
    skipped = [(match.start(), match.end()) for match in COMMENT_OR_LITERAL_RE.finditer(source)]
    skipped_starts = [start for start, _ in skipped]
    blanked = bytearray(source)
    resume = 0
    for match in INLINE_ASSEMBLY_RE.finditer(source):
        start = match.start()
        if start < resume:
            continue
        index = bisect_right(skipped_starts, start) - 1
        if index >= 0 and skipped[index][0] <= start < skipped[index][1]:
            continue
        end = inline_assembly_end(source, match.end())
        blanked[start:end] = BLANKABLE_RE.sub(b" ", source[start:end])
        resume = end
    return bytes(blanked)


def inline_assembly_end(source: bytes, keyword_end: int) -> int:
    """Return the end of the ``__asm`` block or statement after ``keyword_end``."""
    cursor = keyword_end
    while cursor < len(source) and source[cursor] in b" \t\r\n":
        cursor += 1
    if source.startswith(b"{", cursor):
        end = scan_inline_assembly(source, cursor + 1, block=True)
        return keyword_end if end is None else end
    end = scan_inline_assembly(source, keyword_end, block=False)
    return len(source) if end is None else end


def scan_inline_assembly(source: bytes, cursor: int, block: bool) -> int | None:
    # A block ends after its closing brace; a statement ends at the line break
    # or before a closing brace of the C code around it.  Assembly comments run
    # to the end of the line and may contain either.
    while cursor < len(source):
        if source.startswith(b"/*", cursor):
            close = source.find(b"*/", cursor + 2)
            if close < 0:
                return None
            cursor = close + 2
            continue
        if source.startswith(b";", cursor) or source.startswith(b"//", cursor):
            newline = source.find(b"\n", cursor)
            if newline < 0:
                return None
            if not block:
                return newline
            cursor = newline
            continue
        if source.startswith(b"}", cursor):
            return cursor + 1 if block else cursor
        if source.startswith(b"\n", cursor) and not block:
            return cursor
        cursor += 1
    return None


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def parse_function_start_comment(text: str, line_text: str, include_no_assembly: bool = False) -> str | None:
    if not include_no_assembly and (
        "No assembly extracted" in line_text
        or ASM_EXCLUSION_MARKER_RE.search(line_text) is not None
        or STUB_EXCLUSION_MARKER_RE.search(line_text) is not None
    ):
        return None
    # A scalar/vector deleting destructor (sdtor) is a compiler-generated COMDAT,
    # not part of the source function. When the source documents its address
    # alongside the real destructor (e.g. "/* Foo sdtor (COMDAT,
    # compiler-generated) */"), exclude it so it is not combined with — or
    # mistaken for — the real destructor body.
    if "compiler-generated" in line_text.lower():
        return None
    if "Function start:" not in text:
        return None

    start = text.find("0x")
    if start < 0:
        return None

    address = []
    for ch in text[start + 2:]:
        if ch in hexdigits:
            address.append(ch.upper())
        elif address:
            break

    if not address:
        return None
    return "".join(address)


def find_function_declarator(node):
    if node.type == "function_declarator":
        return node
    for child in node.children:
        found = find_function_declarator(child)
        if found is not None:
            return found
    return None


def function_name_from_definition(source: bytes, function_node) -> str | None:
    declarator = function_node.child_by_field_name("declarator")
    if declarator is None:
        declarator = function_node

    function_declarator = find_function_declarator(declarator)
    if function_declarator is None:
        return None

    name_node = function_declarator.child_by_field_name("declarator")
    if name_node is None:
        return None

    return node_text(source, name_node).strip()


def parameter_list_from_definition(function_node):
    declarator = function_node.child_by_field_name("declarator")
    if declarator is None:
        declarator = function_node

    function_declarator = find_function_declarator(declarator)
    if function_declarator is None:
        return None

    parameter_list = function_declarator.child_by_field_name("parameters")
    if parameter_list is not None:
        return parameter_list
    for child in function_declarator.children:
        if child.type == "parameter_list":
            return child
    return None


def parameter_type_name(source: bytes, parameter_node) -> str | None:
    text = node_text(source, parameter_node).strip()
    if text == "void":
        return None

    type_name = None
    for child in walk(parameter_node):
        if child.type in ("primitive_type", "qualified_identifier", "scoped_type_identifier", "type_identifier"):
            type_name = node_text(source, child).strip()
            break
    if not type_name:
        return None

    if "*" in text:
        type_name += "*"
    elif "&" in text:
        type_name += "&"
    return type_name


def parameter_signature(source: bytes, function_node) -> str | None:
    parameter_list = parameter_list_from_definition(function_node)
    if parameter_list is None:
        return None

    parameters = []
    for child in parameter_list.children:
        if child.type != "parameter_declaration":
            continue
        type_name = parameter_type_name(source, child)
        if type_name is not None:
            parameters.append(type_name)
    return ",".join(parameters)


def function_name_with_optional_signature(
    source: bytes,
    function_node,
    signature_names: frozenset[str],
) -> str | None:
    name = function_name_from_definition(source, function_node)
    if name is None or name not in signature_names:
        return name

    signature = parameter_signature(source, function_node)
    if signature is None:
        return name
    return f"{name}({signature})"


def effective_function_start(function_node) -> int:
    parent = function_node.parent
    if parent is not None and parent.type == "linkage_specification":
        body = parent.child_by_field_name("body")
        if (
            body is not None
            and body.type == function_node.type
            and body.start_byte == function_node.start_byte
            and body.end_byte == function_node.end_byte
        ):
            return parent.start_byte
    return function_node.start_byte


def gap_is_only_comments_or_whitespace(
    source: bytes,
    start: int,
    end: int,
    transparent_spans: list[tuple[int, int]],
    span_starts: list[int],
) -> bool:
    current = start

    while current < end:
        span_idx = bisect_right(span_starts, current) - 1
        if span_idx >= 0:
            span_start, span_end = transparent_spans[span_idx]
            if span_start <= current < span_end:
                current = span_end
                continue

        next_span_idx = span_idx + 1
        if next_span_idx < len(transparent_spans) and transparent_spans[next_span_idx][0] == current:
            current = transparent_spans[next_span_idx][1]
            continue

        if chr(source[current]).isspace():
            current += 1
            continue

        return False

    return True


def parse_source_function_markers(
    path: str,
    include_no_assembly: bool = False,
    signature_names: frozenset[str] = frozenset(),
) -> list[SourceFunctionMarker]:
    with open(path, "rb") as f:
        source = f.read()

    sanitized = sanitize_source(source)
    tree = make_cpp_parser().parse(sanitized)
    lines = source.splitlines()
    transparent_spans: list[tuple[int, int]] = []
    comments: list[tuple[int, int, int, str]] = []
    functions: list[tuple[int, str]] = []

    for node in walk(tree.root_node):
        if node.type == "comment":
            transparent_spans.append((node.start_byte, node.end_byte))
            line = node.start_point.row
            line_text = lines[line].decode("utf-8", errors="ignore") if line < len(lines) else ""
            address = parse_function_start_comment(node_text(source, node), line_text, include_no_assembly)
            if address is not None:
                comments.append((node.start_byte, node.end_byte, node.start_point.row + 1, address))
        elif node.type in ("preproc_call", "preproc_def", "preproc_function_def",
                           "preproc_include", "preproc_if", "preproc_ifdef",
                           "preproc_else", "preproc_elif"):
            transparent_spans.append((node.start_byte, node.end_byte))
        elif node.type == "function_definition":
            name = function_name_with_optional_signature(source, node, signature_names)
            if name is not None:
                functions.append((effective_function_start(node), name))

    comments.sort()
    functions.sort()
    transparent_spans.sort()
    function_starts = [start for start, _ in functions]
    transparent_starts = [start for start, _ in transparent_spans]
    markers: list[SourceFunctionMarker] = []

    for _, comment_end, line, address in comments:
        function_idx = bisect_right(function_starts, comment_end - 1)
        if function_idx >= len(functions):
            continue

        function_start, name = functions[function_idx]
        # Measure the gap in the parsed text: an arm blanked by the
        # preprocessor declaration separates nothing the compiler sees.
        if gap_is_only_comments_or_whitespace(sanitized, comment_end, function_start, transparent_spans, transparent_starts):
            markers.append(SourceFunctionMarker(address=address, name=name, line=line, function_start=function_start))

    return markers


def parse_source_function_comments(path: str, include_no_assembly: bool = False) -> list[SourceCommentMarker]:
    with open(path, "rb") as f:
        source = f.read()

    tree = make_cpp_parser().parse(sanitize_source(source))
    lines = source.splitlines()
    markers: list[SourceCommentMarker] = []

    for node in walk(tree.root_node):
        if node.type != "comment":
            continue
        line = node.start_point.row
        line_text = lines[line].decode("utf-8", errors="ignore") if line < len(lines) else ""
        address = parse_function_start_comment(node_text(source, node), line_text, include_no_assembly)
        if address is not None:
            markers.append(SourceCommentMarker(address=address, line=line + 1))

    return markers


def parse_source_functions(
    path: str,
    include_no_assembly: bool = False,
    signature_names: frozenset[str] = frozenset(),
) -> list[SourceFunction]:
    return [
        SourceFunction(address=marker.address, name=marker.name, line=marker.line)
        for marker in parse_source_function_markers(path, include_no_assembly, signature_names)
    ]


def parse_source_function_groups(
    path: str,
    include_no_assembly: bool = False,
    signature_names: frozenset[str] = frozenset(),
) -> list[SourceFunctionGroup]:
    groups: list[SourceFunctionGroup] = []
    current = None

    for marker in parse_source_function_markers(path, include_no_assembly, signature_names):
        group_key = (marker.function_start, marker.name)
        if current is None or current["key"] != group_key:
            if current is not None:
                groups.append(SourceFunctionGroup(
                    addresses=tuple(current["addresses"]),
                    name=current["name"],
                    line=current["line"],
                ))
            current = {
                "key": group_key,
                "addresses": [],
                "name": marker.name,
                "line": marker.line,
            }
        current["addresses"].append(marker.address)

    if current is not None:
        groups.append(SourceFunctionGroup(
            addresses=tuple(current["addresses"]),
            name=current["name"],
            line=current["line"],
        ))

    return groups
