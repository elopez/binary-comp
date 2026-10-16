"""Function annotations in assembly sources.

Hand-written assembly carries the same annotation as C, written as an
assembler comment directly before the procedure it marks:

    ; Function start: 0x00439C40
    ; Comment lines may follow.
    CopyRasterDriverName proc

A code label (``Name:``, ``Name::`` or ``Name label near``) may stand in for
the ``proc`` line, and ``align``/``even`` directives may sit between the two.
The marker names the procedure as written; decoration added by ``.model
flat, c`` or ``stdcall`` is left to map matching, as it is for C.
"""

from __future__ import annotations

import re

from .cpp import SourceCommentMarker, SourceFunctionMarker, parse_function_start_comment

ASSEMBLY_EXTENSIONS = (".asm",)

IDENTIFIER = r"[A-Za-z_?@$][\w?@$]*"
PROC_RE = re.compile(rf"^\s*({IDENTIFIER})\s+proc\b", re.IGNORECASE)
LABEL_RE = re.compile(rf"^\s*({IDENTIFIER})\s*(?:::?|\s+label\s+(?:near|far|proc)\b)\s*$", re.IGNORECASE)
TRANSPARENT_RE = re.compile(r"^\s*(?:align\b|even\s*$)", re.IGNORECASE)


def is_assembly_source(path: str) -> bool:
    return path.lower().endswith(ASSEMBLY_EXTENSIONS)


def split_comment(line: str) -> tuple[str, str]:
    """Split a line into code and comment, ignoring semicolons in quotes."""
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == ";":
            return line[:index], line[index + 1:]
    return line, ""


def parse_assembly_function_markers(path: str, include_no_assembly: bool = False) -> list[SourceFunctionMarker]:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        lines = handle.read().splitlines()

    markers: list[SourceFunctionMarker] = []
    pending: list[tuple[str, int]] = []
    for number, line in enumerate(lines, 1):
        code, comment = split_comment(line)
        if not code.strip():
            address = parse_function_start_comment(comment, line, include_no_assembly) if comment else None
            if address is not None:
                pending.append((address, number))
            continue
        if TRANSPARENT_RE.match(code):
            continue
        match = PROC_RE.match(code) or LABEL_RE.match(code)
        if match is not None:
            markers.extend(
                SourceFunctionMarker(address=address, name=match.group(1), line=marker_line, function_start=number)
                for address, marker_line in pending
            )
        pending = []
    return markers


def parse_assembly_function_comments(path: str, include_no_assembly: bool = False) -> list[SourceCommentMarker]:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        lines = handle.read().splitlines()

    markers: list[SourceCommentMarker] = []
    for number, line in enumerate(lines, 1):
        code, comment = split_comment(line)
        if code.strip() or not comment:
            continue
        address = parse_function_start_comment(comment, line, include_no_assembly)
        if address is not None:
            markers.append(SourceCommentMarker(address=address, line=number))
    return markers
