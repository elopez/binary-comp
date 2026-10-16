"""Minimal PE reader used by analyzers.

The project avoids a heavy PE dependency for now because the existing tools only
need section metadata, VA reads, and simple string extraction. BSS reads return
zero-filled bytes, matching the loader view of the image.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable


EXECUTABLE_FLAG = 0x20000000
READABLE_FLAG = 0x40000000


@dataclass(frozen=True)
class Section:
    name: str
    start: int
    end: int
    rawptr: int
    rawsize: int
    virtual_size: int
    flags: int

    @property
    def size(self) -> int:
        return self.end - self.start


class PEImage:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.data = f.read()

        if self.data[:2] != b"MZ":
            raise ValueError(f"not an MZ executable: {path}")
        peoff = struct.unpack_from("<I", self.data, 0x3C)[0]
        if self.data[peoff:peoff + 4] != b"PE\0\0":
            raise ValueError(f"not a PE executable: {path}")

        section_count = struct.unpack_from("<H", self.data, peoff + 6)[0]
        optional_size = struct.unpack_from("<H", self.data, peoff + 20)[0]
        optional_header = peoff + 24
        entry_point_rva = struct.unpack_from("<I", self.data, optional_header + 16)[0]
        self.image_base = struct.unpack_from("<I", self.data, optional_header + 28)[0]
        self.entry_point = self.image_base + entry_point_rva

        # Data directory 5 holds the base relocations (PE32 directories start
        # 96 bytes into the optional header).
        self._reloc_directory = (0, 0)
        if optional_size >= 96 + 6 * 8:
            self._reloc_directory = struct.unpack_from("<II", self.data, optional_header + 96 + 5 * 8)
        self._relocations: frozenset[int] | None = None

        self.sections: list[Section] = []
        section_table = optional_header + optional_size
        for index in range(section_count):
            offset = section_table + index * 40
            name = self.data[offset:offset + 8].split(b"\0", 1)[0].decode("ascii", errors="ignore")
            virtual_size, rva, raw_size, raw_pointer = struct.unpack_from("<IIII", self.data, offset + 8)
            flags = struct.unpack_from("<I", self.data, offset + 36)[0]
            start = self.image_base + rva
            end = start + max(virtual_size, raw_size)
            self.sections.append(Section(name, start, end, raw_pointer, raw_size, virtual_size, flags))

    def relocations(self) -> frozenset[int]:
        """VAs of the dwords that IMAGE_REL_BASED_HIGHLOW relocations patch."""
        if self._relocations is None:
            found: set[int] = set()
            rva, size = self._reloc_directory
            table = self.read(self.image_base + rva, size) if rva and size else None
            cursor = 0
            while table and cursor + 8 <= len(table):
                page, block_size = struct.unpack_from("<II", table, cursor)
                if block_size < 8:
                    break
                for entry_at in range(cursor + 8, min(cursor + block_size, len(table)) - 1, 2):
                    entry = struct.unpack_from("<H", table, entry_at)[0]
                    if entry >> 12 == 3:
                        found.add(self.image_base + page + (entry & 0xFFF))
                cursor += block_size
            self._relocations = frozenset(found)
        return self._relocations

    def section_for_va(self, va: int) -> Section | None:
        for section in self.sections:
            if section.start <= va < section.end:
                return section
        return None

    def section_named(self, name: str) -> Section | None:
        wanted = name.lower()
        for section in self.sections:
            if section.name.lower() == wanted:
                return section
        return None

    def section_end_for_va(self, va: int) -> int | None:
        section = self.section_for_va(va)
        return section.end if section else None

    def read(self, va: int, size: int) -> bytes | None:
        result = bytearray()
        cursor = va
        remaining = size

        while remaining > 0:
            section = self.section_for_va(cursor)
            if section is None:
                return None

            offset = cursor - section.start
            chunk_size = min(remaining, section.end - cursor)
            if offset >= section.rawsize:
                result.extend(b"\0" * chunk_size)
            else:
                available = min(chunk_size, section.rawsize - offset)
                raw_offset = section.rawptr + offset
                result.extend(self.data[raw_offset:raw_offset + available])
                if available < chunk_size:
                    result.extend(b"\0" * (chunk_size - available))

            cursor += chunk_size
            remaining -= chunk_size

        return bytes(result)

    def c_string_at(
        self,
        va: int,
        max_len: int = 160,
        predicate: Callable[[str], bool] | None = None,
    ) -> str | None:
        section = self.section_for_va(va)
        if section is None:
            return None
        if section.flags & EXECUTABLE_FLAG:
            return None
        if section.flags and not (section.flags & READABLE_FLAG):
            return None

        data = self.read(va, min(max_len, section.end - va))
        if not data:
            return None
        if not (32 <= data[0] <= 126):
            return None
        end = data.find(b"\0")
        if end <= 0:
            return None
        raw = data[:end]
        if any(ch < 32 or ch > 126 for ch in raw):
            return None
        text = raw.decode("ascii", errors="ignore")
        if predicate is not None and not predicate(text):
            return None
        return text
