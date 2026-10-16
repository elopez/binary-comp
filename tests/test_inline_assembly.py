from __future__ import annotations

import pytest

from binary_comp.source.cpp import blank_inline_assembly


pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_cpp")

from binary_comp.source.cpp import parse_source_function_groups  # noqa: E402


NAKED_FUNCTIONS = b"""\
/* Function start: 0x00401000 */
__declspec(naked) unsigned int read_pixel(int x, int y)
{
    __asm {
        push ebp
        mov ebp, esp
        mov eax, dword ptr [ebp + 8]   ; x, never } here
        cmp eax, 0
        jle done
        mov eax, 1
done:
        pop ebp
        ret
    }
}

/* Function start: 0x00401040 */
__declspec(naked) void write_pixel(void)
{
    __asm
    {
        xor eax, eax
        ret
    }
}

/* Function start: 0x00401050 */
int after_pixels(void)
{
    __asm int 3
    return 0;
}
"""


def blank(source: str) -> str:
    return blank_inline_assembly(source.encode()).decode()


def test_inline_assembly_bodies_keep_their_functions(tmp_path):
    source = tmp_path / "raster.c"
    source.write_bytes(NAKED_FUNCTIONS)

    groups = parse_source_function_groups(str(source))

    assert [(group.name, group.addresses) for group in groups] == [
        ("read_pixel", ("00401000",)),
        ("write_pixel", ("00401040",)),
        ("after_pixels", ("00401050",)),
    ]


def test_blanking_keeps_offsets_line_breaks_and_surrounding_code():
    blanked = blank_inline_assembly(NAKED_FUNCTIONS)

    assert len(blanked) == len(NAKED_FUNCTIONS)
    assert blanked.count(b"\n") == NAKED_FUNCTIONS.count(b"\n")
    text = blanked.decode()
    for instruction in ("push ebp", "mov eax", "never", "done:", "xor eax", "int 3"):
        assert instruction not in text
    assert "__asm" not in text
    assert "return 0;" in text
    assert text.count("{") == text.count("}") == 3


def test_statement_ends_at_the_line_break_or_a_closing_brace():
    assert blank("if (x) { __asm int 3 }\ny();\n") == "if (x) {             }\ny();\n"
    assert blank("_asm mov eax, 1 ; note }\nz();\n") == " " * 24 + "\nz();\n"


def test_assembly_spellings_in_comments_strings_and_gcc_syntax_are_kept():
    source = (
        '/* __asm { not code } */\n'
        'const char *name = "__asm";\n'
        '__asm__("nop");\n'
        'int __asmx;\n'
    )

    assert blank(source) == source
