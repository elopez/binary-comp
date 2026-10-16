from __future__ import annotations

import json

import pytest

from binary_comp.config import ConfigError, load_project_target
from binary_comp.source.preprocessor import (
    UNDEFINED,
    PreprocessorConfig,
    active_preprocessor,
    blank_inactive_conditionals,
    configure_preprocessor,
    evaluate_condition,
)


pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_cpp")

from binary_comp.source.cpp import parse_source_function_groups  # noqa: E402


WIN32_BUILD = PreprocessorConfig(
    enabled=True,
    defined=(("_MSC_VER", "1020"),),
    undefined=frozenset({"PORT"}),
)

UNBALANCED_ARMS = b"""\
/* Function start: 0x00401000 */
int read_key(int key)
{
    int character = 0;
    if (key == 1) {
        character = 1;
    }
#ifdef PORT
    else if (port_is_alpha(key)) {
#else
    else if (original_is_alpha(key)
                 && key != 0) {
#endif
        character = key | 0x20;
    }
    return character;
}

/* Function start: 0x00401080 */
int after_read_key(void)
{
    return 2;
}
"""


@pytest.fixture(autouse=True)
def restore_preprocessor():
    saved = active_preprocessor()
    yield
    configure_preprocessor(saved)


def blank(source: str, config: PreprocessorConfig = WIN32_BUILD) -> str:
    return blank_inactive_conditionals(source.encode(), config).decode()


def test_disabled_configuration_leaves_source_unchanged():
    assert blank_inactive_conditionals(UNBALANCED_ARMS, PreprocessorConfig()) is UNBALANCED_ARMS


def test_blanking_keeps_offsets_and_line_breaks():
    blanked = blank_inactive_conditionals(UNBALANCED_ARMS, WIN32_BUILD)

    assert len(blanked) == len(UNBALANCED_ARMS)
    assert [i for i, b in enumerate(blanked) if b == 10] == [
        i for i, b in enumerate(UNBALANCED_ARMS) if b == 10
    ]
    text = blanked.decode()
    assert "port_is_alpha" not in text
    assert "#ifdef" not in text and "#else" not in text and "#endif" not in text
    assert "original_is_alpha(key)" in text


def test_unbalanced_arms_keep_every_definition_in_the_file(tmp_path):
    source = tmp_path / "input.c"
    source.write_bytes(UNBALANCED_ARMS)
    configure_preprocessor(WIN32_BUILD)

    groups = parse_source_function_groups(str(source))

    assert [(group.name, group.addresses, group.line) for group in groups] == [
        ("read_key", ("00401000",), 1),
        ("after_read_key", ("00401080",), 19),
    ]


def test_annotation_before_a_conditional_definition_names_the_active_arm(tmp_path):
    source = tmp_path / "input.c"
    source.write_text(
        """\
/* Function start: 0x00401000 */
#ifdef PORT
int port_decode(int value)
{
    return value;
}
#else
int decode(int value)
{
    return value + 1;
}
#endif
"""
    )
    configure_preprocessor(WIN32_BUILD)

    groups = parse_source_function_groups(str(source))

    assert [(group.name, group.addresses) for group in groups] == [("decode", ("00401000",))]


def test_undecided_conditional_is_parsed_as_written():
    source = """\
#ifdef UNDECLARED
int a;
#else
int b;
#endif
"""
    assert blank(source) == source


def test_short_circuit_decides_despite_an_unknown_operand():
    source = """\
#if !defined(PORT) || defined(UNDECLARED)
int original;
#endif
#if defined(PORT) && defined(UNDECLARED)
int port;
#endif
"""
    blanked = blank(source)

    assert "int original;" in blanked
    assert "int port;" not in blanked
    assert "#" not in blanked


def test_elif_chain_takes_the_first_true_arm():
    source = """\
#if _MSC_VER >= 1100
int vc5;
#elif _MSC_VER >= 1000
int vc4;
#else
int older;
#endif
"""
    blanked = blank(source)

    assert "int vc4;" in blanked
    assert "vc5" not in blanked and "older" not in blanked


def test_definitions_in_the_file_are_tracked_in_order():
    source = """\
#define FEATURE 2
#if FEATURE > 1
int enabled;
#endif
#undef FEATURE
#ifdef FEATURE
int stale;
#endif
"""
    blanked = blank(source)

    assert "int enabled;" in blanked
    assert "int stale;" not in blanked
    assert "#define FEATURE 2" in blanked


def test_definitions_in_an_undecided_arm_become_unknown():
    source = """\
#ifdef UNDECLARED
#define FEATURE
#endif
#ifdef FEATURE
int maybe;
#endif
"""
    assert blank(source) == source


def test_nested_conditionals_follow_the_active_arm():
    source = """\
#ifndef PORT
#ifdef _MSC_VER
int msvc;
#else
int other;
#endif
#ifdef UNDECLARED
int kept_a;
#else
int kept_b;
#endif
#else
#ifdef _MSC_VER
int port_msvc;
#endif
#endif
"""
    blanked = blank(source)

    assert "int msvc;" in blanked
    assert "other" not in blanked and "port_msvc" not in blanked
    assert "#ifdef UNDECLARED\nint kept_a;\n#else\nint kept_b;\n#endif" in blanked


def test_directives_inside_comments_are_not_directives():
    source = """\
/* commented out:
#ifdef PORT
*/
#ifdef PORT /* port only,
               continued */
int port;
#endif
int always;
"""
    blanked = blank(source)

    assert blanked.startswith("/* commented out:\n#ifdef PORT\n*/\n")
    assert "port only" not in blanked and "continued" not in blanked
    assert "int port;" not in blanked
    assert "int always;" in blanked


def test_malformed_conditionals_leave_source_unchanged():
    source = """\
#ifdef PORT
int port;
"""
    assert blank(source) == source


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3 == 7", 1),
        ("(1 + 2) * 3", 9),
        ("-7 / 2", -3),
        ("-7 % 2", -1),
        ("1 << 4 | 1", 17),
        ("0x10 + 010", 24),
        ("'A' == 65", 1),
        ("VALUE > 1000 ? 2 : 3", 2),
        ("defined VALUE && !defined(GONE)", 1),
        ("GONE", 0),
        ("MISSING", None),
        ("MISSING || 1", 1),
        ("MISSING && 0", 0),
        ("MISSING ? 4 : 4", 4),
        ("LIMIT(2)", None),
        ("1 / 0", None),
        ("1 +", None),
    ],
)
def test_condition_evaluation(expression, expected):
    state = {"VALUE": "1020", "GONE": UNDEFINED}

    assert evaluate_condition(expression, state) == expected


def write_config(tmp_path, document) -> str:
    path = tmp_path / "binary-comp.json"
    path.write_text(json.dumps(document))
    return str(path)


def minimal_target(**extra):
    target = {
        "original_exe": "original.exe",
        "rebuilt_exe": "rebuilt.exe",
        "map": "rebuilt.map",
        "source_dirs": ["src"],
    }
    target.update(extra)
    return target


def test_load_project_target_installs_the_declaration(tmp_path):
    config_path = write_config(tmp_path, {
        "preprocessor": {"undefined": ["IGNORED_BY_TARGET"]},
        "targets": {
            "full": minimal_target(preprocessor={
                "defined": {"_MSC_VER": 1020, "_WIN32": "1"},
                "undefined": ["PORT"],
            }),
        },
    })

    _, target = load_project_target(config_path, "full")

    expected = PreprocessorConfig(
        enabled=True,
        defined=(("_MSC_VER", "1020"), ("_WIN32", "1")),
        undefined=frozenset({"PORT"}),
    )
    assert target.preprocessor == expected
    assert active_preprocessor() == expected


def test_target_inherits_the_project_declaration(tmp_path):
    config_path = write_config(tmp_path, {
        "preprocessor": {"defined": ["_WIN32"]},
        "targets": {"full": minimal_target()},
    })

    _, target = load_project_target(config_path, "full")

    assert target.preprocessor == PreprocessorConfig(enabled=True, defined=(("_WIN32", "1"),))


def test_target_without_declaration_disables_filtering(tmp_path):
    configure_preprocessor(WIN32_BUILD)
    config_path = write_config(tmp_path, {"targets": {"full": minimal_target()}})

    _, target = load_project_target(config_path, "full")

    assert target.preprocessor == PreprocessorConfig()
    assert active_preprocessor() == PreprocessorConfig()


@pytest.mark.parametrize(
    "declaration",
    [
        {"defined": ["PORT"], "undefined": ["PORT"]},
        {"defined": ["not a name"]},
        {"defined": {"PORT": True}},
        {"undefined": "PORT"},
        ["PORT"],
    ],
)
def test_invalid_declarations_are_rejected(tmp_path, declaration):
    config_path = write_config(tmp_path, {
        "targets": {"full": minimal_target(preprocessor=declaration)},
    })

    with pytest.raises(ConfigError):
        load_project_target(config_path, "full")
