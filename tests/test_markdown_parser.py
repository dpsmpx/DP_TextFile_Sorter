"""Тесты разбора Markdown с учётом специфики Obsidian."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from md_sorter import markdown_parser
from md_sorter.config import ScoringWeights
from md_sorter.markdown_parser import (
    extract_headings,
    extract_inline_tags,
    extract_wiki_links,
    parse_frontmatter,
    parse_note,
    split_code_blocks,
)
from md_sorter.models import NoteRecord


def make_record(name: str = "note.md", relative: str = "note.md") -> NoteRecord:
    """Минимальная запись о файле для тестов разбора."""
    return NoteRecord(
        path=Path("/tmp") / name,
        relative_path=PurePosixPath(relative),
        name=name,
        stem=Path(name).stem,
        size=0,
        mtime=0.0,
    )


FRONTMATTER_NOTE = """---
tags:
  - python
  - programming
aliases: [Питон, Py]
---

# Заголовок

Текст.
"""


@pytest.fixture(params=[True, False], ids=["pyyaml", "fallback"])
def yaml_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Прогоняет тесты frontmatter и с PyYAML, и со встроенным парсером."""
    if not request.param:
        monkeypatch.setattr(markdown_parser, "_yaml", None)
    return request.param


def test_frontmatter_parsed_in_both_modes(yaml_mode: bool) -> None:
    data, body = parse_frontmatter(FRONTMATTER_NOTE)
    assert data["tags"] == ["python", "programming"]
    assert data["aliases"] == ["Питон", "Py"]
    assert body.lstrip().startswith("# Заголовок")


def test_missing_frontmatter_returns_text_unchanged() -> None:
    data, body = parse_frontmatter("# Просто заголовок\n")
    assert data == {}
    assert body == "# Просто заголовок\n"


def test_broken_frontmatter_does_not_raise(yaml_mode: bool) -> None:
    text = "---\ntags: [unclosed\n  : : :\n---\n\n# Тело\n"
    data, body = parse_frontmatter(text)
    assert isinstance(data, dict)
    assert "# Тело" in body


def test_horizontal_rule_is_not_frontmatter() -> None:
    text = "# Заголовок\n\n---\n\nТекст после разделителя.\n"
    data, body = parse_frontmatter(text)
    assert data == {}
    assert body == text


def test_split_code_blocks_extracts_language() -> None:
    text = "Текст\n\n```python\nimport os\n```\n\nЕщё текст\n"
    plain, blocks = split_code_blocks(text)
    assert blocks == [("python", "import os")]
    assert "import os" not in plain
    assert "Ещё текст" in plain


def test_unclosed_code_block_is_captured() -> None:
    _, blocks = split_code_blocks("```bash\npkg install python\n")
    assert blocks == [("bash", "pkg install python")]


def test_headings_are_extracted_with_levels() -> None:
    headings = extract_headings("# Первый\n## Второй\nтекст\n###### Шестой\n")
    assert headings == [(1, "Первый"), (2, "Второй"), (6, "Шестой")]


def test_inline_tags_support_nesting_and_ignore_colors() -> None:
    tags = extract_inline_tags("#linux/arch и #python, цвет #fff, число #123")
    assert "linux/arch" in tags
    assert "python" in tags
    assert "fff" not in tags
    assert "123" not in tags


def test_wiki_links_drop_aliases_and_anchors() -> None:
    links = extract_wiki_links("[[Заметка|алиас]] и [[Другая#Раздел]] и [[Третья]]")
    assert links == ["Заметка", "Другая", "Третья"]


def test_parse_note_collects_all_features() -> None:
    text = FRONTMATTER_NOTE + "\n```python\nimport os\n```\n\n[[Другая заметка]] #linux/termux\n"
    note = parse_note(make_record(relative="Linux/note.md"), text, ScoringWeights())

    assert note.title == "Заголовок"
    assert "python" in note.tags
    assert "linux/termux" in note.tags
    assert note.code_languages == ("python",)
    assert note.wiki_links == ("Другая заметка",)
    assert note.tokens["python"] > note.tokens["текст"]
    # Токены исходного пути тоже учитываются.
    assert "linux" in note.tokens


def test_parse_note_on_empty_file_uses_name() -> None:
    note = parse_note(make_record("Termux шпаргалка.md", "Termux шпаргалка.md"), "", ScoringWeights())
    assert "termux" in note.tokens
    assert note.title == ""
