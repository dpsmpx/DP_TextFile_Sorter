"""Тесты построения категорий и зеркалирования структуры."""

from __future__ import annotations

from pathlib import PurePosixPath

from md_sorter.config import Config
from md_sorter.models import REVIEW_DIR_NAME, SORTED_DIR_NAME
from md_sorter.structure import (
    build_categories,
    build_category_profiles,
    ensure_sorted_dir,
    mirror_structure,
    sanitize_component,
)


def paths(*items: str) -> list[PurePosixPath]:
    """Удобная запись списка относительных каталогов."""
    return [PurePosixPath(item) for item in items]


def test_categories_know_their_context() -> None:
    categories = build_categories(paths("Linux", "Linux/Arch", "Linux/Termux"))
    termux = next(item for item in categories if item.key == "Linux/Termux")
    assert termux.parent_key == "Linux"
    assert termux.depth == 2
    assert termux.siblings == ("Arch",)


def test_review_directory_is_not_a_category() -> None:
    categories = build_categories(paths("Linux", REVIEW_DIR_NAME))
    assert all(category.key != REVIEW_DIR_NAME for category in categories)


def test_category_profile_includes_lexicon_terms() -> None:
    categories = build_categories(paths("Linux", "Linux/Termux"))
    build_category_profiles(categories, [], Config())
    termux = next(item for item in categories if item.key == "Linux/Termux")
    assert "termux" in termux.tokens
    assert "linux" in termux.tokens  # контекст родителя
    assert "pkg" in termux.tokens  # из встроенного словаря


def test_mirror_creates_only_directories(tmp_path) -> None:
    root = tmp_path / "INBOX"
    (root / "Programming" / "Python").mkdir(parents=True)
    (root / "Programming" / "Python" / "note.md").write_text("# note", encoding="utf-8")

    config = Config(inbox=root)
    import logging

    logger = logging.getLogger("md_sorter.test")
    ensure_sorted_dir(config, logger)
    mapping = mirror_structure(paths("Programming", "Programming/Python"), config, logger)

    assert (root / SORTED_DIR_NAME / "Programming" / "Python").is_dir()
    assert not (root / SORTED_DIR_NAME / "Programming" / "Python" / "note.md").exists()
    assert mapping["Programming/Python"].as_posix() == "Programming/Python"


def test_mirror_is_noop_in_dry_run(tmp_path) -> None:
    import logging

    root = tmp_path / "INBOX"
    root.mkdir()
    config = Config(inbox=root, dry_run=True)
    mapping = mirror_structure(paths("Games", "Games/Doom"), config, logging.getLogger("md_sorter.test"))
    assert not (root / SORTED_DIR_NAME).exists()
    assert mapping["Games/Doom"].as_posix() == "Games/Doom"


def test_mirror_preserves_existing_content(tmp_path) -> None:
    import logging

    root = tmp_path / "INBOX"
    existing = root / SORTED_DIR_NAME / "Linux"
    existing.mkdir(parents=True)
    keeper = existing / "old.md"
    keeper.write_text("# уже отсортировано", encoding="utf-8")

    config = Config(inbox=root)
    mirror_structure(paths("Linux", "Linux/Arch"), config, logging.getLogger("md_sorter.test"))
    assert keeper.read_text(encoding="utf-8") == "# уже отсортировано"


def test_sanitize_component_handles_platform_quirks() -> None:
    assert sanitize_component("C++") == "C++"
    assert sanitize_component("AUX") == "AUX_"
    assert sanitize_component('bad:name?') == "bad_name_"
    assert sanitize_component("trailing. ") == "trailing"
    assert sanitize_component("") == "_"
