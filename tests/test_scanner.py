"""Тесты обхода дерева: исключения, symlink, скрытые и проблемные файлы."""

from __future__ import annotations

import os
import stat

import pytest

from md_sorter.models import SORTED_DIR_NAME
from md_sorter.scanner import read_note_text, scan_tree


def relative_names(result: object) -> set[str]:
    """Множество относительных путей найденных заметок."""
    return {note.relative_path.as_posix() for note in result.notes}  # type: ignore[attr-defined]


def test_finds_markdown_case_insensitively(make_vault, config_factory, logger) -> None:
    root = make_vault({"a.md": "# a", "b.MD": "# b", "c.Md": "# c", "d.txt": "не markdown"})
    result = scan_tree(root, config_factory(root), logger)
    assert relative_names(result) == {"a.md", "b.MD", "c.Md"}


def test_sorted_directory_is_excluded(make_vault, config_factory, logger) -> None:
    root = make_vault(
        {
            "note.md": "# note",
            f"{SORTED_DIR_NAME}/Linux/copied.md": "# копия",
            f"{SORTED_DIR_NAME}/_NEEDS_REVIEW/other.md": "# ещё копия",
        }
    )
    result = scan_tree(root, config_factory(root), logger)
    assert relative_names(result) == {"note.md"}
    assert all(SORTED_DIR_NAME not in path.parts for path in result.directories)


def test_hidden_entries_skipped_by_default(make_vault, config_factory, logger) -> None:
    root = make_vault({".hidden.md": "# скрытая", ".obsidian/plugin.md": "# плагин", "ok.md": "# ok"})
    assert relative_names(scan_tree(root, config_factory(root), logger)) == {"ok.md"}


def test_hidden_entries_included_on_demand(make_vault, config_factory, logger) -> None:
    root = make_vault({".hidden.md": "# скрытая", "ok.md": "# ok"})
    config = config_factory(root, include_hidden=True)
    assert ".hidden.md" in relative_names(scan_tree(root, config, logger))


def test_ignored_directories_are_pruned(make_vault, config_factory, logger) -> None:
    root = make_vault({"node_modules/pkg/readme.md": "# нет", "ok.md": "# ok"})
    assert relative_names(scan_tree(root, config_factory(root), logger)) == {"ok.md"}


@pytest.mark.skipif(os.name == "nt", reason="символические ссылки требуют прав в Windows")
def test_symlinks_skipped_by_default(make_vault, config_factory, logger) -> None:
    root = make_vault({"real.md": "# real"})
    (root / "link.md").symlink_to(root / "real.md")
    assert relative_names(scan_tree(root, config_factory(root), logger)) == {"real.md"}


@pytest.mark.skipif(os.name == "nt", reason="символические ссылки требуют прав в Windows")
def test_symlink_loop_does_not_hang(make_vault, config_factory, logger) -> None:
    root = make_vault({"sub/real.md": "# real"})
    (root / "sub" / "loop").symlink_to(root, target_is_directory=True)
    config = config_factory(root, follow_symlinks=True)
    result = scan_tree(root, config, logger)
    assert "sub/real.md" in relative_names(result)


@pytest.mark.skipif(os.geteuid() == 0, reason="root игнорирует права доступа")
def test_unreadable_file_is_reported_not_fatal(make_vault, config_factory, logger) -> None:
    root = make_vault({"ok.md": "# ok", "secret.md": "# secret"})
    secret = root / "secret.md"
    secret.chmod(0)
    try:
        config = config_factory(root)
        result = scan_tree(root, config, logger)
        assert "secret.md" in relative_names(result)
        with pytest.raises(PermissionError):
            read_note_text(secret, config)
    finally:
        secret.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_directories_collected_for_mirroring(make_vault, config_factory, logger) -> None:
    root = make_vault({}, directories=("Programming/Python", "Games/Doom"))
    result = scan_tree(root, config_factory(root), logger)
    assert {path.as_posix() for path in result.directories} == {
        "Programming",
        "Programming/Python",
        "Games",
        "Games/Doom",
    }


def test_large_file_is_truncated_not_skipped(make_vault, config_factory, logger) -> None:
    root = make_vault({"big.md": "# заголовок\n" + ("слово " * 50_000)})
    config = config_factory(root, max_analysis_chars=1000)
    text, truncated = read_note_text(root / "big.md", config)
    assert truncated is True
    assert len(text) == 1000


def test_invalid_encoding_does_not_raise(make_vault, config_factory, logger) -> None:
    root = make_vault({})
    broken = root / "broken.md"
    broken.write_bytes(b"# \xff\xfe\x00 broken bytes")
    text, _ = read_note_text(broken, config_factory(root))
    assert isinstance(text, str)


def test_bom_is_stripped(make_vault, config_factory, logger) -> None:
    root = make_vault({})
    path = root / "bom.md"
    path.write_bytes("﻿# Заголовок\n".encode("utf-8"))
    text, _ = read_note_text(path, config_factory(root))
    assert text.startswith("# Заголовок")
