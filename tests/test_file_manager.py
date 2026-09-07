"""Тесты безопасного копирования и разрешения конфликтов имён."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from md_sorter.models import CopyOutcome, NoteRecord, SORTED_DIR_NAME
from md_sorter.file_manager import file_sha256, load_manifest, place_note, save_manifest


def make_record(root: Path, relative: str) -> NoteRecord:
    """Создаёт запись о существующем файле."""
    path = root / relative
    stat = path.stat()
    return NoteRecord(
        path=path,
        relative_path=PurePosixPath(relative),
        name=path.name,
        stem=path.stem,
        size=stat.st_size,
        mtime=stat.st_mtime,
    )


def test_copy_leaves_source_untouched(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root)
    result = place_note(make_record(root, "note.md"), PurePosixPath("Linux"), config, logger)

    assert result.outcome is CopyOutcome.COPIED
    assert (root / "note.md").read_text(encoding="utf-8") == "# заметка"
    assert (root / SORTED_DIR_NAME / "Linux" / "note.md").read_text(encoding="utf-8") == "# заметка"


def test_identical_file_is_not_copied_twice(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root)
    record = make_record(root, "note.md")

    first = place_note(record, PurePosixPath("Linux"), config, logger)
    second = place_note(record, PurePosixPath("Linux"), config, logger)

    assert first.outcome is CopyOutcome.COPIED
    assert second.outcome is CopyOutcome.IDENTICAL
    assert len(list((root / SORTED_DIR_NAME / "Linux").glob("*.md"))) == 1


def test_name_conflict_creates_suffixed_copy(make_vault, config_factory, logger) -> None:
    root = make_vault({"A/note.md": "# первая", "B/note.md": "# вторая"})
    config = config_factory(root)

    first = place_note(make_record(root, "A/note.md"), PurePosixPath("Linux"), config, logger)
    second = place_note(make_record(root, "B/note.md"), PurePosixPath("Linux"), config, logger)

    assert first.destination.name == "note.md"
    assert second.destination.name == "note_1.md"
    assert second.outcome is CopyOutcome.RENAMED
    target = root / SORTED_DIR_NAME / "Linux"
    assert (target / "note.md").read_text(encoding="utf-8") == "# первая"
    assert (target / "note_1.md").read_text(encoding="utf-8") == "# вторая"


def test_dry_run_writes_nothing(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root, dry_run=True)
    result = place_note(make_record(root, "note.md"), PurePosixPath("Linux"), config, logger)

    assert result.outcome is CopyOutcome.DRY_RUN
    assert result.destination.as_posix() == "Linux/note.md"
    assert not (root / SORTED_DIR_NAME).exists()


def test_copy_into_root_of_result(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root)
    result = place_note(make_record(root, "note.md"), PurePosixPath("."), config, logger)
    assert result.destination.as_posix() == "note.md"
    assert (root / SORTED_DIR_NAME / "note.md").is_file()


def test_move_requires_explicit_permission(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root, copy_mode="move")  # без allow_move
    place_note(make_record(root, "note.md"), PurePosixPath("Linux"), config, logger)
    assert (root / "note.md").exists(), "без --allow-move исходник обязан остаться"


def test_move_mode_removes_source_when_allowed(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root, copy_mode="move", allow_move=True)
    place_note(make_record(root, "note.md"), PurePosixPath("Linux"), config, logger)
    assert not (root / "note.md").exists()
    assert (root / SORTED_DIR_NAME / "Linux" / "note.md").is_file()


def test_manifest_roundtrip(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root, manifest=True)
    (root / SORTED_DIR_NAME).mkdir()
    entries = {"note.md": {"destination": "Linux/note.md", "category": "Linux", "score": 0.9}}

    save_manifest(config, entries, logger, version="test")
    assert load_manifest(config, logger) == entries


def test_broken_manifest_is_ignored(make_vault, config_factory, logger) -> None:
    root = make_vault({"note.md": "# заметка"})
    config = config_factory(root, manifest=True)
    (root / SORTED_DIR_NAME).mkdir()
    (root / SORTED_DIR_NAME / ".md_sorter_manifest.json").write_text("{битый", encoding="utf-8")
    assert load_manifest(config, logger) == {}


def test_file_sha256_matches_content(make_vault) -> None:
    import hashlib

    root = make_vault({"note.md": "# заметка"})
    expected = hashlib.sha256((root / "note.md").read_bytes()).hexdigest()
    assert file_sha256(root / "note.md") == expected
