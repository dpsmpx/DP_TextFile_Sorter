"""Проверка масштабируемости: работа не должна деградировать на больших хранилищах."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pytest

from md_sorter.cli import EXIT_OK, main

TOPICS: dict[str, str] = {
    "Programming/Python": "python pip venv django pandas pathlib pytest",
    "Programming/C++": "cpp cmake gcc stl boost template namespace",
    "Linux/Arch": "pacman aur yay archlinux systemd makepkg",
    "Linux/Termux": "termux pkg android proot storage смартфон",
    "Games/Minecraft": "minecraft мод forge редстоун ферма выживание",
    "Games/Doom": "doom wad дробовик демоны gzdoom карта",
}


@pytest.mark.parametrize("count", [400])
def test_large_vault_completes_quickly(tmp_path: Path, count: int) -> None:
    """400 заметок должны обрабатываться заметно быстрее минуты."""
    root = tmp_path / "BIG"
    for directory in TOPICS:
        (root / directory).mkdir(parents=True, exist_ok=True)

    generator = random.Random(20240907)
    for index in range(count):
        topic, terms = generator.choice(list(TOPICS.items()))
        vocabulary = terms.split()
        body = " ".join(generator.choices(vocabulary, k=25))
        (root / f"note_{index:05d}.md").write_text(
            f"# {vocabulary[0]} {index}\n\n{body}\n", encoding="utf-8"
        )

    start = time.perf_counter()
    assert main(["--root", str(root), "--dry-run", "--no-color", "--no-config", "-q"]) == EXIT_OK
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"обработка {count} заметок заняла {elapsed:.1f} c"


def test_repeated_run_does_not_reread_sorted_results(tmp_path: Path) -> None:
    """Второй запуск не должен считать уже отсортированные копии исходниками."""
    root = tmp_path / "INBOX"
    (root / "Linux" / "Termux").mkdir(parents=True)
    for index in range(20):
        (root / f"note_{index}.md").write_text(
            f"# Termux {index}\n\npkg install, proot, android, storage.\n", encoding="utf-8"
        )

    assert main(["--root", str(root), "--no-color", "--no-config", "-q"]) == EXIT_OK
    first = sorted(path.name for path in (root / "Sorted_md_files").rglob("*.md"))

    assert main(["--root", str(root), "--no-color", "--no-config", "-q"]) == EXIT_OK
    second = sorted(path.name for path in (root / "Sorted_md_files").rglob("*.md"))

    assert first == second
    assert len(second) == 20
