"""Сквозные тесты: запуск программы целиком через CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import tree_digest
from md_sorter.cli import EXIT_FATAL, EXIT_OK, main
from md_sorter.models import MANIFEST_NAME, REVIEW_DIR_NAME, SORTED_DIR_NAME

VAULT = {
    "Как установить пакеты в Termux.md": (
        "# Установка пакетов в Termux\n\nИспользуется pkg install, termux-setup-storage.\n"
    ),
    "network_setup.md": (
        "# Настройка сети\n\npacman и yay из AUR, дистрибутив Arch Linux.\n"
    ),
    "pathlib.md": "# Пути\n\n```python\nimport os\nfrom pathlib import Path\n```\n\npip, venv.\n",
    "redstone.md": "---\ntags: [minecraft]\n---\n\n# Редстоун\n\nПоршни, наблюдатели, ферма.\n",
    "мысль.md": "# Мысль\n\nРазобрать коробки на балконе, позвонить в сервис.\n",
    "Linux/Arch/pacman.md": "# pacman\n\npacman -Syu, чистка кэша, AUR.\n",
    "Programming/Python/декораторы.md": "# Декораторы\n\n```python\nfrom functools import wraps\n```\n",
    "Linux/Termux/proot.md": "# proot\n\nЗапуск дистрибутива внутри termux, pkg install proot.\n",
    "Games/Doom/wad.md": "# WAD\n\nРедактор карт, дробовик, демоны, gzdoom.\n",
    "Games/Minecraft/ферма.md": "# Ферма\n\nМод через forge, редстоун, наблюдатели.\n",
    "Programming/C++/шаблоны.md": "# Шаблоны\n\n```cpp\ntemplate <typename T>\n```\n\nstl, cmake.\n",
    "покупки.md": "# Список\n\nМолоко, хлеб, батарейки, зайти на почту.\n",
}
DIRECTORIES = (
    "Programming/Python",
    "Programming/C++",
    "Games/Minecraft",
    "Games/Doom",
    "Linux/Arch",
    "Linux/Termux",
)


@pytest.fixture
def vault(make_vault) -> Path:
    """Готовое хранилище с заметками и структурой категорий."""
    return make_vault(VAULT, DIRECTORIES)


def run(root: Path, *args: str) -> int:
    """Запускает CLI с отключённым цветом и без чтения конфигурации проекта."""
    return main(["--root", str(root), "--no-color", "--no-config", *args])


def test_dry_run_changes_nothing(vault: Path) -> None:
    before = tree_digest(vault)
    assert run(vault, "--dry-run") == EXIT_OK
    assert not (vault / SORTED_DIR_NAME).exists()
    assert tree_digest(vault) == before


def test_full_run_mirrors_structure_and_copies(vault: Path) -> None:
    sources_before = tree_digest(vault, exclude=SORTED_DIR_NAME)
    assert run(vault) == EXIT_OK

    sorted_dir = vault / SORTED_DIR_NAME
    for directory in DIRECTORIES:
        assert (sorted_dir / directory).is_dir(), directory

    assert (sorted_dir / "Linux" / "Termux" / "Как установить пакеты в Termux.md").is_file()
    assert (sorted_dir / "Linux" / "Arch" / "network_setup.md").is_file()
    assert (sorted_dir / "Programming" / "Python" / "pathlib.md").is_file()
    assert (sorted_dir / "Games" / "Minecraft" / "redstone.md").is_file()

    assert tree_digest(vault, exclude=SORTED_DIR_NAME) == sources_before


def test_structure_copy_creates_empty_directories(make_vault) -> None:
    """Копируется структура, а не файлы: пустой каталог остаётся пустым."""
    root = make_vault(
        {"Games/Doom/секрет.md": "# Секрет\n\nДробовик, демоны, gzdoom, wad.\n"},
        ("Games/Doom", "Games/Minecraft", "Linux/Arch"),
    )
    run(root)
    sorted_dir = root / SORTED_DIR_NAME
    assert (sorted_dir / "Games" / "Minecraft").is_dir()
    assert list((sorted_dir / "Games" / "Minecraft").iterdir()) == []
    assert (sorted_dir / "Linux" / "Arch").is_dir()
    assert list((sorted_dir / "Linux" / "Arch").iterdir()) == []
    # Исходник остался на месте, а его копия легла в свою категорию.
    assert (root / "Games" / "Doom" / "секрет.md").is_file()
    assert (sorted_dir / "Games" / "Doom" / "секрет.md").is_file()


def test_uncertain_goes_to_review(vault: Path) -> None:
    run(vault)
    review = vault / SORTED_DIR_NAME / REVIEW_DIR_NAME
    assert (review / "мысль.md").is_file()


def test_second_run_is_idempotent(vault: Path) -> None:
    assert run(vault) == EXIT_OK
    first = tree_digest(vault)
    assert run(vault) == EXIT_OK
    assert tree_digest(vault) == first

    files = list((vault / SORTED_DIR_NAME).rglob("*.md"))
    assert len(files) == len(set(files))


def test_no_nested_sorted_directory(vault: Path) -> None:
    run(vault)
    run(vault)
    assert not (vault / SORTED_DIR_NAME / SORTED_DIR_NAME).exists()
    assert not list((vault / SORTED_DIR_NAME).rglob(SORTED_DIR_NAME))


def test_changed_source_creates_new_copy_without_touching_old(vault: Path) -> None:
    run(vault)
    target = vault / SORTED_DIR_NAME / "Programming" / "Python" / "pathlib.md"
    original_copy = target.read_text(encoding="utf-8")

    (vault / "pathlib.md").write_text(
        "# Пути\n\n```python\nimport os\n```\n\nОбновлённая версия, pip и venv.\n",
        encoding="utf-8",
    )
    run(vault)

    assert target.read_text(encoding="utf-8") == original_copy
    assert (vault / SORTED_DIR_NAME / "Programming" / "Python" / "pathlib_1.md").is_file()


def test_name_conflicts_are_suffixed(make_vault) -> None:
    root = make_vault(
        {
            "A/note.md": "# Termux\n\npkg install, termux-setup-storage, android.\n",
            "B/note.md": "# Termux иначе\n\npkg upgrade, proot, хранилище телефона.\n",
        },
        ("Linux/Termux",),
    )
    run(root)
    termux = root / SORTED_DIR_NAME / "Linux" / "Termux"
    names = sorted(path.name for path in termux.glob("*.md"))
    assert names == ["note.md", "note_1.md"]


def test_extension_case_is_ignored(make_vault) -> None:
    root = make_vault(
        {"A.md": "# Termux\n\npkg install\n", "B.MD": "# Termux\n\npkg upgrade\n"},
        ("Linux/Termux",),
    )
    run(root, "--threshold", "0.0")
    copied = sorted(path.name for path in (root / SORTED_DIR_NAME).rglob("*") if path.is_file())
    assert copied == ["A.md", "B.MD"]


def test_unicode_and_spaces_in_names(make_vault) -> None:
    root = make_vault(
        {"Заметка про Termux 🚀.md": "# Termux\n\npkg install python\n"},
        ("Linux/Termux",),
    )
    assert run(root) == EXIT_OK
    assert (root / SORTED_DIR_NAME / "Linux" / "Termux" / "Заметка про Termux 🚀.md").is_file()


def test_vault_without_categories_sends_all_to_review(make_vault) -> None:
    root = make_vault({"a.md": "# Termux\n\npkg install\n"})
    assert run(root) == EXIT_OK
    assert (root / SORTED_DIR_NAME / REVIEW_DIR_NAME / "a.md").is_file()


def test_empty_vault_is_not_an_error(make_vault) -> None:
    root = make_vault({}, ("Linux",))
    assert run(root) == EXIT_OK


def test_manifest_is_opt_in(vault: Path) -> None:
    run(vault)
    assert not (vault / SORTED_DIR_NAME / MANIFEST_NAME).exists()

    run(vault, "--manifest")
    assert (vault / SORTED_DIR_NAME / MANIFEST_NAME).is_file()


def test_manifest_run_stays_idempotent(vault: Path) -> None:
    run(vault, "--manifest")
    first = tree_digest(vault)
    run(vault, "--manifest")
    assert tree_digest(vault) == first


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("root", "мысль.md"), ("review", f"{REVIEW_DIR_NAME}/мысль.md")],
)
def test_uncertain_strategies(vault: Path, strategy: str, expected: str) -> None:
    run(vault, "--uncertain-strategy", strategy)
    assert (vault / SORTED_DIR_NAME / expected).is_file()


def test_uncertain_strategy_skip_copies_nothing(vault: Path) -> None:
    run(vault, "--uncertain-strategy", "skip")
    assert not (vault / SORTED_DIR_NAME / REVIEW_DIR_NAME).exists()
    assert not (vault / SORTED_DIR_NAME / "мысль.md").exists()


def test_verbose_explains_decision(vault: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "--dry-run", "--verbose")
    output = capsys.readouterr().out
    assert "Почему:" in output
    assert "Score:" in output
    assert "совпадение терминов" in output


def test_output_matches_expected_shape(vault: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "--dry-run")
    output = capsys.readouterr().out
    assert "[INFO] Root:" in output
    assert "Searching Markdown files..." in output
    assert "Found:" in output and "files" in output
    assert "destination directories" in output
    assert "Processed:" in output and "Uncertain:" in output


def test_running_inside_sorted_directory_is_refused(vault: Path) -> None:
    run(vault)
    inside = vault / SORTED_DIR_NAME / "Linux"
    assert main(["--root", str(inside), "--no-color", "--no-config"]) == EXIT_FATAL


def test_missing_root_is_fatal(tmp_path: Path) -> None:
    assert main(["--root", str(tmp_path / "нет"), "--no-color", "--no-config"]) == EXIT_FATAL


def test_move_without_permission_is_refused(vault: Path) -> None:
    assert run(vault, "--move") == EXIT_FATAL
    assert (vault / "pathlib.md").is_file()


def test_config_file_is_applied(make_vault) -> None:
    root = make_vault({"a.md": "# Termux\n\npkg install\n"}, ("Linux/Termux",))
    (root / "md_sorter.toml").write_text(
        "[md_sorter]\nthreshold = 0.99\nmin_evidence = 1.0\n"
        'uncertain_strategy = "root"\n',
        encoding="utf-8",
    )
    assert main(["--root", str(root), "--no-color"]) == EXIT_OK
    assert (root / SORTED_DIR_NAME / "a.md").is_file()


def test_cli_overrides_config_file(make_vault) -> None:
    root = make_vault({"a.md": "# Termux\n\npkg install python\n"}, ("Linux/Termux",))
    (root / "md_sorter.toml").write_text("[md_sorter]\nthreshold = 0.99\n", encoding="utf-8")
    main(["--root", str(root), "--no-color", "--threshold", "0.05"])
    assert (root / SORTED_DIR_NAME / "Linux" / "Termux" / "a.md").is_file()


def test_broken_file_does_not_stop_the_run(make_vault) -> None:
    root = make_vault({"ok.md": "# Termux\n\npkg install\n"}, ("Linux/Termux",))
    (root / "broken.md").write_bytes(b"\xff\xfe\x00\x01 not text")
    assert run(root) == EXIT_OK
    assert (root / SORTED_DIR_NAME / "Linux" / "Termux" / "ok.md").is_file()


def test_aliases_from_config_steer_classification(make_vault) -> None:
    root = make_vault(
        {"мотоцикл.md": "# Обслуживание\n\nЦепь, масло, тормозные колодки, пробег.\n"},
        ("Hobby/Moto", "Linux"),
    )
    (root / "md_sorter.toml").write_text(
        '[md_sorter.aliases]\n"Hobby/Moto" = ["мотоцикл", "цепь", "масло", "колодки", "пробег"]\n',
        encoding="utf-8",
    )
    assert main(["--root", str(root), "--no-color"]) == EXIT_OK
    assert (root / SORTED_DIR_NAME / "Hobby" / "Moto" / "мотоцикл.md").is_file()


# --- Новое правило приёма и режимы, добавленные для повышения полноты ---------


def test_uncertain_strategy_best_places_everything(vault: Path) -> None:
    """Режим «раскладывать всё»: в _NEEDS_REVIEW попадают только заметки без сигнала."""
    run(vault, "--uncertain-strategy", "best")
    review = vault / SORTED_DIR_NAME / REVIEW_DIR_NAME
    placed = list((vault / SORTED_DIR_NAME).rglob("*.md"))
    in_review = list(review.rglob("*.md")) if review.exists() else []
    assert len(placed) > len(in_review) * 3


def test_dominance_rule_accepts_low_absolute_score(make_vault) -> None:
    """Личная категория, которой нет во встроенном словаре, всё равно узнаётся."""
    root = make_vault(
        {
            "Хобби/Мотоциклы/масло.md": "# Масло\n\nЗамена масла, фильтр, уровень.\n",
            "Хобби/Мотоциклы/цепь.md": "# Цепь\n\nСмазка цепи, натяжение, звёзды.\n",
            "Хобби/Мотоциклы/колодки.md": "# Колодки\n\nТормозные колодки, прокачка, диск.\n",
            "Хобби/Аквариум/вода.md": "# Вода\n\nПодмена воды, сифонка грунта, нитраты.\n",
            "Хобби/Аквариум/фильтр.md": "# Фильтр\n\nВнешний фильтр, промывка губки.\n",
            "новая заметка.md": "# Обслуживание\n\nПроверил натяжение цепи и уровень масла.\n",
        },
        ("Хобби/Мотоциклы", "Хобби/Аквариум"),
    )
    assert run(root) == EXIT_OK
    assert (root / SORTED_DIR_NAME / "Хобби" / "Мотоциклы" / "новая заметка.md").is_file()


def test_strict_settings_still_send_notes_to_review(vault: Path) -> None:
    """Оба основания для приёма отключаются, поведение остаётся предсказуемым."""
    run(vault, "--threshold", "0.99", "--min-evidence", "1.0", "--self-training-rounds", "0")
    review = vault / SORTED_DIR_NAME / REVIEW_DIR_NAME
    assert review.is_dir()
    assert len(list(review.glob("*.md"))) >= 5


def test_evaluate_mode_reports_and_changes_nothing(vault: Path) -> None:
    """Самопроверка ничего не создаёт и печатает измеримые метрики."""
    before = tree_digest(vault)
    assert main(["--root", str(vault), "--no-color", "--no-config", "--evaluate"]) == EXIT_OK
    assert tree_digest(vault) == before


def test_evaluate_output_contains_metrics(vault: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--root", str(vault), "--no-color", "--no-config", "--evaluate"])
    output = capsys.readouterr().out
    assert "Заметок с известной категорией" in output
    assert "Лучший кандидат совпал с вашим выбором" in output
    assert "Принято решений (полнота)" in output


def test_evaluate_without_filed_notes_is_fatal(make_vault) -> None:
    """Без разложенных заметок измерять нечего — об этом надо сказать прямо."""
    root = make_vault({"a.md": "# Termux\n\npkg install\n"}, ("Linux/Termux",))
    assert main(["--root", str(root), "--no-color", "--no-config", "--evaluate"]) == EXIT_FATAL
