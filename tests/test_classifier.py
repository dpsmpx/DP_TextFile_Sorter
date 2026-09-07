"""Тесты классификации и принятия решения."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from md_sorter.classifier import available_classifiers, create_classifier
from md_sorter.classifier.lexical import LexicalClassifier
from md_sorter.config import Config, ScoringWeights
from md_sorter.decision import decide, resolve_destination
from md_sorter.markdown_parser import parse_note
from md_sorter.models import (
    Candidate,
    Decision,
    NoteRecord,
    ParsedNote,
    REVIEW_DIR_NAME,
    Status,
)
from md_sorter.structure import build_categories, build_category_profiles

DIRECTORIES = (
    "Programming",
    "Programming/Python",
    "Programming/C++",
    "Programming/JavaScript",
    "Games",
    "Games/Minecraft",
    "Games/Doom",
    "Linux",
    "Linux/Arch",
    "Linux/Termux",
)


#: Фоновый корпус: IDF считается по всему хранилищу, поэтому оценки
#: осмысленны только на реалистичном наборе заметок, а не на одной.
CORPUS: dict[str, str] = {
    "Как установить пакеты в Termux.md": (
        "# Установка пакетов в Termux\n\nДля установки используется pkg install.\n"
        "Хранилище подключается через termux-setup-storage.\n"
    ),
    "network_setup.md": (
        "# Настройка сети\n\nУстанавливаем через pacman, дополнительно yay из AUR.\n"
        "Это дистрибутив Arch Linux.\n"
    ),
    "шпаргалка по путям.md": (
        "# Пути\n\n```python\nimport os\nfrom pathlib import Path\n```\n"
    ),
    "схемы ферм.md": (
        "---\ntags:\n  - minecraft\n---\n\n# Схемы\n\nПоршни и наблюдатели, редстоун.\n"
    ),
    "сборка проекта.md": (
        "# Сборка\n\n```cpp\n#include <iostream>\n```\n\nCMake и gcc.\n"
    ),
    "уровни и секреты.md": (
        "# Секреты\n\nWAD-файлы, дробовик, демоны, gzdoom.\n"
    ),
    "мысль.md": (
        "# Мысль\n\nНадо разобрать коробки на балконе и позвонить в сервис.\n"
    ),
    "pacman шпаргалка.md": (
        "# Шпаргалка pacman\n\n`pacman -Syu`, чистка кэша, AUR через yay.\n"
    ),
    "termux storage.md": (
        "# Хранилище Termux\n\ntermux-setup-storage, доступ к файлам телефона.\n"
    ),
    "промисы.md": (
        "# Асинхронность\n\n```javascript\nawait fetch(url)\n```\n\nnpm и node.\n"
    ),
    "мод на ферму.md": (
        "# Мод\n\nУстановка мода через Forge, автоматическая ферма.\n"
    ),
    "шаблоны cpp.md": (
        "# Шаблоны\n\n```cpp\ntemplate <typename T>\n```\n\nstl и boost.\n"
    ),
    "покупки.md": (
        "# Список\n\nМолоко, хлеб, батарейки, зайти на почту.\n"
    ),
}


def build_note(name: str, text: str, source_dir: str = ".") -> ParsedNote:
    """Разбирает заметку с заданным именем и исходным каталогом."""
    relative = PurePosixPath(name) if source_dir == "." else PurePosixPath(source_dir) / name
    record = NoteRecord(
        path=Path("/tmp") / name,
        relative_path=relative,
        name=name,
        stem=Path(name).stem,
        size=len(text),
        mtime=0.0,
    )
    return parse_note(record, text, ScoringWeights())


def classify_corpus(
    extra: dict[str, tuple[str, str]] | None = None,
    **overrides: object,
) -> dict[str, Decision]:
    """Классифицирует весь корпус и возвращает решения по именам файлов.

    Args:
        extra: дополнительные заметки ``имя -> (текст, исходный каталог)``.
        overrides: переопределения параметров конфигурации.
    """
    config = Config(inbox=Path("/tmp"))
    for key, value in overrides.items():
        setattr(config, key, value)

    notes = [build_note(name, text) for name, text in CORPUS.items()]
    notes.extend(
        build_note(name, text, source_dir)
        for name, (text, source_dir) in (extra or {}).items()
    )

    categories = build_categories([PurePosixPath(item) for item in DIRECTORIES])
    notes_by_category = build_category_profiles(categories, notes, config)
    by_key = {category.key: category for category in categories}

    classifier = LexicalClassifier(config)
    classifier.fit(notes, categories, notes_by_category)

    return {
        note.record.name: decide(note.record, classifier.rank(note, index), by_key, config)
        for index, note in enumerate(notes)
    }


def test_prefers_specific_subcategory_over_parent() -> None:
    decision = classify_corpus()["Как установить пакеты в Termux.md"]
    assert decision.status is Status.SORTED
    assert decision.category == "Linux/Termux"


def test_arch_note_beats_sibling_and_parent() -> None:
    """Признаки Arch должны обойти и соседа Termux, и родителя Linux.

    Проверяется именно порядок кандидатов: абсолютная оценка зависит от размера
    хранилища (IDF считается по всему корпусу), а порядок — нет.
    """
    decision = classify_corpus()["network_setup.md"]
    ranked = [candidate.category for candidate in decision.candidates]
    assert ranked[0] == "Linux/Arch"
    assert ranked.index("Linux/Arch") < ranked.index("Linux")


def test_code_block_language_drives_classification() -> None:
    assert classify_corpus()["шпаргалка по путям.md"].category == "Programming/Python"
    assert classify_corpus()["сборка проекта.md"].category == "Programming/C++"


def test_frontmatter_tags_are_strong_signal() -> None:
    assert classify_corpus()["схемы ферм.md"].category == "Games/Minecraft"


def test_unrelated_note_is_uncertain() -> None:
    decision = classify_corpus()["мысль.md"]
    assert decision.status is Status.UNCERTAIN
    assert decision.category is None


def test_source_path_supports_but_does_not_dictate() -> None:
    """Заметка про Python из каталога Games всё равно тянется к Python."""
    extra = {
        "декораторы.md": (
            "# Декораторы\n\n```python\nfrom functools import wraps\n```\n\npip, venv, pytest.\n",
            "Games",
        )
    }
    decision = classify_corpus(extra)["декораторы.md"]
    assert decision.candidates[0].category == "Programming/Python"


def test_source_path_prior_does_not_boost_name_match() -> None:
    """Каталог не считается «именем» заметки: бонус за совпадение имени не даётся."""
    extra = {"декораторы.md": ("# Декораторы\n\n```python\nimport functools\n```\n", "Games")}
    decision = classify_corpus(extra)["декораторы.md"]
    games = next(item for item in decision.candidates if item.category == "Games")
    assert "name_match" not in games.components
    assert games.components["source_path_prior"] > 0.0


def test_source_path_prior_helps_when_content_is_thin() -> None:
    """Скупая заметка остаётся в своём каталоге, а не уходит в чужую категорию."""
    extra = {"черновик.md": ("Пара строк без ключевых слов.\n", "Linux/Termux")}
    decision = classify_corpus(extra)["черновик.md"]
    assert decision.category in {"Linux/Termux", None}


def test_dominant_leader_is_accepted_below_absolute_threshold() -> None:
    """Абсолютная величина косинуса зависит от «толщины» профиля категории.

    Поэтому лидер, оторвавшийся от соперников, принимается даже при заведомо
    недостижимом пороге — иначе личные категории пользователя, которых нет во
    встроенном словаре, никогда не набирали бы нужную оценку.
    """
    decisions = classify_corpus(threshold=0.99)
    sorted_names = [
        name for name, item in decisions.items() if item.status is Status.SORTED
    ]
    assert sorted_names, "относительное правило должно принимать явных лидеров"
    assert decisions["termux storage.md"].category == "Linux/Termux"


def test_strict_settings_make_everything_uncertain() -> None:
    """Оба основания для приёма отключаются независимо друг от друга."""
    decisions = classify_corpus(threshold=0.99, min_evidence=1.0)
    assert all(item.status is Status.UNCERTAIN for item in decisions.values())


def test_weak_leader_without_dominance_is_uncertain() -> None:
    """Слабый сигнал без отрыва от соперников остаётся сомнительным."""
    categories = build_categories([PurePosixPath(item) for item in DIRECTORIES])
    config = Config(inbox=Path("/tmp"))
    record = build_note("x.md", "# x").record
    candidates = [
        Candidate(category="Linux/Arch", score=0.04),
        Candidate(category="Games/Doom", score=0.035),
    ]
    decision = decide(record, candidates, {c.key: c for c in categories}, config)
    assert decision.status is Status.UNCERTAIN


def test_single_weak_candidate_is_accepted_when_alone() -> None:
    """Если конкурентов нет вовсе, слабого, но реального сигнала достаточно."""
    categories = build_categories([PurePosixPath(item) for item in DIRECTORIES])
    config = Config(inbox=Path("/tmp"))
    record = build_note("x.md", "# x").record
    candidates = [Candidate(category="Linux/Arch", score=0.09)]
    decision = decide(record, candidates, {c.key: c for c in categories}, config)
    assert decision.status is Status.SORTED
    assert decision.category == "Linux/Arch"


def test_ambiguous_siblings_fall_back_to_parent() -> None:
    categories = build_categories([PurePosixPath(item) for item in DIRECTORIES])
    config = Config(inbox=Path("/tmp"))
    record = build_note("x.md", "# x").record
    candidates = [
        Candidate(category="Linux/Arch", score=0.60),
        Candidate(category="Linux/Termux", score=0.59),
        Candidate(category="Linux", score=0.55),
    ]
    decision = decide(record, candidates, {c.key: c for c in categories}, config)
    assert decision.status is Status.SORTED
    assert decision.category == "Linux"


def test_ambiguous_unrelated_categories_are_uncertain() -> None:
    categories = build_categories([PurePosixPath(item) for item in DIRECTORIES])
    config = Config(inbox=Path("/tmp"))
    record = build_note("x.md", "# x").record
    candidates = [
        Candidate(category="Linux/Arch", score=0.60),
        Candidate(category="Games/Doom", score=0.59),
    ]
    decision = decide(record, candidates, {c.key: c for c in categories}, config)
    assert decision.status is Status.UNCERTAIN


def test_no_categories_means_uncertain() -> None:
    config = Config(inbox=Path("/tmp"))
    record = build_note("x.md", "# x").record
    decision = decide(record, [], {}, config)
    assert decision.status is Status.UNCERTAIN
    assert "нет каталогов" in decision.reason


def test_uncertain_strategies_pick_expected_directory() -> None:
    config = Config(inbox=Path("/tmp"))
    record = build_note("x.md", "# x").record
    decision = decide(record, [], {}, config)
    mapping = {"Linux": PurePosixPath("Linux")}

    config.uncertain_strategy = "review"
    assert resolve_destination(decision, mapping, config).as_posix() == REVIEW_DIR_NAME
    config.uncertain_strategy = "root"
    assert resolve_destination(decision, mapping, config).as_posix() == "."
    config.uncertain_strategy = "skip"
    assert resolve_destination(decision, mapping, config) is None

    config.uncertain_strategy = "ancestor"
    decision.candidates = [
        Candidate(category="Linux/Arch", score=0.5),
        Candidate(category="Linux/Termux", score=0.49),
    ]
    assert resolve_destination(decision, mapping, config).as_posix() == "Linux"


def test_explanation_is_available_for_verbose_mode() -> None:
    decision = classify_corpus()["network_setup.md"]
    best = decision.candidates[0]
    assert best.evidence, "объяснение должно быть непустым"
    assert "similarity" in best.components
    assert any("Arch" in item or "arch" in item for item in best.evidence)


def test_registry_exposes_all_classifiers() -> None:
    assert "lexical" in available_classifiers()
    assert isinstance(create_classifier(Config(inbox=Path("/tmp"))), LexicalClassifier)


# --- Опциональные классификаторы --------------------------------------------


def _corpus_for_optional() -> tuple[list[ParsedNote], list[object], dict[str, list[int]], Config]:
    """Готовит корпус и категории для проверки альтернативных алгоритмов."""
    config = Config(inbox=Path("/tmp"))
    notes = [build_note(name, text) for name, text in CORPUS.items()]
    categories = build_categories([PurePosixPath(item) for item in DIRECTORIES])
    notes_by_category = build_category_profiles(categories, notes, config)
    return notes, categories, notes_by_category, config


def test_tfidf_classifier_ranks_like_lexical() -> None:
    """TF-IDF — сменная реализация той же задачи, порядок кандидатов сохраняется."""
    pytest.importorskip("sklearn")
    from md_sorter.classifier.tfidf import TfidfClassifier

    notes, categories, notes_by_category, config = _corpus_for_optional()
    classifier = TfidfClassifier(config)
    classifier.fit(notes, categories, notes_by_category)

    by_name = {note.record.name: index for index, note in enumerate(notes)}
    index = by_name["Как установить пакеты в Termux.md"]
    candidates = classifier.rank(notes[index], index)
    assert candidates[0].category == "Linux/Termux"


def test_tfidf_excludes_note_from_its_own_category() -> None:
    """Заметка не должна опознавать саму себя через профиль своего каталога."""
    pytest.importorskip("sklearn")
    from md_sorter.classifier.tfidf import TfidfClassifier

    config = Config(inbox=Path("/tmp"))
    notes = [
        build_note("уникальная.md", "# Совершенно уникальный текст\n\nквазиморфный субстрат.\n", "Games"),
        build_note("termux.md", "# Termux\n\npkg install python\n"),
    ]
    categories = build_categories([PurePosixPath(item) for item in DIRECTORIES])
    notes_by_category = build_category_profiles(categories, notes, config)

    classifier = TfidfClassifier(config)
    classifier.fit(notes, categories, notes_by_category)
    candidates = classifier.rank(notes[0], 0)
    scores = {item.category: item.score for item in candidates}
    assert scores.get("Games", 0.0) < 0.9, "профиль каталога не должен состоять из самой заметки"


def test_unavailable_classifier_reports_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Отсутствие зависимости — понятное сообщение, а не трассировка."""
    import builtins

    from md_sorter.classifier.tfidf import TfidfClassifier

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("sklearn"):
            raise ImportError("нет модуля")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    notes, categories, notes_by_category, config = _corpus_for_optional()
    with pytest.raises(RuntimeError, match="scikit-learn"):
        TfidfClassifier(config).fit(notes, categories, notes_by_category)
