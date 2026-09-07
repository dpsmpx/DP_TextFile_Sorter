"""Тесты обучения на самом хранилище: соседи, ссылки, самообучение.

Встроенный словарь не знает личных категорий пользователя («Хобби/Мотоциклы»),
поэтому основным источником смысла должно быть само хранилище.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from md_sorter.classifier.lexical import LexicalClassifier
from md_sorter.config import Config, ScoringWeights
from md_sorter.decision import decide
from md_sorter.markdown_parser import parse_note
from md_sorter.models import NoteRecord, ParsedNote, Status
from md_sorter.structure import build_categories, build_category_profiles

# Категории намеренно вне встроенного словаря.
PERSONAL_DIRECTORIES = ("Хобби", "Хобби/Мотоциклы", "Хобби/Аквариум", "Дом", "Дом/Ремонт")


def make_note(name: str, text: str, source_dir: str = ".") -> ParsedNote:
    """Разбирает заметку с заданным исходным каталогом."""
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


def classify(notes: list[ParsedNote], **overrides: object) -> dict[str, object]:
    """Классифицирует набор заметок и возвращает решения по именам файлов."""
    config = Config(root=Path("/tmp"))
    for key, value in overrides.items():
        setattr(config, key, value)

    categories = build_categories([PurePosixPath(item) for item in PERSONAL_DIRECTORIES])
    notes_by_category = build_category_profiles(categories, notes, config)
    by_key = {category.key: category for category in categories}

    classifier = LexicalClassifier(config)
    classifier.fit(notes, categories, notes_by_category)

    verdicts = [
        decide(note.record, classifier.rank(note, index), by_key, config)
        for index, note in enumerate(notes)
    ]
    for _ in range(max(0, config.self_training_rounds)):
        seeds = {
            index: verdict.category
            for index, verdict in enumerate(verdicts)
            if verdict.status is Status.SORTED
            and verdict.category
            and verdict.score >= config.self_training_min_score
        }
        if not seeds or classifier.learn_from(seeds) == 0:
            break
        verdicts = [
            decide(note.record, classifier.rank(note, index), by_key, config)
            for index, note in enumerate(notes)
        ]

    return {note.record.name: verdict for note, verdict in zip(notes, verdicts)}


def filed_moto_notes() -> list[ParsedNote]:
    """Несколько заметок, уже разложенных пользователем в «Хобби/Мотоциклы»."""
    texts = [
        "# Замена масла\n\nСлил старое масло, поменял фильтр, проверил уровень.\n",
        "# Цепь\n\nСмазка цепи, натяжение, износ звёзд, пробег между обслуживанием.\n",
        "# Колодки\n\nЗамена тормозных колодок, прокачка тормозов, диск.\n",
        "# Подготовка к сезону\n\nАккумулятор, свечи, карбюратор, резина.\n",
    ]
    return [
        make_note(f"мото_{index}.md", text, "Хобби/Мотоциклы")
        for index, text in enumerate(texts)
    ]


def filed_aquarium_notes() -> list[ParsedNote]:
    """Заметки, уже разложенные в «Хобби/Аквариум»."""
    texts = [
        "# Подмена воды\n\nЕженедельная подмена, сифонка грунта, отстоянная вода.\n",
        "# Фильтр\n\nВнешний фильтр, промывка губки, производительность.\n",
        "# Водоросли\n\nБорьба с водорослями, нитраты, освещение, растения.\n",
    ]
    return [
        make_note(f"аква_{index}.md", text, "Хобби/Аквариум")
        for index, text in enumerate(texts)
    ]


def test_neighbours_teach_personal_category() -> None:
    """Заметка без единого слова «мотоцикл» опознаётся по соседям."""
    notes = [
        *filed_moto_notes(),
        *filed_aquarium_notes(),
        make_note("Новая запись.md", "# Обслуживание\n\nПроверил натяжение цепи и уровень масла.\n"),
    ]
    decision = classify(notes)["Новая запись.md"]
    assert decision.status is Status.SORTED
    assert decision.category == "Хобби/Мотоциклы"


def test_neighbours_distinguish_sibling_categories() -> None:
    """Соседи должны разводить две личные подкатегории одного родителя."""
    notes = [
        *filed_moto_notes(),
        *filed_aquarium_notes(),
        make_note("вода.md", "# Уход\n\nПодмена воды, промывка губки фильтра, нитраты.\n"),
    ]
    assert classify(notes)["вода.md"].category == "Хобби/Аквариум"


def test_single_filed_note_does_not_capture_everything() -> None:
    """Категория с одной заметкой не должна притягивать всё похожее на неё.

    Одного соседа мало для вывода — голос демпфируется числом заметок.
    """
    notes = [
        make_note("одна.md", "# Ремонт\n\nШтукатурка стен, плитка, розетки.\n", "Дом/Ремонт"),
        *filed_moto_notes(),
        make_note("похожая.md", "# Стены\n\nШтукатурка и плитка в ванной.\n"),
    ]
    decision = classify(notes, knn_min_support=5)["похожая.md"]
    scores = {item.category: item.score for item in decision.candidates}
    assert scores.get("Дом/Ремонт", 0.0) < 0.9


def test_wiki_link_votes_for_target_category() -> None:
    """Ссылка на уже разложенную заметку — голос за её категорию."""
    notes = [
        *filed_moto_notes(),
        *filed_aquarium_notes(),
        make_note(
            "непонятная.md",
            "# Итоги\n\nСделал что планировал.\n\nСм. [[мото_0]] и [[мото_1]].\n",
        ),
    ]
    decision = classify(notes)["непонятная.md"]
    assert decision.category == "Хобби/Мотоциклы"


def cold_vault_notes() -> list[ParsedNote]:
    """Хранилище, где всё лежит в корне: категории пусты, разметки нет.

    Часть заметок названа прямо («Мотоцикл ...») — их опознает первый проход
    по названию каталога. Остальные не содержат имени категории вовсе.
    """
    named = [
        ("мотоцикл обслуживание.md", "# Мотоцикл\n\nОбслуживание мотоцикла, цепь, масло, пробег.\n"),
        ("мотоцикл сезон.md", "# Мотоцикл весной\n\nМотоцикл после зимы: свечи, аккумулятор, резина.\n"),
        ("мотоцикл тормоза.md", "# Мотоцикл: тормоза\n\nКолодки мотоцикла, прокачка, диск.\n"),
        ("аквариум запуск.md", "# Аквариум\n\nЗапуск аквариума, грунт, фильтр, растения.\n"),
        ("аквариум вода.md", "# Аквариум: вода\n\nПодмена воды в аквариуме, нитраты, сифонка.\n"),
        ("аквариум свет.md", "# Аквариум: свет\n\nОсвещение аквариума, водоросли, режим.\n"),
    ]
    unnamed = [
        ("смазка и уровень.md", "# Обслуживание\n\nСмазал цепь, проверил уровень масла и пробег.\n"),
        ("промывка и грунт.md", "# Уход\n\nПромыл губку, проверил нитраты, сифонил грунт.\n"),
    ]
    return [make_note(name, text) for name, text in named + unnamed]


def test_self_training_helps_cold_vault() -> None:
    """В хранилище, где ничего не разложено, второй проход учится на первом."""
    notes = cold_vault_notes()

    cold = classify(notes, self_training_rounds=0)
    warm = classify(notes, self_training_rounds=1)

    assert cold["смазка и уровень.md"].status is Status.UNCERTAIN
    assert warm["смазка и уровень.md"].category == "Хобби/Мотоциклы"
    assert warm["промывка и грунт.md"].category == "Хобби/Аквариум"


def test_self_training_can_be_disabled() -> None:
    """Отключение самообучения — предсказуемый и доступный вариант."""
    notes = cold_vault_notes()
    without = classify(notes, self_training_rounds=0)
    assert without["промывка и грунт.md"].status is Status.UNCERTAIN


def test_self_training_learns_only_from_confident_decisions() -> None:
    """Сомнительные решения не должны размножать собственную ошибку."""
    notes = [
        make_note("мотоцикл.md", "# Мотоцикл\n\nЦепь, масло, пробег.\n"),
        make_note("ерунда.md", "# Ерунда\n\nКвазиморфный субстрат бламинирует.\n"),
    ]
    decisions = classify(notes, self_training_rounds=2)
    assert decisions["ерунда.md"].status is Status.UNCERTAIN
