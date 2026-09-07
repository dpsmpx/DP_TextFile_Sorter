#!/usr/bin/env python3
"""Стенд для замера качества классификации.

Создаёт хранилище, похожее на настоящее: личные категории пользователя,
которых нет во встроенном словаре, часть заметок уже разложена по папкам,
остальные лежат в корне. Для корневых заметок известна правильная категория,
поэтому можно измерить и полноту (сколько вообще классифицировано), и
точность (сколько классифицировано верно).

Использование::

    python tools/benchmark.py                    # текущий алгоритм
    python tools/benchmark.py --threshold 0.1    # с другими параметрами
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from md_sorter.classifier import create_classifier  # noqa: E402
from md_sorter.config import Config  # noqa: E402
from md_sorter.decision import decide  # noqa: E402
from md_sorter.logging_setup import setup_logging  # noqa: E402
from md_sorter.markdown_parser import parse_note  # noqa: E402
from md_sorter.models import Status  # noqa: E402
from md_sorter.scanner import read_note_text, scan_tree  # noqa: E402
from md_sorter.structure import build_categories, build_category_profiles  # noqa: E402

# Личные категории пользователя. Намеренно почти нет пересечений со встроенным
# словарём: именно так выглядит настоящее хранилище.
TOPICS: dict[str, dict[str, list[str]]] = {
    "Работа/Отчёты": {
        "terms": ["отчёт", "квартал", "показатели", "сводка", "презентация", "дедлайн", "руководитель"],
        "titles": ["Отчёт за квартал", "Сводка по показателям", "Подготовка презентации", "Итоги месяца"],
    },
    "Работа/Совещания": {
        "terms": ["совещание", "протокол", "повестка", "участники", "решили", "созвон", "планёрка"],
        "titles": ["Протокол планёрки", "Повестка созвона", "Итоги совещания", "Заметки со встречи"],
    },
    "Хобби/Мотоциклы": {
        "terms": ["мотоцикл", "цепь", "масло", "колодки", "пробег", "шлем", "карбюратор", "сцепление"],
        "titles": ["Замена масла", "Обслуживание цепи", "Тормозные колодки", "Подготовка к сезону"],
    },
    "Хобби/Аквариум": {
        "terms": ["аквариум", "рыбки", "фильтр", "водоросли", "грунт", "подмена воды", "нитраты"],
        "titles": ["Запуск аквариума", "Подмена воды", "Борьба с водорослями", "Выбор фильтра"],
    },
    "Здоровье/Тренировки": {
        "terms": ["тренировка", "подход", "разминка", "приседания", "гантели", "растяжка", "пульс"],
        "titles": ["План тренировок", "Программа на неделю", "Разминка перед залом", "Работа с гантелями"],
    },
    "Здоровье/Питание": {
        "terms": ["питание", "белок", "калории", "завтрак", "овощи", "рацион", "витамины"],
        "titles": ["Рацион на неделю", "Подсчёт калорий", "Завтраки", "Белок в рационе"],
    },
    "Учёба/Матанализ": {
        "terms": ["интеграл", "производная", "предел", "теорема", "ряд", "функция", "доказательство"],
        "titles": ["Производные", "Определённый интеграл", "Ряды Тейлора", "Пределы функций"],
    },
    "Учёба/История": {
        "terms": ["век", "реформа", "император", "династия", "летопись", "восстание", "договор"],
        "titles": ["Реформы XIX века", "Смутное время", "Династия и наследование", "Летописные своды"],
    },
    "Дом/Ремонт": {
        "terms": ["ремонт", "штукатурка", "плитка", "розетка", "шпаклёвка", "ламинат", "плинтус"],
        "titles": ["Ремонт ванной", "Укладка плитки", "Перенос розеток", "Выравнивание стен"],
    },
    "Дом/Рецепты": {
        "terms": ["рецепт", "тесто", "духовка", "соус", "обжарить", "специи", "порция"],
        "titles": ["Тесто для пирога", "Соус к пасте", "Запекание в духовке", "Домашний хлеб"],
    },
    "Проекты/Сайт": {
        "terms": ["сайт", "вёрстка", "макет", "домен", "хостинг", "форма", "шапка"],
        "titles": ["Макет главной", "Настройка домена", "Форма обратной связи", "Вёрстка шапки"],
    },
    "Проекты/Книга": {
        "terms": ["глава", "сюжет", "персонаж", "черновик", "редактура", "рукопись", "эпизод"],
        "titles": ["Черновик главы", "Арка персонажа", "План сюжета", "Правки рукописи"],
    },
}

FILLER = (
    "нужно нельзя обычно затем далее пример вариант случай момент причина "
    "смысл вопрос ответ список пункт запись мысль идея время место способ"
).split()


def _note_text(
    generator: random.Random,
    topic: str,
    length: str,
    *,
    with_tag: bool,
    link_target: str | None,
) -> str:
    """Собирает текст заметки заданной «плотности» признаков."""
    data = TOPICS[topic]
    title = generator.choice(data["titles"])
    terms = data["terms"]

    counts = {"короткая": (2, 4), "средняя": (5, 14), "длинная": (10, 40)}[length]
    topic_words = generator.choices(terms, k=counts[0])
    filler_words = generator.choices(FILLER, k=counts[1])
    body_words = topic_words + filler_words
    generator.shuffle(body_words)

    parts = []
    if with_tag:
        tag = topic.split("/")[-1].lower()
        parts.append(f"---\ntags:\n  - {tag}\n---\n")
    parts.append(f"# {title}\n")
    parts.append(" ".join(body_words) + ".\n")
    if link_target:
        parts.append(f"\nСм. также [[{link_target}]].\n")
    return "\n".join(parts)


def build_vault(root: Path, generator: random.Random, *, filed_per_topic: int = 6) -> dict[str, str]:
    """Создаёт хранилище и возвращает «правильные ответы» для корневых заметок."""
    for topic in TOPICS:
        (root / topic).mkdir(parents=True, exist_ok=True)

    truth: dict[str, str] = {}
    filed_titles: dict[str, list[str]] = {topic: [] for topic in TOPICS}

    # 1. Часть заметок пользователь уже разложил — это обучающий материал.
    for topic in TOPICS:
        for index in range(filed_per_topic):
            length = generator.choice(["средняя", "длинная", "средняя"])
            text = _note_text(generator, topic, length, with_tag=index < 2, link_target=None)
            name = f"{topic.split('/')[-1].lower()}_{index}.md"
            (root / topic / name).write_text(text, encoding="utf-8")
            filed_titles[topic].append(Path(name).stem)

    # 2. Остальные лежат в корне и подлежат сортировке.
    for index in range(240):
        topic = generator.choice(list(TOPICS))
        length = generator.choices(
            ["короткая", "средняя", "длинная"], weights=[0.35, 0.45, 0.20]
        )[0]
        with_tag = generator.random() < 0.25
        link = None
        if generator.random() < 0.20 and filed_titles[topic]:
            link = generator.choice(filed_titles[topic])

        # Имя файла у половины заметок бессодержательное — как в жизни.
        if generator.random() < 0.5:
            name = f"Заметка {index}.md"
        else:
            name = f"{generator.choice(TOPICS[topic]['titles'])} {index}.md"

        text = _note_text(generator, topic, length, with_tag=with_tag, link_target=link)
        (root / name).write_text(text, encoding="utf-8")
        truth[name] = topic

    return truth


def evaluate(root: Path, truth: dict[str, str], **overrides: object) -> dict[str, float]:
    """Прогоняет классификацию и считает полноту и точность."""
    config = Config(inbox=root)
    for key, value in overrides.items():
        setattr(config, key, value)

    logger = setup_logging(quiet=True, color=False)
    logger.disabled = True

    scan = scan_tree(root, config, logger)
    notes = []
    for record in scan.notes:
        text, truncated = read_note_text(record.path, config)
        notes.append(parse_note(record, text, config.weights, truncated=truncated))

    categories = build_categories(scan.directories)
    notes_by_category = build_category_profiles(categories, notes, config)
    by_key = {category.key: category for category in categories}

    classifier = create_classifier(config)
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

    total = classified = correct = top1_correct = 0
    for index, note in enumerate(notes):
        expected = truth.get(note.record.relative_path.as_posix())
        if expected is None:
            continue
        total += 1
        decision = verdicts[index]
        candidates = decision.candidates
        if candidates and candidates[0].category == expected:
            top1_correct += 1
        if decision.status is Status.SORTED:
            classified += 1
            if decision.category == expected:
                correct += 1

    return {
        "всего": total,
        "классифицировано": classified,
        "полнота, %": 100.0 * classified / max(1, total),
        "верно из классифицированных, %": 100.0 * correct / max(1, classified),
        "верно из всех, %": 100.0 * correct / max(1, total),
        "лучший кандидат верен, %": 100.0 * top1_correct / max(1, total),
    }


def main(argv: list[str] | None = None) -> int:
    """Создаёт стенд, прогоняет классификацию и печатает метрики."""
    parser = argparse.ArgumentParser(description="Замер качества классификации")
    parser.add_argument("--seed", type=int, default=20240907)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--keep", type=Path, default=None, help="сохранить хранилище")
    parser.add_argument(
        "--filed-per-topic",
        type=int,
        default=6,
        help="сколько заметок уже разложено по каждой категории (0 — холодный старт)",
    )
    args = parser.parse_args(argv)

    overrides = {
        key: value
        for key, value in (
            ("threshold", args.threshold),
            ("margin", args.margin),
            ("classifier", args.classifier),
        )
        if value is not None
    }

    target = args.keep or Path(tempfile.mkdtemp(prefix="md_sorter_bench_"))
    if target.exists() and args.keep:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        truth = build_vault(
            target, random.Random(args.seed), filed_per_topic=args.filed_per_topic
        )
        metrics = evaluate(target, truth, **overrides)
        width = max(len(name) for name in metrics)
        for name, value in metrics.items():
            print(f"{name:<{width}} : {value:7.1f}" if isinstance(value, float) else f"{name:<{width}} : {value:7d}")
    finally:
        if not args.keep:
            shutil.rmtree(target, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
