"""Построение системы категорий и зеркалирование структуры каталогов."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from .config import Config, ConfigError
from .lexicon import expand_terms
from .models import Category, ParsedNote, REVIEW_DIR_NAME, TokenBag
from .text import add_tokens, merge_bag, tokenize, tokenize_name

_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_INVALID_CHARS_RE = re.compile(r'[<>:"|?*\x00-\x1f]')

#: Во сколько раз суммарный вклад «своих» заметок может превышать вес
#: собственных признаков категории (имя, путь, лексикон, алиасы).
#: Ограничение не даёт каталогу с одной заметкой превратиться в детектор
#: дубликатов этой заметки.
_NOTE_MASS_RATIO = 1.0

#: Запасной предел массы, если у категории вообще нет собственных признаков.
_NOTE_MASS_FLOOR = 3.0


def _add_ranked_block(bag: TokenBag, terms: list[str], base: float) -> None:
    """Добавляет блок синонимов с убывающим по порядку весом.

    Два соображения:

    1. Суммарная масса блока растёт как ``sqrt(N)``, а не как ``N``. Длинный
       список синонимов иначе раздувал бы норму вектора категории и занижал
       близость к заметкам, совпавшим по названию.
    2. Внутри блока вес убывает как ``1/sqrt(i)``: списки в словаре и алиасы
       пользователя пишутся от самого характерного термина к менее значимым
       (``arch`` -> ``pacman``, ``aur``, ``yay``, ...).
    """
    if not terms:
        return
    scale = base / 2.0
    for position, term in enumerate(terms, start=1):
        bag[term] = bag.get(term, 0.0) + scale / (position**0.5)


def sanitize_component(name: str) -> str:
    """Приводит имя каталога к виду, допустимому на всех платформах."""
    cleaned = _INVALID_CHARS_RE.sub("_", name).rstrip(" .")
    if not cleaned:
        cleaned = "_"
    if cleaned.split(".")[0].lower() in _WINDOWS_RESERVED:
        cleaned = f"{cleaned}_"
    return cleaned


def build_categories(directories: list[PurePosixPath]) -> list[Category]:
    """Создаёт объекты категорий из относительных путей каталогов."""
    children: dict[str, list[str]] = defaultdict(list)
    for path in directories:
        parent = path.parent.as_posix()
        children["" if parent == "." else parent].append(path.name)

    categories: list[Category] = []
    for path in directories:
        if path.name == REVIEW_DIR_NAME:
            continue
        parent_key = "" if path.parent.as_posix() == "." else path.parent.as_posix()
        siblings = tuple(name for name in children[parent_key] if name != path.name)
        categories.append(
            Category(
                rel_path=path,
                name=path.name,
                parents=tuple(path.parent.parts) if parent_key else (),
                depth=len(path.parts),
                siblings=siblings,
            )
        )
    return categories


def build_category_profiles(
    categories: list[Category],
    notes: list[ParsedNote],
    config: Config,
) -> dict[str, list[int]]:
    """Наполняет профили категорий токенами.

    Источники: имя каталога, полный путь, встроенный лексикон, пользовательские
    алиасы и заметки, которые уже лежат в этом каталоге.

    Returns:
        Отображение «ключ категории -> индексы заметок, лежащих в ней»; оно
        нужно классификатору, чтобы исключить вклад самой заметки при её оценке.
    """
    weights = config.weights
    notes_by_dir: dict[str, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        notes_by_dir[note.record.source_dir.as_posix()].append(index)

    for category in categories:
        bag: TokenBag = {}
        name_tokens = tokenize_name(category.name)
        add_tokens(bag, name_tokens, weights.category_name)
        for part in category.rel_path.parent.parts:
            add_tokens(bag, tokenize_name(part), weights.category_path)

        _add_ranked_block(bag, expand_terms(name_tokens), weights.category_lexicon)

        alias_terms: list[str] = []
        for alias in config.aliases.get(category.key, ()):
            alias_terms.extend(tokenize(alias))
        _add_ranked_block(bag, alias_terms, weights.category_alias)

        category.tokens = bag

    for category in categories:
        indices = notes_by_dir.get(category.key, [])
        if not indices:
            continue
        factor = _effective_note_factor(category.tokens, [notes[i] for i in indices], config)
        category.note_factor = factor
        if factor <= 0.0:
            continue
        for index in indices:
            merge_bag(category.tokens, notes[index].tokens, factor)

    return {category.key: notes_by_dir.get(category.key, []) for category in categories}


def _effective_note_factor(
    identity: TokenBag,
    notes: list[ParsedNote],
    config: Config,
) -> float:
    """Вычисляет вес одной «своей» заметки в профиле категории.

    Вклад делится на ``sqrt(N)``, чтобы каталог с сотней заметок не заглушал
    собственное название, и дополнительно масштабируется так, чтобы суммарная
    масса заметок не превышала массу собственных признаков категории. Без этого
    ограничения каталог с единственной заметкой начинает притягивать её копии
    вместо тематически близких файлов.
    """
    if not notes:
        return 0.0
    base = config.weights.category_notes / (len(notes) ** 0.5)
    note_mass = sum(sum(note.tokens.values()) for note in notes) * base
    if note_mass <= 0.0:
        return 0.0
    identity_mass = sum(identity.values())
    allowed = max(identity_mass * _NOTE_MASS_RATIO, _NOTE_MASS_FLOOR)
    return base * min(1.0, allowed / note_mass)


def mirror_structure(
    directories: list[PurePosixPath],
    config: Config,
    logger: logging.Logger,
) -> dict[str, PurePosixPath]:
    """Создаёт в ``Sorted_md_files`` копию структуры каталогов ROOT.

    Копируются только каталоги; файлы не переносятся. Существующие каталоги
    не удаляются и не очищаются. В режиме ``--dry-run`` ничего не создаётся,
    отображение строится только в памяти.

    Returns:
        Отображение «ключ категории -> фактический относительный путь».
    """
    mapping: dict[str, PurePosixPath] = {}
    used: set[str] = set()

    for path in sorted(directories, key=lambda item: (len(item.parts), item.as_posix())):
        parent_key = "" if path.parent.as_posix() == "." else path.parent.as_posix()
        parent_path = mapping.get(parent_key, PurePosixPath(parent_key)) if parent_key else None

        safe_name = sanitize_component(path.name)
        target = PurePosixPath(safe_name) if parent_path is None else parent_path / safe_name
        if target.as_posix() in used:
            target = _unique_sibling(target, used)
        if target.name != path.name:
            logger.warning(
                "Имя каталога изменено для совместимости с файловой системой: %s -> %s",
                path.as_posix(),
                target.as_posix(),
            )
        used.add(target.as_posix())
        mapping[path.as_posix()] = target

    if config.dry_run:
        logger.info("Каталоги не создаются (--dry-run): %d шт.", len(mapping))
        return mapping

    created = 0
    for target in mapping.values():
        destination = config.sorted_dir / Path(*target.parts)
        try:
            destination.mkdir(parents=True, exist_ok=True)
            created += 1
        except OSError as exc:
            logger.error("Не удалось создать каталог %s: %s", target.as_posix(), exc)
    logger.debug("Каталогов подготовлено: %d", created)
    return mapping


def _unique_sibling(target: PurePosixPath, used: set[str]) -> PurePosixPath:
    """Подбирает свободное имя при коллизии после нормализации."""
    for index in range(2, 1000):
        candidate = target.with_name(f"{target.name}_{index}")
        if candidate.as_posix() not in used:
            return candidate
    raise RuntimeError(f"Не удалось подобрать уникальное имя для {target}")


def ensure_sorted_dir(config: Config, logger: logging.Logger) -> None:
    """Создаёт ``Sorted_md_files``, не трогая уже существующее содержимое."""
    if config.dry_run:
        return
    try:
        config.sorted_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Не удалось создать {config.sorted_dir}: {exc}") from exc
    logger.debug("Каталог результата готов: %s", config.sorted_dir)
