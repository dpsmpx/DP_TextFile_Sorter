"""Командный интерфейс и оркестрация всего конвейера сортировки."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import __version__
from .classifier import available_classifiers, create_classifier
from .classifier.base import Classifier
from .config import (
    Config,
    ConfigError,
    apply_config_data,
    find_config_file,
    load_config_file,
    save_config,
    validate,
)
from .decision import decide, resolve_destination
from .file_manager import PlacementResult, load_manifest, place_note, save_manifest
from .logging_setup import log_result, setup_logging
from .markdown_parser import parse_note
from .models import (
    Category,
    CopyOutcome,
    Decision,
    NoteRecord,
    ParsedNote,
    SORTED_DIR_NAME,
    Status,
)
from .report import Reporter
from .scanner import read_note_text, scan_tree
from .structure import (
    build_categories,
    build_category_profiles,
    count_unused_directories,
    ensure_sorted_dir,
    mirror_structure,
    prune_empty_directories,
)

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_WITH_ERRORS = 2


# ----------------------------------------------------------------------
# Разбор аргументов
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="sorter.py",
        description=(
            "Раскладывает копии Markdown-заметок по существующей структуре каталогов, "
            "используя её как систему категорий. Исходные файлы не изменяются."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"md_sorter {__version__}")
    paths = parser.add_argument_group("пути")
    paths.add_argument("--inbox", type=Path, default=None,
                       help="каталог с несортированными заметками (по умолчанию — текущий)")
    paths.add_argument("--vault", type=Path, default=None,
                       help="корень хранилища, откуда берётся структура категорий "
                            "(по умолчанию — сам inbox; для схемы «ХРАНИЛИЩЕ/INBOX» "
                            "укажите ..)")
    paths.add_argument("--output", type=Path, default=None,
                       help="куда складывать результат (по умолчанию <inbox>/Sorted_md_files)")
    paths.add_argument("--root", type=Path, default=None,
                       help="устаревший синоним --inbox")
    paths.add_argument("--save-config", nargs="?", const="", default=None, metavar="PATH",
                       help="сохранить пути и настройки в файл и больше их не вводить "
                            "(по умолчанию <inbox>/md_sorter.toml)")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="только показать план, ничего не копировать")
    parser.add_argument("--evaluate", action="store_true",
                        help="измерить качество на уже разложенных заметках "
                             "этого хранилища и выйти")

    output = parser.add_argument_group("вывод")
    output.add_argument("-v", "--verbose", action="store_true", default=None, help="объяснять каждое решение")
    output.add_argument("--debug", action="store_true", default=None,
                        help="показывать внутренние веса и отладку")
    output.add_argument("-q", "--quiet", action="store_true", default=None,
                        help="только предупреждения и итог")
    output.add_argument("--log-file", type=Path, default=None, help="дублировать лог в файл")
    color = output.add_mutually_exclusive_group()
    color.add_argument("--color", dest="color", action="store_true", default=None, help="включить цвет")
    color.add_argument("--no-color", dest="color", action="store_false", help="отключить цвет")

    algo = parser.add_argument_group("классификация")
    algo.add_argument("--classifier", choices=available_classifiers(), default=None,
                      help="алгоритм классификации")
    algo.add_argument("--threshold", type=float, default=None, help="минимальный score для уверенного решения")
    algo.add_argument("--margin", type=float, default=None, help="минимальный отрыв лидера от второго места")
    algo.add_argument("--uncertain-strategy",
                      choices=("review", "ancestor", "root", "skip", "best"), default=None,
                      help="что делать с сомнительными заметками "
                           "(best — всё равно положить к лучшему кандидату)")
    algo.add_argument("--dominance", type=float, default=None,
                      help="во сколько раз лидер должен обойти соперника")
    algo.add_argument("--min-evidence", type=float, default=None,
                      help="минимальная оценка для относительного правила приёма")
    algo.add_argument("--knn-neighbors", type=int, default=None,
                      help="сколько соседних разложенных заметок голосует")
    algo.add_argument("--self-training-rounds", type=int, default=None,
                      help="проходов самообучения на уверенных решениях (0 — отключить)")
    algo.add_argument("--self-training-min-score", type=float, default=None,
                      help="минимальная оценка решения, чтобы стать обучающим примером")
    algo.add_argument("--top-candidates", type=int, default=None, help="сколько вариантов показывать")
    algo.add_argument("--embedding-model", default=None, help="модель для классификатора embeddings")

    scan = parser.add_argument_group("сканирование")
    scan.add_argument("--ignore", action="append", default=None, metavar="DIR",
                      help="дополнительный каталог-исключение (можно повторять)")
    scan.add_argument("--include-hidden", action="store_true", default=None, help="учитывать скрытые файлы")
    scan.add_argument("--follow-symlinks", action="store_true", default=None, help="следовать по symlink")
    scan.add_argument("--max-file-size", type=int, default=None, help="предел размера файла в байтах")
    scan.add_argument("--max-analysis-chars", type=int, default=None, help="сколько символов анализировать")
    scan.add_argument("--jobs", type=int, default=None, help="процессов для разбора (0 — авто)")

    write = parser.add_argument_group("запись")
    write.add_argument("--manifest", dest="manifest", action="store_true", default=None,
                       help="вести файл состояния Sorted_md_files/.md_sorter_manifest.json")
    write.add_argument("--no-manifest", dest="manifest", action="store_false", help="не вести файл состояния")
    write.add_argument("--move", dest="copy_mode", action="store_const", const="move", default=None,
                       help="переносить, а не копировать (требует --allow-move)")
    write.add_argument("--allow-move", action="store_true", default=None,
                       help="подтвердить перенос исходных файлов")
    write.add_argument("--keep-empty-dirs", dest="prune_empty", action="store_false", default=None,
                       help="не удалять из результата папки, оставшиеся пустыми")
    write.add_argument("--prune-empty-dirs", dest="prune_empty", action="store_true",
                       help="удалять пустые папки результата (включено по умолчанию)")

    config_group = parser.add_argument_group("конфигурация")
    config_group.add_argument("--config", type=Path, default=None, help="путь к файлу конфигурации")
    config_group.add_argument("--no-config", action="store_true", help="игнорировать md_sorter.toml")

    return parser


def _resolve(path: Path) -> Path:
    """Приводит путь к абсолютному виду, разворачивая ``~`` и ``..``."""
    try:
        return path.expanduser().resolve()
    except OSError as exc:  # pragma: no cover - недоступный путь
        raise ConfigError(f"Не удалось разобрать путь {path}: {exc}") from exc


def build_config(args: argparse.Namespace) -> Config:
    """Собирает конфигурацию: значения по умолчанию, файл, аргументы CLI."""
    cli_inbox = args.inbox or args.root
    config = Config(inbox=_resolve(cli_inbox or Path.cwd()))

    if not args.no_config:
        search_from = [config.inbox]
        if args.vault is not None:
            search_from.append(_resolve(args.vault))
        config_path = args.config or find_config_file(*search_from)
        if config_path is not None:
            if not config_path.is_file():
                raise ConfigError(f"Файл конфигурации не найден: {config_path}")
            apply_config_data(config, load_config_file(config_path))

    # Пути из командной строки перекрывают сохранённые в файле.
    if cli_inbox is not None:
        config.inbox = cli_inbox
    if args.vault is not None:
        config.vault = args.vault
    if args.output is not None:
        config.output = args.output

    config.inbox = _resolve(config.inbox)
    config.vault = _resolve(config.vault) if config.vault is not None else None
    config.output = _resolve(config.output) if config.output is not None else None
    if config.vault == config.inbox:
        config.vault = None  # раздельной схемы нет, работаем по-старому

    # Аргументы CLI имеют приоритет над файлом конфигурации; ``None`` означает
    # «параметр не задан», поэтому все флаги объявлены с ``default=None``.
    overridable = (
        "dry_run", "verbose", "debug", "quiet", "log_file", "color", "classifier",
        "threshold", "margin", "uncertain_strategy", "top_candidates", "embedding_model",
        "include_hidden", "follow_symlinks", "max_file_size", "max_analysis_chars",
        "jobs", "manifest", "copy_mode", "allow_move", "prune_empty",
        "dominance", "min_evidence", "knn_neighbors", "self_training_rounds",
        "self_training_min_score",
    )
    for name in overridable:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)

    if args.ignore:
        config.ignored_directories = tuple({*config.ignored_directories, *args.ignore})

    validate(config)
    return config


# ----------------------------------------------------------------------
# Разбор заметок
# ----------------------------------------------------------------------
def _parse_one(record: NoteRecord, config: Config) -> tuple[ParsedNote | None, str]:
    """Читает и разбирает одну заметку; ошибки возвращаются, а не бросаются."""
    try:
        text, truncated = read_note_text(record.path, config)
    except OSError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - один битый файл не должен ронять запуск
        return None, f"неожиданная ошибка чтения: {exc}"
    try:
        return parse_note(record, text, config.weights, truncated=truncated), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"неожиданная ошибка разбора: {exc}"


#: Конфигурация рабочего процесса; заполняется один раз при его старте.
_WORKER_CONFIG: Config | None = None


def _init_worker(config: Config) -> None:
    """Инициализатор пула: сохраняет конфигурацию в процессе-работнике."""
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _parse_worker(record: NoteRecord) -> tuple[ParsedNote | None, str]:
    """Обёртка верхнего уровня для :class:`ProcessPoolExecutor`."""
    if _WORKER_CONFIG is None:  # pragma: no cover - защита от неверного использования
        raise RuntimeError("Пул процессов не инициализирован")
    return _parse_one(record, _WORKER_CONFIG)


def parse_notes(
    records: Sequence[NoteRecord],
    config: Config,
    logger: logging.Logger,
) -> tuple[list[ParsedNote], list[tuple[NoteRecord, str]]]:
    """Разбирает все заметки, при необходимости — в несколько процессов."""
    jobs = config.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1 if len(records) >= config.parallel_threshold else 1
    jobs = max(1, min(jobs, len(records) or 1))

    parsed: list[ParsedNote] = []
    failures: list[tuple[NoteRecord, str]] = []

    if jobs > 1:
        try:
            with ProcessPoolExecutor(
                max_workers=jobs, initializer=_init_worker, initargs=(config,)
            ) as pool:
                results = list(pool.map(_parse_worker, records, chunksize=16))
        except Exception as exc:  # noqa: BLE001 - multiprocessing доступно не везде
            logger.warning("Параллельный разбор недоступен (%s), продолжаем в один поток", exc)
            results = [_parse_one(record, config) for record in records]
    else:
        results = [_parse_one(record, config) for record in records]

    for record, (note, error) in zip(records, results):
        if note is None:
            failures.append((record, error))
            continue
        if record.size > config.max_file_size:
            logger.warning(
                "Файл больше max_file_size (%d Б), проанализировано только начало: %s",
                record.size,
                record.relative_path,
            )
        elif note.truncated:
            logger.debug("Текст обрезан до max_analysis_chars: %s", record.relative_path)
        parsed.append(note)

    return parsed, failures


# ----------------------------------------------------------------------
# Основной конвейер
# ----------------------------------------------------------------------
def _validate_directory(path: Path, label: str, *, writable: bool) -> None:
    """Проверяет, что каталог пригоден для работы."""
    if not path.exists():
        raise ConfigError(f"{label} не существует: {path}")
    if not path.is_dir():
        raise ConfigError(f"{label} — это не каталог: {path}")
    if SORTED_DIR_NAME in path.parts:
        raise ConfigError(
            f"{label} находится внутри {SORTED_DIR_NAME}: "
            "это привело бы к повторной сортировке результата"
        )
    if not os.access(path, os.R_OK):
        raise ConfigError(f"Нет прав на чтение: {path}")
    if writable and not os.access(path, os.W_OK):
        raise ConfigError(f"Нет прав на запись: {path}")


def _validate_paths(config: Config) -> None:
    """Проверяет inbox и корень хранилища перед началом работы."""
    _validate_directory(config.inbox, "Каталог inbox", writable=False)
    if config.split_layout:
        _validate_directory(config.vault_dir, "Корень хранилища", writable=False)
    if not config.dry_run:
        target_parent = config.sorted_dir.parent
        _validate_directory(
            target_parent if target_parent.exists() else config.inbox,
            "Каталог для результата",
            writable=True,
        )


def evaluate(config: Config, logger: logging.Logger) -> int:
    """Измеряет качество классификации на уже разложенных заметках хранилища.

    Заметки, которые пользователь сам положил в категории, — готовая разметка:
    можно проверить, угадывает ли программа его собственный выбор, и подобрать
    параметры под конкретное хранилище, ничего при этом не копируя.
    """
    _validate_paths(config)
    logger.info("Inbox: %s", config.inbox)
    if config.split_layout:
        logger.info("Хранилище: %s", config.vault_dir)
    logger.info("Проверка на уже разложенных заметках...")

    corpus = _collect(config, logger)
    categories_by_key = {category.key: category for category in corpus.categories}
    if not corpus.notes or not categories_by_key:
        logger.error("Недостаточно данных: нужны категории и разложенные по ним заметки")
        return EXIT_FATAL

    notes_by_category = build_category_profiles(
        corpus.categories, corpus.notes, config, filed_indices=corpus.filed_indices
    )

    # Проверяем на заметках, чью категорию выбрал сам пользователь.
    known = [
        (index, note)
        for index, note in enumerate(corpus.notes)
        if note.record.source_dir.as_posix() in categories_by_key
        and (corpus.filed_indices is None or index in corpus.filed_indices)
    ]
    if not known:
        logger.error(
            "В категориях нет ни одной заметки — измерять не на чем. "
            "Разложите вручную хотя бы по нескольку заметок в каждую папку."
        )
        return EXIT_FATAL

    classifier = create_classifier(config)
    classifier.fit(corpus.notes, corpus.categories, notes_by_category)
    try:
        verdicts = {
            index: decide(
                note.record, classifier.rank(note, index), categories_by_key, config
            )
            for index, note in known
        }
    finally:
        classifier.close()

    total = accepted = correct = top1 = 0
    misses: dict[str, int] = {}
    for index, note in known:
        expected = note.record.source_dir.as_posix()
        verdict = verdicts[index]
        total += 1
        if verdict.candidates and verdict.candidates[0].category == expected:
            top1 += 1
        if verdict.status is Status.SORTED:
            accepted += 1
            if verdict.category == expected:
                correct += 1
            else:
                misses[expected] = misses.get(expected, 0) + 1

    log_result(logger, "Заметок с известной категорией: %d", total)
    log_result(logger, "Лучший кандидат совпал с вашим выбором: %.1f%%", 100.0 * top1 / total)
    log_result(logger, "Принято решений (полнота): %.1f%%", 100.0 * accepted / total)
    if accepted:
        log_result(logger, "Из принятых верно: %.1f%%", 100.0 * correct / accepted)

    if misses:
        logger.info("Категории с наибольшим числом расхождений:")
        for key, count in sorted(misses.items(), key=lambda item: -item[1])[:5]:
            logger.info("  %s — %d", key, count)

    if accepted < total * 0.9:
        logger.info(
            "Полноту можно поднять: --threshold пониже, --dominance ближе к 1.0 "
            "или --uncertain-strategy best"
        )
    return EXIT_OK


@dataclass(slots=True)
class Corpus:
    """Разобранное содержимое хранилища и inbox.

    Заметки хранилища — обучающий материал: они формируют профили категорий и
    участвуют в голосовании соседей, но сами не сортируются и не копируются.
    """

    categories: list[Category]
    directories: list[PurePosixPath]
    notes: list[ParsedNote]
    targets: slice
    filed_indices: frozenset[int] | None
    failures: list[tuple[NoteRecord, str]]
    errors: int

    @property
    def target_notes(self) -> list[ParsedNote]:
        """Заметки, которые нужно разложить."""
        return self.notes[self.targets]

    @property
    def target_offset(self) -> int:
        """Индекс первой сортируемой заметки в общем списке."""
        return self.targets.start or 0


def _collect(config: Config, logger: logging.Logger) -> Corpus:
    """Обходит хранилище и inbox, разбирает найденные заметки."""
    # Редкий случай: хранилище лежит внутри inbox — тогда его заметки нельзя
    # принимать за несортированные.
    nested_vault = _is_strictly_inside(config.vault_dir, config.inbox)
    inbox_scan = scan_tree(
        config.inbox,
        config,
        logger,
        excluded_paths=[config.vault_dir] if nested_vault else [],
    )

    if not config.split_layout:
        parsed, failures = parse_notes(inbox_scan.notes, config, logger)
        return Corpus(
            categories=build_categories(inbox_scan.directories),
            directories=inbox_scan.directories,
            notes=parsed,
            targets=slice(0, len(parsed)),
            filed_indices=None,
            failures=failures,
            errors=inbox_scan.errors,
        )

    vault_scan = scan_tree(
        config.vault_dir, config, logger, excluded_paths=[config.inbox]
    )
    logger.info(
        "Структура категорий берётся из %s (заметок-образцов: %d)",
        config.vault_dir,
        len(vault_scan.notes),
    )
    training, training_failures = parse_notes(vault_scan.notes, config, logger)
    targets, target_failures = parse_notes(inbox_scan.notes, config, logger)

    return Corpus(
        categories=build_categories(vault_scan.directories),
        directories=vault_scan.directories,
        notes=[*training, *targets],
        targets=slice(len(training), len(training) + len(targets)),
        filed_indices=frozenset(range(len(training))),
        failures=[*training_failures, *target_failures],
        errors=vault_scan.errors + inbox_scan.errors,
    )


def _is_strictly_inside(path: Path, parent: Path) -> bool:
    """True, если ``path`` лежит внутри ``parent`` и не совпадает с ним."""
    if path == parent:
        return False
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def run(config: Config, logger: logging.Logger) -> int:
    """Выполняет полный цикл: обход, классификацию, размещение, отчёт."""
    _validate_paths(config)
    reporter = Reporter(config, logger)
    reporter.announce_root(str(config.inbox))

    logger.info("Searching Markdown files...")
    corpus = _collect(config, logger)

    if not config.dry_run:
        ensure_sorted_dir(config, logger)

    categories = corpus.categories
    reporter.announce_scan(len(corpus.target_notes), len(categories))

    # Копия структуры создаётся независимо от того, нашлись ли заметки:
    # это отдельный результат работы, а не побочный эффект классификации.
    mapping = mirror_structure(corpus.directories, config, logger)

    if not corpus.target_notes:
        logger.info("Markdown-файлы не найдены, работа завершена")
        if config.prune_empty:
            # Раскладывать нечего, поэтому пусто всё зеркало целиком.
            reporter.count_pruned(prune_empty_directories(config, logger))
        reporter.summary(corpus.errors)
        return EXIT_WITH_ERRORS if corpus.errors else EXIT_OK

    parsed = corpus.notes
    failures = corpus.failures
    notes_by_category = build_category_profiles(
        categories, parsed, config, filed_indices=corpus.filed_indices
    )
    categories_by_key = {category.key: category for category in categories}

    classifier = create_classifier(config)
    classifier.fit(parsed, categories, notes_by_category)

    targets = corpus.target_notes
    offset = corpus.target_offset

    manifest = load_manifest(config, logger) if config.manifest else {}
    new_manifest: dict[str, dict[str, object]] = {}
    decisions: list[Decision] = []
    # Занятые за этот запуск имена: в --dry-run файловая система не меняется,
    # поэтому конфликты нужно отслеживать отдельно.
    reserved: set[str] = set()

    for record, error in failures:
        decision = Decision(record=record, status=Status.ERROR, error=error)
        reporter.report_decision(decision)
        decisions.append(decision)

    total = len(targets)
    try:
        verdicts = _decide_all(targets, offset, classifier, categories_by_key, config)
        verdicts = _self_train(
            targets, offset, classifier, categories_by_key, config, logger, verdicts
        )
        _place_all(
            parsed=targets,
            verdicts=verdicts,
            mapping=mapping,
            config=config,
            logger=logger,
            reporter=reporter,
            manifest=manifest,
            new_manifest=new_manifest,
            decisions=decisions,
            reserved=reserved,
            total=total,
        )
    finally:
        classifier.close()

    if config.dry_run:
        reporter.preview([item for item in decisions if item.status is not Status.ERROR])
        if config.prune_empty:
            reporter.preview_pruning(
                count_unused_directories(mapping, _used_directories(decisions))
            )
    else:
        if config.manifest:
            save_manifest(config, new_manifest, logger, version=__version__)
        if config.prune_empty:
            reporter.count_pruned(prune_empty_directories(config, logger))

    stats = reporter.summary(corpus.errors)
    return EXIT_WITH_ERRORS if stats.errors else EXIT_OK



def _decide_all(
    targets: Sequence[ParsedNote],
    offset: int,
    classifier: Classifier,
    categories_by_key: dict[str, Category],
    config: Config,
) -> list[Decision]:
    """Классифицирует заметки inbox, ничего не копируя и не печатая.

    Args:
        offset: индекс первой сортируемой заметки в общем корпусе. При
            раздельной схеме перед ними идут заметки хранилища, и классификатор
            адресует их по сквозному номеру.
    """
    return [
        decide(note.record, classifier.rank(note, offset + index), categories_by_key, config)
        for index, note in enumerate(targets)
    ]


def _self_train(
    targets: Sequence[ParsedNote],
    offset: int,
    classifier: Classifier,
    categories_by_key: dict[str, Category],
    config: Config,
    logger: logging.Logger,
    verdicts: list[Decision],
) -> list[Decision]:
    """Возвращает уверенные решения в обучающий набор и переклассифицирует.

    В хранилище, где пользователь ещё ничего не разложил, категории описаны
    только своими названиями. Первый проход опознаёт то, что названо прямо;
    эти заметки становятся примерами, по которым второй проход узнаёт всё
    остальное. Обучение идёт только на уверенных решениях, поэтому сомнительные
    случаи не размножают собственную ошибку.
    """
    for round_number in range(1, max(0, config.self_training_rounds) + 1):
        seeds = {
            offset + index: verdict.category
            for index, verdict in enumerate(verdicts)
            if verdict.status is Status.SORTED
            and verdict.category
            and verdict.score >= config.self_training_min_score
        }
        if not seeds:
            break
        added = classifier.learn_from(seeds)
        if added == 0:
            break
        logger.debug(
            "Проход самообучения %d: в обучающий набор добавлено %d заметок",
            round_number,
            added,
        )
        verdicts = _decide_all(targets, offset, classifier, categories_by_key, config)
    return verdicts


def _place_all(
    *,
    parsed: Sequence[ParsedNote],
    verdicts: Sequence[Decision],
    mapping: dict[str, PurePosixPath],
    config: Config,
    logger: logging.Logger,
    reporter: Reporter,
    manifest: dict[str, dict[str, object]],
    new_manifest: dict[str, dict[str, object]],
    decisions: list[Decision],
    reserved: set[str],
    total: int,
) -> None:
    """Размещает заметки согласно решениям, наполняя отчёт и манифест."""
    for index, (note, decision) in enumerate(zip(parsed, verdicts), start=1):
        target = resolve_destination(decision, mapping, config)
        reporter.report_decision(decision)

        if target is None:
            decision.outcome = None
        else:
            result = _place(note.record, target, config, logger, manifest, reserved)
            decision.destination = result.destination
            decision.outcome = result.outcome
            decision.error = result.error
        reporter.report_placement(decision)

        if config.manifest and decision.destination is not None:
            new_manifest[note.record.relative_path.as_posix()] = {
                "destination": decision.destination.as_posix(),
                "category": decision.category or "",
                "score": decision.score,
                "size": note.record.size,
                "mtime": note.record.mtime,
                "status": decision.status.value,
            }

        decisions.append(decision)
        reporter.progress(index, total)


def _used_directories(decisions: Sequence[Decision]) -> set[str]:
    """Каталоги результата, в которые действительно что-то попало."""
    used: set[str] = set()
    for decision in decisions:
        if decision.destination is None:
            continue
        parent = decision.destination.parent.as_posix()
        used.add("" if parent == "." else parent)
    return used


def _place(
    record: NoteRecord,
    target: PurePosixPath,
    config: Config,
    logger: logging.Logger,
    manifest: dict[str, dict[str, object]],
    reserved: set[str],
) -> PlacementResult:
    """Размещает заметку, по возможности избегая лишнего хэширования.

    Если манифест утверждает, что файл уже лежит по тому же адресу и с тех пор
    не менялся, копирование пропускается без чтения содержимого.
    """
    key = record.relative_path.as_posix()
    entry = manifest.get(key)
    if entry and not config.dry_run:
        destination = str(entry.get("destination", ""))
        same_place = destination.rsplit("/", 1)[0] if "/" in destination else "."
        expected_dir = target.as_posix()
        recorded_mtime = entry.get("mtime")
        if (
            destination
            and same_place == expected_dir
            and entry.get("size") == record.size
            and isinstance(recorded_mtime, (int, float))
            and float(recorded_mtime) == record.mtime
            and (config.sorted_dir / destination).is_file()
        ):
            logger.debug("Не изменился с прошлого запуска, пропуск: %s", key)
            reserved.add(destination)
            return PlacementResult(CopyOutcome.IDENTICAL, PurePosixPath(destination))

    return place_note(record, target, config, logger, reserved=reserved)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: разбирает аргументы, настраивает лог и запускает конвейер."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = build_config(args)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FATAL

    logger = setup_logging(
        verbose=config.verbose,
        debug=config.debug,
        quiet=config.quiet,
        log_file=config.log_file,
        color=config.color,
    )

    if args.save_config is not None:
        target = Path(args.save_config) if args.save_config else config.inbox / "md_sorter.toml"
        try:
            written = save_config(config, target.expanduser())
        except OSError as exc:
            logger.error("Не удалось сохранить настройки: %s", exc)
            return EXIT_FATAL
        log_result(logger, "Настройки сохранены: %s", written)
        log_result(logger, "Дальше достаточно запускать: python sorter.py")

    try:
        if args.evaluate:
            return evaluate(config, logger)
        return run(config, logger)
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_FATAL
    except KeyboardInterrupt:  # pragma: no cover - интерактивное прерывание
        logger.warning("Прервано пользователем")
        return EXIT_FATAL
    except (RuntimeError, ValueError) as exc:
        # Например, выбран классификатор, зависимости которого не установлены.
        logger.error("%s", exc)
        return EXIT_FATAL
    except Exception as exc:  # noqa: BLE001 - последний рубеж
        logger.error("Непредвиденная ошибка: %s", exc, exc_info=config.debug)
        return EXIT_FATAL
