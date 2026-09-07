"""Командный интерфейс и оркестрация всего конвейера сортировки."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from . import __version__
from .classifier import available_classifiers, create_classifier
from .classifier.base import Classifier
from .config import Config, ConfigError, apply_config_data, find_config_file, load_config_file, validate
from .decision import decide, resolve_destination
from .file_manager import PlacementResult, load_manifest, place_note, save_manifest
from .logging_setup import setup_logging
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
from .structure import build_categories, build_category_profiles, ensure_sorted_dir, mirror_structure

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
    parser.add_argument("--root", type=Path, default=None, help="каталог хранилища (по умолчанию — текущий)")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="только показать план, ничего не копировать")

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
    algo.add_argument("--uncertain-strategy", choices=("review", "ancestor", "root", "skip"), default=None,
                      help="что делать с сомнительными заметками")
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

    config_group = parser.add_argument_group("конфигурация")
    config_group.add_argument("--config", type=Path, default=None, help="путь к файлу конфигурации")
    config_group.add_argument("--no-config", action="store_true", help="игнорировать md_sorter.toml")

    return parser


def build_config(args: argparse.Namespace) -> Config:
    """Собирает конфигурацию: значения по умолчанию, файл, аргументы CLI."""
    root = (args.root or Path.cwd()).expanduser()
    try:
        root = root.resolve()
    except OSError as exc:  # pragma: no cover - недоступный путь
        raise ConfigError(f"Не удалось определить корневой каталог: {exc}") from exc

    config = Config(root=root)

    if not args.no_config:
        config_path = args.config or find_config_file(root)
        if config_path is not None:
            if not config_path.is_file():
                raise ConfigError(f"Файл конфигурации не найден: {config_path}")
            apply_config_data(config, load_config_file(config_path))
            config.root = root

    # Аргументы CLI имеют приоритет над файлом конфигурации; ``None`` означает
    # «параметр не задан», поэтому все флаги объявлены с ``default=None``.
    overridable = (
        "dry_run", "verbose", "debug", "quiet", "log_file", "color", "classifier",
        "threshold", "margin", "uncertain_strategy", "top_candidates", "embedding_model",
        "include_hidden", "follow_symlinks", "max_file_size", "max_analysis_chars",
        "jobs", "manifest", "copy_mode", "allow_move",
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
def _validate_root(config: Config) -> None:
    """Проверяет пригодность корневого каталога."""
    root = config.root
    if not root.exists():
        raise ConfigError(f"Каталог не существует: {root}")
    if not root.is_dir():
        raise ConfigError(f"Это не каталог: {root}")
    if SORTED_DIR_NAME in root.parts:
        raise ConfigError(
            f"Запуск внутри {SORTED_DIR_NAME} запрещён: "
            "это привело бы к повторной сортировке результата"
        )
    if not os.access(root, os.R_OK):
        raise ConfigError(f"Нет прав на чтение каталога: {root}")
    if not config.dry_run and not os.access(root, os.W_OK):
        raise ConfigError(f"Нет прав на запись в каталог: {root}")


def run(config: Config, logger: logging.Logger) -> int:
    """Выполняет полный цикл: обход, классификацию, размещение, отчёт."""
    _validate_root(config)
    reporter = Reporter(config, logger)
    reporter.announce_root(str(config.root))

    logger.info("Searching Markdown files...")
    scan = scan_tree(config, logger)

    if not config.dry_run:
        ensure_sorted_dir(config, logger)

    categories = build_categories(scan.directories)
    reporter.announce_scan(len(scan.notes), len(categories))

    # Копия структуры создаётся независимо от того, нашлись ли заметки:
    # это отдельный результат работы, а не побочный эффект классификации.
    mapping = mirror_structure(scan.directories, config, logger)

    if not scan.notes:
        logger.info("Markdown-файлы не найдены, работа завершена")
        reporter.summary(scan.errors)
        return EXIT_WITH_ERRORS if scan.errors else EXIT_OK

    parsed, failures = parse_notes(scan.notes, config, logger)
    notes_by_category = build_category_profiles(categories, parsed, config)
    categories_by_key = {category.key: category for category in categories}

    classifier = create_classifier(config)
    classifier.fit(parsed, categories, notes_by_category)

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

    total = len(parsed)
    try:
        _classify_all(
            parsed=parsed,
            classifier=classifier,
            categories_by_key=categories_by_key,
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
    elif config.manifest:
        save_manifest(config, new_manifest, logger, version=__version__)

    stats = reporter.summary(scan.errors)
    return EXIT_WITH_ERRORS if stats.errors else EXIT_OK



def _classify_all(
    *,
    parsed: Sequence[ParsedNote],
    classifier: Classifier,
    categories_by_key: dict[str, Category],
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
    """Классифицирует и размещает каждую заметку, наполняя отчёт и манифест."""
    for index, note in enumerate(parsed, start=1):
        decision = decide(note.record, classifier.rank(note, index - 1), categories_by_key, config)
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

    try:
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
