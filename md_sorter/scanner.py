"""Рекурсивный обход ROOT: поиск ``.md`` и сбор структуры каталогов.

Используется :func:`os.scandir`, а не ``Path.rglob``: он быстрее и, главное,
позволяет отсечь ветку до входа в неё — это критично для хранилищ на тысячи
файлов и для гарантированного исключения ``Sorted_md_files``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .config import Config
from .models import NoteRecord

MARKDOWN_SUFFIX = ".md"


@dataclass(slots=True)
class ScanResult:
    """Результат обхода дерева каталогов."""

    notes: list[NoteRecord] = field(default_factory=list)
    directories: list[PurePosixPath] = field(default_factory=list)
    errors: int = 0
    skipped: int = 0


def _is_hidden(name: str) -> bool:
    """Скрытым считается имя, начинающееся с точки."""
    return name.startswith(".")


def _relative_posix(path: Path, root: Path) -> PurePosixPath:
    """Путь относительно ROOT в posix-виде (одинаково на всех платформах)."""
    return PurePosixPath(path.relative_to(root).as_posix())


def scan_tree(config: Config, logger: logging.Logger) -> ScanResult:
    """Обходит ROOT и возвращает заметки и структуру каталогов.

    Ошибки отдельных файлов и каталогов логируются и не прерывают обход.
    """
    root = config.root
    result = ScanResult()
    ignored = {name.lower() for name in config.ignored_directories}
    try:
        sorted_dir_real = config.sorted_dir.resolve()
    except OSError:  # pragma: no cover - недоступный путь
        sorted_dir_real = config.sorted_dir

    visited: set[tuple[int, int]] = set()
    stack: list[Path] = [root]

    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except PermissionError as exc:
            logger.error("Нет доступа к каталогу %s: %s", _safe_rel(current, root), exc)
            result.errors += 1
            continue
        except OSError as exc:
            logger.error("Не удалось прочитать каталог %s: %s", _safe_rel(current, root), exc)
            result.errors += 1
            continue

        for entry in entries:
            try:
                _handle_entry(
                    entry=entry,
                    root=root,
                    config=config,
                    logger=logger,
                    result=result,
                    stack=stack,
                    ignored=ignored,
                    sorted_dir_real=sorted_dir_real,
                    visited=visited,
                )
            except OSError as exc:
                logger.error("Ошибка обработки %s: %s", _safe_rel(Path(entry.path), root), exc)
                result.errors += 1

    result.directories.sort(key=lambda item: (len(item.parts), item.as_posix()))
    result.notes.sort(key=lambda item: item.relative_path.as_posix())
    return result


def _handle_entry(
    *,
    entry: os.DirEntry[str],
    root: Path,
    config: Config,
    logger: logging.Logger,
    result: ScanResult,
    stack: list[Path],
    ignored: set[str],
    sorted_dir_real: Path,
    visited: set[tuple[int, int]],
) -> None:
    """Обрабатывает один элемент каталога: каталог, файл или ссылку."""
    path = Path(entry.path)
    name = entry.name

    is_symlink = entry.is_symlink()
    if is_symlink and not config.follow_symlinks:
        logger.debug("Пропущена символическая ссылка: %s", _safe_rel(path, root))
        result.skipped += 1
        return

    if entry.is_dir(follow_symlinks=config.follow_symlinks):
        if name.lower() in ignored:
            logger.debug("Каталог исключён по конфигурации: %s", _safe_rel(path, root))
            return
        if not config.include_hidden and _is_hidden(name):
            logger.debug("Скрытый каталог пропущен: %s", _safe_rel(path, root))
            return
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved == sorted_dir_real or _is_within(resolved, sorted_dir_real):
            logger.debug("Каталог результата исключён: %s", _safe_rel(path, root))
            return
        if config.follow_symlinks:
            # Отмечается каждый каталог, а не только ссылка: иначе ссылка на
            # уже пройденного предка увела бы обход на лишний круг.
            try:
                stat = entry.stat(follow_symlinks=True)
            except OSError:
                stat = None
            if stat is not None:
                key = (stat.st_dev, stat.st_ino)
                if key in visited:
                    logger.warning(
                        "Каталог уже был пройден (цикл символических ссылок): %s",
                        _safe_rel(path, root),
                    )
                    result.skipped += 1
                    return
                visited.add(key)
        result.directories.append(_relative_posix(path, root))
        stack.append(path)
        return

    if not entry.is_file(follow_symlinks=config.follow_symlinks):
        if not is_symlink:
            logger.debug("Не обычный файл, пропущен: %s", _safe_rel(path, root))
        return

    if not name.lower().endswith(MARKDOWN_SUFFIX):
        return
    if not config.include_hidden and _is_hidden(name):
        logger.debug("Скрытый файл пропущен: %s", _safe_rel(path, root))
        return

    try:
        stat = entry.stat(follow_symlinks=True)
    except OSError as exc:
        logger.error("Не удалось получить сведения о файле %s: %s", _safe_rel(path, root), exc)
        result.errors += 1
        return

    relative = _relative_posix(path, root)
    result.notes.append(
        NoteRecord(
            path=path,
            relative_path=relative,
            name=name,
            stem=path.stem,
            size=stat.st_size,
            mtime=stat.st_mtime,
        )
    )


def _is_within(path: Path, parent: Path) -> bool:
    """Проверяет, находится ли ``path`` внутри ``parent``."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_rel(path: Path, root: Path) -> str:
    """Относительный путь для сообщений, с запасным вариантом."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def read_note_text(path: Path, config: Config) -> tuple[str, bool]:
    """Читает текст заметки с ограничением объёма.

    Returns:
        Пара ``(текст, был_ли_обрезан)``. Ошибки декодирования не прерывают
        работу: непригодные байты заменяются, файл всё равно классифицируется.
    """
    limit = min(config.max_file_size, config.max_analysis_chars * 4)
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    truncated = len(raw) > limit
    if truncated:
        raw = raw[:limit]
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    text = raw.decode(encoding, errors="replace")
    if len(text) > config.max_analysis_chars:
        text = text[: config.max_analysis_chars]
        truncated = True
    return text, truncated
