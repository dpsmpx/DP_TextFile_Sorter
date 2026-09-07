"""Безопасное размещение копий заметок и работа с файлом состояния.

Инвариант модуля: за пределами ``ROOT/Sorted_md_files`` не выполняется ни одной
операции записи. Единственное исключение — режим ``move``, который требует
явного флага ``--allow-move`` и по умолчанию недоступен.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import Config
from .models import CopyOutcome, MANIFEST_NAME, NoteRecord

#: Предел попыток подобрать свободное имя ``note_N.md``.
MAX_NAME_ATTEMPTS = 1000

_HASH_CHUNK = 1 << 20


@dataclass(slots=True)
class PlacementResult:
    """Итог размещения одного файла."""

    outcome: CopyOutcome
    destination: PurePosixPath | None = None
    error: str = ""


def file_sha256(path: Path) -> str:
    """Считает SHA-256 файла потоково, не загружая его целиком в память."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_names(stem: str, suffix: str) -> Iterator[str]:
    """Ленивая последовательность ``note.md``, ``note_1.md``, ``note_2.md``, ..."""
    yield f"{stem}{suffix}"
    for index in range(1, MAX_NAME_ATTEMPTS):
        yield f"{stem}_{index}{suffix}"


def place_note(
    record: NoteRecord,
    target_dir: PurePosixPath,
    config: Config,
    logger: logging.Logger,
    *,
    source_hash: str | None = None,
    reserved: set[str] | None = None,
) -> PlacementResult:
    """Копирует заметку в каталог назначения, разрешая конфликты имён.

    Совпадение SHA-256 с уже лежащим файлом означает, что это результат
    прошлого запуска: копирование пропускается. Различие означает разные
    заметки с одинаковым именем — добавляется суффикс ``_1``, ``_2``, ...

    Args:
        reserved: уже занятые в этом запуске пути назначения. Нужен в режиме
            ``--dry-run``, где файлы не создаются: без него предпросмотр
            показал бы два разных файла под одним именем и скрыл бы конфликт,
            который реальный запуск обязательно разрешит.
    """
    parts = [part for part in target_dir.parts if part not in ("", ".")]
    destination_dir = config.sorted_dir.joinpath(*parts)
    relative_dir = PurePosixPath(*parts) if parts else PurePosixPath(".")

    if not config.dry_run:
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return PlacementResult(CopyOutcome.FAILED, error=f"не удалось создать каталог: {exc}")

    stem = Path(record.name).stem
    suffix = Path(record.name).suffix

    for index, name in enumerate(_candidate_names(stem, suffix)):
        destination = destination_dir / name
        relative = relative_dir / name if parts else PurePosixPath(name)
        if reserved is not None and relative.as_posix() in reserved:
            continue

        try:
            exists = destination.exists()
        except OSError as exc:
            return PlacementResult(
                CopyOutcome.FAILED, error=f"проверка назначения не удалась: {exc}"
            )

        if exists:
            try:
                if source_hash is None:
                    source_hash = file_sha256(record.path)
                if destination.is_file() and file_sha256(destination) == source_hash:
                    return PlacementResult(CopyOutcome.IDENTICAL, relative)
            except OSError as exc:
                return PlacementResult(
                    CopyOutcome.FAILED, error=f"сравнение файлов не удалось: {exc}"
                )
            continue

        if config.dry_run:
            if reserved is not None:
                reserved.add(relative.as_posix())
            return PlacementResult(CopyOutcome.DRY_RUN, relative)

        try:
            _copy_atomic(record.path, destination)
        except FileExistsError:
            continue  # имя заняли между проверкой и созданием — берём следующее
        except OSError as exc:
            return PlacementResult(CopyOutcome.FAILED, error=f"копирование не удалось: {exc}")

        if reserved is not None:
            reserved.add(relative.as_posix())

        if config.effective_copy_mode() == "move":
            try:
                record.path.unlink()
            except OSError as exc:
                logger.error("Не удалось удалить исходный файл %s: %s", record.relative_path, exc)

        return PlacementResult(CopyOutcome.COPIED if index == 0 else CopyOutcome.RENAMED, relative)

    return PlacementResult(
        CopyOutcome.FAILED,
        error=f"не удалось подобрать свободное имя после {MAX_NAME_ATTEMPTS} попыток",
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    """Копирует файл так, чтобы в назначении не возникло полуфайла.

    Имя резервируется через ``O_EXCL`` (защита от гонки), затем данные пишутся
    во временный файл и переименовываются поверх резерва одной операцией.
    """
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)

    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=".mdsorter-", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        _copy_metadata(temporary, source)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _copy_metadata(destination: Path, source: Path) -> None:
    """Переносит время изменения и права, но не считает это обязательным.

    На Android каталог ``/sdcard`` смонтирован через FUSE: ``chmod`` и ``utime``
    там запрещены и бросают ``PermissionError``. Содержимое файла при этом
    копируется нормально, поэтому метаданные — не повод объявлять копирование
    неудачным.
    """
    try:
        shutil.copystat(source, destination)
    except OSError:
        pass


def manifest_path(config: Config) -> Path:
    """Путь к файлу состояния внутри каталога результата."""
    return config.sorted_dir / MANIFEST_NAME


def load_manifest(config: Config, logger: logging.Logger) -> dict[str, dict[str, object]]:
    """Читает файл состояния предыдущего запуска.

    Повреждённый или несовместимый манифест не является ошибкой: он просто
    игнорируется, идемпотентность в этом случае обеспечивается сверкой хэшей.
    """
    path = manifest_path(config)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Манифест %s не прочитан (%s), продолжаем без него", MANIFEST_NAME, exc)
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def save_manifest(
    config: Config,
    entries: dict[str, dict[str, object]],
    logger: logging.Logger,
    *,
    version: str,
) -> None:
    """Атомарно записывает файл состояния."""
    path = manifest_path(config)
    payload = {
        "version": version,
        "inbox": str(config.inbox),
        "vault": str(config.vault_dir),
        "entries": entries,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=".mdsorter-", suffix=".json"
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary_name, path)
    except OSError as exc:
        logger.warning("Не удалось сохранить манифест: %s", exc)
