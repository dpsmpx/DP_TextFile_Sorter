"""Структуры данных, общие для всех этапов работы сортировщика."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath

#: Имя каталога с результатом сортировки. Всегда исключается из анализа.
SORTED_DIR_NAME = "Sorted_md_files"

#: Имя каталога для заметок, которые не удалось уверенно классифицировать.
REVIEW_DIR_NAME = "_NEEDS_REVIEW"

#: Имя файла состояния (создаётся только при явном ``--manifest``).
MANIFEST_NAME = ".md_sorter_manifest.json"

#: Взвешенный мешок токенов: токен -> суммарный вес.
TokenBag = dict[str, float]


@dataclass(slots=True)
class NoteRecord:
    """Метаданные найденного Markdown-файла (без содержимого)."""

    path: Path
    relative_path: PurePosixPath
    name: str
    stem: str
    size: int
    mtime: float

    @property
    def source_dir(self) -> PurePosixPath:
        """Относительный каталог, в котором лежит заметка (``.`` для корня)."""
        return self.relative_path.parent


@dataclass(slots=True)
class ParsedNote:
    """Результат разбора одного Markdown-файла."""

    record: NoteRecord
    #: Все признаки заметки, включая основной текст.
    tokens: TokenBag = field(default_factory=dict)
    #: Только «сильные» признаки: имя файла, заголовки, теги, алиасы,
    #: языки кодовых блоков, ссылки и исходный путь — без основного текста.
    strong_tokens: TokenBag = field(default_factory=dict)
    title: str = ""
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    headings: tuple[str, ...] = ()
    wiki_links: tuple[str, ...] = ()
    code_languages: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def key_terms(self) -> frozenset[str]:
        """Множество токенов «сильных» признаков."""
        return frozenset(self.strong_tokens)


@dataclass(slots=True)
class Category:
    """Каталог назначения и его смысловой профиль."""

    rel_path: PurePosixPath
    name: str
    parents: tuple[str, ...]
    depth: int
    siblings: tuple[str, ...] = ()
    tokens: TokenBag = field(default_factory=dict)
    #: Фактический вес одной «своей» заметки в профиле (см. structure.py).
    note_factor: float = 0.0

    @property
    def key(self) -> str:
        """Строковый ключ категории в posix-виде, например ``Linux/Termux``."""
        return self.rel_path.as_posix()

    @property
    def parent_key(self) -> str:
        """Ключ родительской категории или пустая строка для верхнего уровня."""
        parent = self.rel_path.parent
        return "" if parent.as_posix() in (".", "") else parent.as_posix()


@dataclass(slots=True)
class Candidate:
    """Один вариант размещения заметки с разбивкой вклада признаков."""

    category: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


class Status(str, Enum):
    """Итог обработки одной заметки."""

    SORTED = "sorted"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"
    ERROR = "error"


class CopyOutcome(str, Enum):
    """Итог копирования одного файла."""

    COPIED = "copied"
    RENAMED = "renamed"
    IDENTICAL = "identical"
    DRY_RUN = "dry-run"
    FAILED = "failed"


@dataclass(slots=True)
class Decision:
    """Решение по одной заметке: куда, почему и с какой уверенностью."""

    record: NoteRecord
    status: Status
    category: str | None = None
    score: float = 0.0
    reason: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    destination: PurePosixPath | None = None
    outcome: CopyOutcome | None = None
    error: str = ""


@dataclass(slots=True)
class Stats:
    """Агрегированная статистика запуска."""

    processed: int = 0
    sorted_ok: int = 0
    uncertain: int = 0
    skipped: int = 0
    errors: int = 0
    identical: int = 0
    renamed: int = 0
    pruned: int = 0

    def as_line(self) -> str:
        """Однострочное представление для финального вывода."""
        return (
            f"Processed: {self.processed}   Sorted: {self.sorted_ok}   "
            f"Uncertain: {self.uncertain}   Skipped: {self.skipped}   "
            f"Errors: {self.errors}"
        )
