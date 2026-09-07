"""Классификатор на локальной embedding-модели (опциональная зависимость).

Смысловая близость считается между эмбеддингом заметки и эмбеддингом описания
категории. Работает офлайн после первой загрузки модели; результаты кэшируются
по SHA-256 текста, поэтому повторные запуски не пересчитывают неизменившиеся
заметки.

Установка::

    pip install sentence-transformers
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from ..config import Config
from ..models import Category, ParsedNote, TokenBag
from .lexical import LexicalClassifier

_IMPORT_HINT = (
    "Классификатор embeddings требует sentence-transformers: "
    "pip install sentence-transformers"
)

#: Сколько символов описания категории и заметки отправлять в модель.
_MAX_CHARS = 2000


class EmbeddingsClassifier(LexicalClassifier):
    """Косинусная близость эмбеддингов заметки и описания категории."""

    name = "embeddings"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._model = None
        self._category_embeddings: list[list[float]] = []
        self._note_embeddings: list[list[float]] = []
        self._cache: dict[str, list[float]] = {}
        self._cache_path: Path | None = None
        self._logger = logging.getLogger("md_sorter")

    def fit(
        self,
        notes: Sequence[ParsedNote],
        categories: Sequence[Category],
        notes_by_category: dict[str, list[int]] | None = None,
    ) -> None:
        """Загружает модель и считает эмбеддинги всех категорий пакетно."""
        super().fit(notes, categories, notes_by_category)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(_IMPORT_HINT) from exc

        self._model = SentenceTransformer(self.config.embedding_model)
        self._cache_path = self.config.sorted_dir / ".cache" / "embeddings.json"
        self._load_cache()

        # Кодирование пакетное: один вызов модели на все категории и один на
        # все заметки. Поэлементные вызовы на хранилище в тысячи файлов
        # обходились бы в тысячи прогонов модели.
        self._category_embeddings = self._encode(
            [_category_document(category) for category in categories]
        )
        self._note_embeddings = self._encode([_note_document(note) for note in notes])

    def _base_scores(
        self,
        note: ParsedNote,
        note_index: int,
        note_vector: TokenBag,
    ) -> list[float]:
        """Косинусная близость эмбеддингов (модель нормализует векторы сама)."""
        if 0 <= note_index < len(self._note_embeddings):
            embedding = self._note_embeddings[note_index]
        else:  # заметка вне корпуса, посчитанного в fit()
            embedding = self._encode([_note_document(note)])[0]
        scores = []
        for category_embedding in self._category_embeddings:
            value = sum(a * b for a, b in zip(embedding, category_embedding))
            scores.append(max(0.0, min(1.0, float(value))))
        return scores

    def close(self) -> None:
        """Сохраняет кэш эмбеддингов на диск."""
        self._save_cache()

    # ------------------------------------------------------------------
    def _encode(self, documents: Sequence[str]) -> list[list[float]]:
        """Кодирует тексты с использованием кэша по хэшу содержимого."""
        missing: list[tuple[int, str]] = []
        result: list[list[float] | None] = [None] * len(documents)
        for index, document in enumerate(documents):
            key = self._cache_key(document)
            cached = self._cache.get(key)
            if cached is not None:
                result[index] = cached
            else:
                missing.append((index, document))

        if missing:
            encoded = self._model.encode(  # type: ignore[union-attr]
                [document for _, document in missing],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for (index, document), vector in zip(missing, encoded):
                values = [float(value) for value in vector]
                self._cache[self._cache_key(document)] = values
                result[index] = values

        return [item or [] for item in result]

    def _cache_key(self, document: str) -> str:
        """Ключ кэша: модель + хэш текста."""
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        return f"{self.config.embedding_model}:{digest}"

    def _load_cache(self) -> None:
        """Читает кэш эмбеддингов, игнорируя повреждённый файл."""
        if self._cache_path is None or not self._cache_path.is_file():
            return
        try:
            with self._cache_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning("Кэш эмбеддингов не прочитан: %s", exc)
            return
        if isinstance(data, dict):
            self._cache = {str(key): list(value) for key, value in data.items()}

    def _save_cache(self) -> None:
        """Записывает кэш эмбеддингов, если это разрешено режимом запуска."""
        if self._cache_path is None or self.config.dry_run or not self._cache:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle)
        except OSError as exc:
            self._logger.warning("Кэш эмбеддингов не сохранён: %s", exc)


def _category_document(category: Category) -> str:
    """Человекочитаемое описание категории для модели."""
    path = " → ".join(category.rel_path.parts)
    terms = " ".join(sorted(category.tokens, key=lambda token: -category.tokens[token])[:40])
    return f"{path}. {terms}"[:_MAX_CHARS]


def _note_document(note: ParsedNote) -> str:
    """Сжатое представление заметки: заголовок, теги и ключевые термины."""
    parts = [note.record.stem, note.title, " ".join(note.tags), " ".join(note.headings)]
    terms = sorted(note.tokens, key=lambda token: -note.tokens[token])[:60]
    parts.append(" ".join(terms))
    return ". ".join(part for part in parts if part)[:_MAX_CHARS]
