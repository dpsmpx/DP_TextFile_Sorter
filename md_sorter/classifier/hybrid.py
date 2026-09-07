"""Гибридный классификатор: взвешенная сумма оценок нескольких моделей.

Типичное применение — сочетание быстрого лексического алгоритма с локальной
embedding-моделью: первый хорошо ловит точные термины, вторая — перефразировки.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import Config
from ..models import Category, ParsedNote, TokenBag
from .lexical import LexicalClassifier


class HybridClassifier(LexicalClassifier):
    """Комбинация базового и дополнительного классификатора."""

    name = "hybrid"

    def __init__(self, config: Config, secondary: LexicalClassifier | None = None) -> None:
        super().__init__(config)
        self._secondary: LexicalClassifier | None = secondary

    def fit(
        self,
        notes: Sequence[ParsedNote],
        categories: Sequence[Category],
        notes_by_category: dict[str, list[int]] | None = None,
    ) -> None:
        """Обучает обе составляющие на одном и том же корпусе."""
        super().fit(notes, categories, notes_by_category)
        if self._secondary is None:
            from .embeddings import EmbeddingsClassifier

            self._secondary = EmbeddingsClassifier(self.config)
        if not isinstance(self._secondary, LexicalClassifier):
            raise ValueError(
                "Гибридный классификатор комбинирует оценки по категориям, поэтому "
                "второй алгоритм должен быть наследником LexicalClassifier"
            )
        self._secondary.fit(notes, categories, notes_by_category)

    def _base_scores(
        self,
        note: ParsedNote,
        note_index: int,
        note_vector: TokenBag,
    ) -> list[float]:
        """Смешивает оценки согласно ``hybrid_weights``."""
        lexical_weight, secondary_weight = self.config.hybrid_weights
        lexical = super()._base_scores(note, note_index, note_vector)
        secondary = self._secondary._base_scores(note, note_index, note_vector)
        total = lexical_weight + secondary_weight or 1.0
        return [
            (lexical_weight * left + secondary_weight * right) / total
            for left, right in zip(lexical, secondary)
        ]

    def close(self) -> None:
        """Освобождает ресурсы дополнительного классификатора."""
        if self._secondary is not None:
            self._secondary.close()
