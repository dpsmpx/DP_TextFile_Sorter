"""Базовый интерфейс классификаторов.

Любой алгоритм — лексический, TF-IDF, embeddings, LLM — реализует один и тот
же контракт: ``fit`` по корпусу и ``rank`` для отдельной заметки. Это позволяет
заменить модель, не трогая остальную программу.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from ..config import Config
from ..models import Candidate, Category, ParsedNote


class Classifier(ABC):
    """Абстрактный классификатор заметок по категориям."""

    #: Короткое имя, под которым классификатор доступен в ``--classifier``.
    name: str = "base"

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def fit(
        self,
        notes: Sequence[ParsedNote],
        categories: Sequence[Category],
        notes_by_category: dict[str, list[int]] | None = None,
    ) -> None:
        """Подготавливает модель по корпусу заметок и профилям категорий.

        Args:
            notes: разобранные заметки.
            categories: доступные категории назначения.
            notes_by_category: индексы заметок, физически лежащих в категории;
                используются, чтобы исключить самоподтверждение при оценке.
        """

    @abstractmethod
    def rank(self, note: ParsedNote, note_index: int = -1) -> list[Candidate]:
        """Возвращает кандидатов, отсортированных по убыванию score."""

    def learn_from(self, assignments: Mapping[int, str]) -> int:
        """Добавляет уверенно классифицированные заметки в обучающий набор.

        Позволяет второму проходу опереться на результаты первого. Реализация
        по умолчанию ничего не делает и сообщает, что новых знаний нет.

        Args:
            assignments: отображение «индекс заметки -> ключ категории».

        Returns:
            Сколько заметок реально добавлено в обучающий набор.
        """
        return 0

    def close(self) -> None:
        """Освобождает ресурсы (кэши, соединения). По умолчанию — ничего."""
