"""Лексико-семантический классификатор (алгоритм по умолчанию).

Идея: и заметка, и каталог представляются взвешенными мешками токенов; близость
считается косинусом IDF-взвешенных векторов. Поверх этого работают правила,
которые кодируют смысл иерархии: поддержка родителя, бонус за конкретность,
приоритет потомка и приор исходного пути.

Классификатор не требует внешних зависимостей, детерминирован и объясним —
поэтому он выбран базовым.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

from ..config import Config
from ..models import Candidate, Category, ParsedNote, TokenBag
from ..text import (
    compute_idf,
    cosine,
    subtract_bag,
    to_vector,
    tokenize_name,
    top_shared_terms,
)
from .base import Classifier


class LexicalClassifier(Classifier):
    """Косинусная близость мешков токенов плюс правила иерархии."""

    name = "lexical"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._categories: list[Category] = []
        self._index_of: dict[str, int] = {}
        self._vectors: list[TokenBag] = []
        self._inverted: dict[str, list[tuple[int, float]]] = {}
        self._name_tokens: list[frozenset[str]] = []
        self._member_of: dict[int, tuple[int, float]] = {}
        self._notes: Sequence[ParsedNote] = ()
        self._idf: dict[str, float] = {}
        self._default_idf: float = 1.0

    def fit(
        self,
        notes: Sequence[ParsedNote],
        categories: Sequence[Category],
        notes_by_category: dict[str, list[int]] | None = None,
    ) -> None:
        """Считает IDF по корпусу и строит инвертированный индекс категорий."""
        self._notes = notes
        self._categories = list(categories)
        self._index_of = {category.key: index for index, category in enumerate(self._categories)}

        documents: list[Sequence[str]] = [tuple(note.tokens) for note in notes]
        documents.extend(tuple(category.tokens) for category in self._categories)
        self._idf, self._default_idf = compute_idf(documents)

        self._vectors = [
            to_vector(category.tokens, self._idf, self._default_idf)
            for category in self._categories
        ]
        self._name_tokens = [
            frozenset(tokenize_name(category.name)) for category in self._categories
        ]

        inverted: dict[str, list[tuple[int, float]]] = {}
        for index, vector in enumerate(self._vectors):
            for token, value in vector.items():
                inverted.setdefault(token, []).append((index, value))
        self._inverted = inverted

        self._member_of = {}
        for key, indices in (notes_by_category or {}).items():
            category_index = self._index_of.get(key)
            if category_index is None or not indices:
                continue
            factor = self._categories[category_index].note_factor
            if factor <= 0.0:
                continue
            for note_index in indices:
                self._member_of[note_index] = (category_index, factor)

    def rank(self, note: ParsedNote, note_index: int = -1) -> list[Candidate]:
        """Оценивает все категории и возвращает лучшие варианты."""
        if not self._categories:
            return []

        note_vector = self.note_vector(note)
        base = self._base_scores(note, note_index, note_vector)
        final = self._apply_rules(note, base)

        # Не меньше трёх кандидатов: слою принятия решения нужен запас,
        # чтобы проверить неоднозначность и найти общего родителя.
        limit = max(3, self.config.top_candidates)
        best = heapq.nlargest(limit, range(len(self._categories)), key=final.__getitem__)
        best = [index for index in best if final[index] > 0.0]
        if not best:
            return []

        best = self._promote_descendant(best, final)
        candidates = [
            self._build_candidate(index, note, note_vector, base, final)
            for index in best[:limit]
        ]
        return candidates

    # ------------------------------------------------------------------
    # Внутренняя кухня
    # ------------------------------------------------------------------
    def note_vector(self, note: ParsedNote) -> TokenBag:
        """IDF-взвешенный нормализованный вектор заметки (полный текст)."""
        return to_vector(note.tokens, self._idf, self._default_idf)

    def _base_scores(
        self,
        note: ParsedNote,
        note_index: int,
        note_vector: TokenBag,
    ) -> list[float]:
        """Косинусная близость заметки ко всем категориям.

        Делается два прохода: по полному тексту и отдельно по «сильным»
        признакам (имя файла, заголовки, теги, языки кода, ссылки). Берётся
        лучший результат — иначе длинная проза размывает короткий, но
        однозначный сигнал вроде ``tags: [minecraft]``.

        Подклассы (TF-IDF, embeddings, LLM) переопределяют только этот метод:
        правила иерархии и формирование объяснений остаются общими.
        """
        strong_vector = to_vector(note.strong_tokens, self._idf, self._default_idf)
        scores = self._similarities(note_vector)
        for index, value in enumerate(self._similarities(strong_vector)):
            if value > scores[index]:
                scores[index] = value

        own = self._member_of.get(note_index)
        if own is not None:
            category_index, factor = own
            category = self._categories[category_index]
            adjusted = subtract_bag(category.tokens, self._notes[note_index].tokens, factor)
            vector = to_vector(adjusted, self._idf, self._default_idf)
            scores[category_index] = max(
                cosine(note_vector, vector), cosine(strong_vector, vector)
            )

        return [max(0.0, min(1.0, value)) for value in scores]

    def _similarities(self, vector: TokenBag) -> list[float]:
        """Близость одного вектора ко всем категориям через инвертированный индекс."""
        scores = [0.0] * len(self._categories)
        if not vector:
            return scores
        for token, value in vector.items():
            for category_index, category_value in self._inverted.get(token, ()):
                scores[category_index] += value * category_value
        return scores

    @staticmethod
    def _strong_tokens(note: ParsedNote) -> frozenset[str]:
        """Токены «сильных» признаков заметки (для проверки точного совпадения)."""
        return note.key_terms

    def _apply_rules(self, note: ParsedNote, base: list[float]) -> list[float]:
        """Накладывает правила иерархии на косинусные оценки."""
        weights = self.config.weights
        strong = self._strong_tokens(note)
        source_dir = note.record.source_dir.as_posix()
        final: list[float] = []

        for index, category in enumerate(self._categories):
            value = base[index]
            parent_index = self._index_of.get(category.parent_key) if category.parent_key else None
            if parent_index is not None:
                value += weights.parent_support * base[parent_index]
            if value > 0.0 and self._name_tokens[index] and self._name_tokens[index] <= strong:
                value *= weights.name_match_boost
            if source_dir == category.key:
                value += weights.source_path_prior
            if value > 0.0 and category.depth > 1:
                value *= 1.0 + weights.depth_bonus * (category.depth - 1)
            final.append(min(1.0, value))

        return final

    def _promote_descendant(self, order: list[int], final: list[float]) -> list[int]:
        """Отдаёт предпочтение более конкретной категории при близких оценках.

        Если лидер является предком другого сильного кандидата, а разрыв между
        ними в пределах ``descendant_epsilon``, вперёд выходит потомок.
        """
        if len(order) < 2:
            return order
        epsilon = self.config.weights.descendant_epsilon
        leader = order[0]
        leader_key = self._categories[leader].key
        threshold = final[leader] * (1.0 - epsilon)

        best_index = leader
        best_depth = self._categories[leader].depth
        for index in order[1:]:
            category = self._categories[index]
            if not category.key.startswith(f"{leader_key}/"):
                continue
            if final[index] >= threshold and category.depth > best_depth:
                best_index = index
                best_depth = category.depth

        if best_index == leader:
            return order
        reordered = [best_index]
        reordered.extend(index for index in order if index != best_index)
        return reordered

    def _build_candidate(
        self,
        index: int,
        note: ParsedNote,
        note_vector: TokenBag,
        base: list[float],
        final: list[float],
    ) -> Candidate:
        """Собирает кандидата вместе с объяснением решения."""
        category = self._categories[index]
        weights = self.config.weights
        strong = self._strong_tokens(note)
        parent_index = self._index_of.get(category.parent_key) if category.parent_key else None

        components = {
            "similarity": round(base[index], 4),
            "parent_support": round(
                weights.parent_support * base[parent_index] if parent_index is not None else 0.0, 4
            ),
            "depth_bonus": round(weights.depth_bonus * (category.depth - 1), 4),
        }

        evidence: list[str] = []
        shared = top_shared_terms(note_vector, self._vectors[index], limit=5)
        if shared:
            terms = ", ".join(token for token, _ in shared)
            evidence.append(f"совпадение терминов: {terms}")
        if self._name_tokens[index] and self._name_tokens[index] <= strong:
            components["name_match"] = weights.name_match_boost
            evidence.append(
                f"имя категории «{category.name}» встречается в имени файла, "
                "заголовке, тегах или ссылках"
            )
        if note.record.source_dir.as_posix() == category.key:
            components["source_path_prior"] = weights.source_path_prior
            evidence.append("заметка уже лежала в этом каталоге")
        if parent_index is not None and base[parent_index] > 0.0:
            parent_key = self._categories[parent_index].key
            evidence.append(f"содержание связано с родительской категорией «{parent_key}»")
        if category.depth > 1:
            evidence.append(
                f"«{category.key}» — более конкретная категория, "
                f"чем «{category.parent_key}»"
            )
        if note.code_languages:
            languages = ", ".join(note.code_languages)
            if any(language in self._name_tokens[index] for language in note.code_languages):
                evidence.append(f"язык кодовых блоков: {languages}")

        return Candidate(
            category=category.key,
            score=round(final[index], 4),
            components=components,
            evidence=evidence,
        )
