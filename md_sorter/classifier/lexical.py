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
from collections.abc import Mapping, Sequence

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

#: Сколько самых весомых токенов заметки попадает в индекс соседей.
_INDEX_TERMS_PER_NOTE = 60

#: Сколько токенов запроса используется при поиске соседей.
_QUERY_TERMS = 40

#: Токены, встречающиеся чаще этой доли заметок, из индекса исключаются.
_MAX_DF_RATIO = 0.25
_MIN_POSTINGS = 20


def _link_key(name: str) -> str:
    """Нормализует имя заметки для сопоставления с целью ``[[wiki-link]]``."""
    return " ".join(tokenize_name(name))


def _combine(*signals: float) -> float:
    """Объединяет независимые свидетельства по правилу «шумного ИЛИ».

    Любого источника достаточно, чтобы поднять оценку, а согласие
    нескольких усиливает её сильнее, чем каждый по отдельности.
    Результат всегда остаётся в диапазоне 0..1.
    """
    remainder = 1.0
    for value in signals:
        remainder *= 1.0 - max(0.0, min(1.0, value))
    return 1.0 - remainder


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
        self._note_postings: dict[str, list[tuple[int, float]]] = {}
        self._note_category: dict[int, int] = {}
        self._link_targets: dict[str, int] = {}
        self._support: list[int] = []

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

        self._build_neighbour_index(notes)
        self._build_link_index(notes)

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

        neighbours = self._neighbour_scores(note_vector, note_index)
        links = self._link_scores(note)
        weights = self.config.weights
        return [
            _combine(
                max(0.0, min(1.0, profile)),
                neighbour * weights.knn_influence,
                link * weights.link_influence,
            )
            for profile, neighbour, link in zip(scores, neighbours, links)
        ]

    # ------------------------------------------------------------------
    # Хранилище как обучающий материал
    # ------------------------------------------------------------------
    def _build_neighbour_index(self, notes: Sequence[ParsedNote]) -> None:
        """Строит индекс по заметкам, которые пользователь уже разложил.

        Это главный источник знаний о личных категориях: встроенный словарь
        не знает, что такое «Хобби/Мотоциклы», а десяток лежащих там заметок
        описывают категорию точнее любого словаря.
        """
        membership = {
            note_index: category_index
            for note_index, (category_index, _) in self._member_of.items()
        }
        self._rebuild_neighbour_index(notes, membership)

    def _rebuild_neighbour_index(
        self,
        notes: Sequence[ParsedNote],
        membership: dict[int, int],
    ) -> None:
        """Перестраивает индекс соседей по заданной разметке заметок."""
        self._note_postings = {}
        self._note_category = {}
        self._support = [0] * len(self._categories)
        if not membership:
            return

        for note_index, category_index in membership.items():
            vector = to_vector(notes[note_index].tokens, self._idf, self._default_idf)
            if not vector:
                continue
            self._note_category[note_index] = category_index
            self._support[category_index] += 1
            top_terms = heapq.nlargest(
                _INDEX_TERMS_PER_NOTE, vector.items(), key=lambda item: item[1]
            )
            for token, value in top_terms:
                self._note_postings.setdefault(token, []).append((note_index, value))

        # Слишком частые токены ничего не различают, но раздувают перебор.
        limit = max(_MIN_POSTINGS, int(len(self._note_category) * _MAX_DF_RATIO))
        self._note_postings = {
            token: postings
            for token, postings in self._note_postings.items()
            if len(postings) <= limit
        }

    def learn_from(self, assignments: Mapping[int, str]) -> int:
        """Пополняет обучающий набор уверенными решениями первого прохода.

        В «холодном» хранилище, где пользователь ещё ничего не разложил, это
        единственный способ получить примеры категорий: очевидные заметки
        опознаются по названию каталога, а всё похожее на них — уже по ним.
        """
        membership = dict(self._note_category)
        added = 0
        for note_index, category_key in assignments.items():
            if note_index in membership:
                continue
            category_index = self._index_of.get(category_key)
            if category_index is None:
                continue
            membership[note_index] = category_index
            added += 1

        if added:
            self._rebuild_neighbour_index(self._notes, membership)
            self._build_link_index(self._notes)
        return added

    def _build_link_index(self, notes: Sequence[ParsedNote]) -> None:
        """Сопоставляет имена уже разложенных заметок их категориям.

        Нужен, чтобы ссылка ``[[Замена масла]]`` работала как голос за ту
        категорию, в которой лежит целевая заметка.
        """
        self._link_targets = {}
        for note_index, category_index in self._note_category.items():
            note = notes[note_index]
            for name in (note.record.stem, note.title, *note.aliases):
                key = _link_key(name)
                if key:
                    self._link_targets.setdefault(key, category_index)

    def _neighbour_scores(self, note_vector: TokenBag, note_index: int) -> list[float]:
        """Голосование ближайших уже разложенных заметок (k-NN)."""
        empty = [0.0] * len(self._categories)
        if not self._note_postings or not note_vector:
            return empty

        query = heapq.nlargest(_QUERY_TERMS, note_vector.items(), key=lambda item: item[1])
        similarity: dict[int, float] = {}
        for token, value in query:
            for other_index, other_value in self._note_postings.get(token, ()):
                if other_index == note_index:
                    continue  # заметка не может подтверждать сама себя
                similarity[other_index] = similarity.get(other_index, 0.0) + value * other_value

        if not similarity:
            return empty

        neighbours = heapq.nlargest(
            max(1, self.config.knn_neighbors), similarity.items(), key=lambda item: item[1]
        )
        total = sum(value for _, value in neighbours)
        if total <= 0.0:
            return empty
        strongest = neighbours[0][1]

        scores = empty
        for other_index, value in neighbours:
            category_index = self._note_category.get(other_index)
            if category_index is not None:
                scores[category_index] += value

        # Доля категории среди соседей, взвешенная силой самого похожего соседа
        # и тем, насколько категория вообще обжита: один сосед — слабая выборка.
        support = max(1, self.config.knn_min_support)
        return [
            min(1.0, (value / total) * strongest * min(1.0, self._support[index] / support))
            for index, value in enumerate(scores)
        ]

    def _link_scores(self, note: ParsedNote) -> list[float]:
        """Голосование категорий, в которых лежат цели ``[[wiki-links]]``."""
        scores = [0.0] * len(self._categories)
        if not self._link_targets or not note.wiki_links:
            return scores

        votes = 0
        for link in note.wiki_links:
            category_index = self._link_targets.get(_link_key(link))
            if category_index is not None:
                scores[category_index] += 1.0
                votes += 1
        if votes == 0:
            return scores

        # Одна ссылка — намёк, две и больше — уверенный сигнал.
        confidence = min(1.0, votes / 2.0)
        return [(value / votes) * confidence for value in scores]

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
