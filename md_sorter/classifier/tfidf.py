"""Классификатор на TF-IDF из scikit-learn (опциональная зависимость).

Отличается от базового ``lexical`` только способом расчёта близости: словарь
и веса строит ``TfidfVectorizer``, а правила иерархии наследуются без изменений.

Установка::

    pip install scikit-learn
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import Category, ParsedNote, TokenBag
from ..text import subtract_bag
from .lexical import LexicalClassifier

_IMPORT_HINT = (
    "Классификатор tfidf требует scikit-learn: pip install scikit-learn "
    "(или используйте --classifier lexical, он работает без зависимостей)"
)


class TfidfClassifier(LexicalClassifier):
    """TF-IDF + косинусная близость поверх тех же мешков токенов."""

    name = "tfidf"

    def fit(
        self,
        notes: Sequence[ParsedNote],
        categories: Sequence[Category],
        notes_by_category: dict[str, list[int]] | None = None,
    ) -> None:
        """Обучает векторизатор на профилях категорий и текстах заметок."""
        super().fit(notes, categories, notes_by_category)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(_IMPORT_HINT) from exc

        self._cosine_similarity = cosine_similarity
        category_documents = [_bag_to_document(category.tokens) for category in categories]
        note_documents = [_bag_to_document(note.tokens) for note in notes]

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",
            sublinear_tf=True,
            min_df=1,
        )
        self._vectorizer.fit(category_documents + note_documents)
        self._category_matrix = self._vectorizer.transform(category_documents)
        # Все заметки преобразуются одной операцией, а не по одной в rank().
        self._note_matrix = self._vectorizer.transform(note_documents)

    def _base_scores(
        self,
        note: ParsedNote,
        note_index: int,
        note_vector: TokenBag,
    ) -> list[float]:
        """Косинусная близость в пространстве TF-IDF."""
        if 0 <= note_index < self._note_matrix.shape[0]:
            matrix = self._note_matrix[note_index]
        else:  # заметка вне корпуса, посчитанного в fit()
            matrix = self._vectorizer.transform([_bag_to_document(note.tokens)])
        similarities = [
            max(0.0, min(1.0, float(value)))
            for value in self._cosine_similarity(matrix, self._category_matrix)[0]
        ]
        self._exclude_self(note_index, matrix, similarities)
        return similarities

    def _exclude_self(self, note_index: int, matrix: object, similarities: list[float]) -> None:
        """Убирает вклад самой заметки из профиля каталога, в котором она лежит.

        Иначе каталог с единственной заметкой опознавал бы её саму, а не тему.
        """
        own = self._member_of.get(note_index)
        if own is None:
            return
        category_index, factor = own
        adjusted = subtract_bag(
            self._categories[category_index].tokens,
            self._notes[note_index].tokens,
            factor,
        )
        vector = self._vectorizer.transform([_bag_to_document(adjusted)])
        value = float(self._cosine_similarity(matrix, vector)[0][0])
        similarities[category_index] = max(0.0, min(1.0, value))


def _bag_to_document(bag: TokenBag) -> str:
    """Превращает взвешенный мешок в псевдотекст: вес -> кратность токена."""
    parts: list[str] = []
    for token, weight in bag.items():
        parts.extend([token] * max(1, int(round(weight))))
    return " ".join(parts)
