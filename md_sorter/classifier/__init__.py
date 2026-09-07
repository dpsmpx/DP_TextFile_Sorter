"""Реестр классификаторов.

Добавление новой модели: реализовать :class:`~md_sorter.classifier.base.Classifier`
и зарегистрировать фабрику в :data:`_FACTORIES`.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import Config
from .base import Classifier
from .lexical import LexicalClassifier


def _make_tfidf(config: Config) -> Classifier:
    from .tfidf import TfidfClassifier

    return TfidfClassifier(config)


def _make_embeddings(config: Config) -> Classifier:
    from .embeddings import EmbeddingsClassifier

    return EmbeddingsClassifier(config)


def _make_llm(config: Config) -> Classifier:
    from .llm import LLMClassifier

    return LLMClassifier(config)


def _make_hybrid(config: Config) -> Classifier:
    from .hybrid import HybridClassifier

    return HybridClassifier(config)


_FACTORIES: dict[str, Callable[[Config], Classifier]] = {
    "lexical": LexicalClassifier,
    "tfidf": _make_tfidf,
    "embeddings": _make_embeddings,
    "llm": _make_llm,
    "hybrid": _make_hybrid,
}


def available_classifiers() -> tuple[str, ...]:
    """Имена зарегистрированных классификаторов."""
    return tuple(_FACTORIES)


def create_classifier(config: Config) -> Classifier:
    """Создаёт классификатор по имени из конфигурации."""
    try:
        factory = _FACTORIES[config.classifier]
    except KeyError:
        raise ValueError(
            f"Неизвестный классификатор {config.classifier!r}; "
            f"доступны: {', '.join(available_classifiers())}"
        ) from None
    return factory(config)


__all__ = ["Classifier", "LexicalClassifier", "available_classifiers", "create_classifier"]
