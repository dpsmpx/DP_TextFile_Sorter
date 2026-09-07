"""Принятие решения по заметке: порог, неоднозначность, стратегии запаса.

Слой намеренно отделён от классификатора: правила «когда мы не уверены»
одинаковы для любой модели — лексической, TF-IDF, embeddings или LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath

from .config import Config
from .models import Candidate, Category, Decision, NoteRecord, REVIEW_DIR_NAME, Status


def _common_parent(first: str, second: str) -> str:
    """Общий предок двух категорий (пустая строка, если его нет)."""
    first_parts = first.split("/")
    second_parts = second.split("/")
    shared: list[str] = []
    for left, right in zip(first_parts, second_parts):
        if left != right:
            break
        shared.append(left)
    return "/".join(shared)


def _is_related(first: str, second: str) -> bool:
    """True, если категории связаны отношением предок/потомок."""
    return first.startswith(f"{second}/") or second.startswith(f"{first}/")


def decide(
    record: NoteRecord,
    candidates: Sequence[Candidate],
    categories: dict[str, Category],
    config: Config,
) -> Decision:
    """Выбирает категорию для заметки или помечает решение сомнительным."""
    ranked = list(candidates)
    if not categories:
        return Decision(
            record=record,
            status=Status.UNCERTAIN,
            reason="в структуре нет каталогов назначения",
            candidates=ranked,
        )
    if not ranked:
        return Decision(
            record=record,
            status=Status.UNCERTAIN,
            reason="ни одна категория не набрала ненулевой оценки",
            candidates=ranked,
        )

    best = ranked[0]
    if best.score < config.threshold:
        return Decision(
            record=record,
            status=Status.UNCERTAIN,
            score=best.score,
            reason=(
                f"лучшая оценка {best.score:.2f} ниже порога {config.threshold:.2f} "
                f"(ближайший кандидат: {best.category})"
            ),
            candidates=ranked,
        )

    if len(ranked) > 1:
        second = ranked[1]
        gap = best.score - second.score
        if gap < config.margin and not _is_related(best.category, second.category):
            parent_key = _common_parent(best.category, second.category)
            parent = categories.get(parent_key) if parent_key else None
            parent_candidate = next(
                (item for item in ranked if item.category == parent_key), None
            )
            resolved_by_parent = (
                parent is not None
                and parent_candidate is not None
                and parent_candidate.score >= config.threshold
            )
            if resolved_by_parent:
                return Decision(
                    record=record,
                    status=Status.SORTED,
                    category=parent_key,
                    score=parent_candidate.score,
                    reason=(
                        f"«{best.category}» и «{second.category}» почти равны "
                        f"(разница {gap:.2f}); выбран общий родитель"
                    ),
                    candidates=ranked,
                )
            return Decision(
                record=record,
                status=Status.UNCERTAIN,
                score=best.score,
                reason=(
                    f"неоднозначность: «{best.category}» {best.score:.2f} и "
                    f"«{second.category}» {second.score:.2f} (разница {gap:.2f} < {config.margin:.2f})"
                ),
                candidates=ranked,
            )

    return Decision(
        record=record,
        status=Status.SORTED,
        category=best.category,
        score=best.score,
        reason=best.evidence[0] if best.evidence else "наибольшая смысловая близость",
        candidates=ranked,
    )


def resolve_destination(
    decision: Decision,
    mapping: dict[str, PurePosixPath],
    config: Config,
) -> PurePosixPath | None:
    """Определяет каталог внутри ``Sorted_md_files`` для решения.

    Returns:
        Относительный путь каталога назначения либо ``None``, если файл
        копировать не нужно (стратегия ``skip``).
    """
    if decision.status is Status.SORTED and decision.category is not None:
        return mapping.get(decision.category, PurePosixPath(decision.category))

    strategy = config.uncertain_strategy
    if strategy == "skip":
        return None
    if strategy == "root":
        return PurePosixPath(".")
    if strategy == "ancestor":
        ancestor = _ancestor_for(decision)
        if ancestor:
            return mapping.get(ancestor, PurePosixPath(ancestor))
    return PurePosixPath(REVIEW_DIR_NAME)


def _ancestor_for(decision: Decision) -> str:
    """Общий предок двух лучших кандидатов, если он существует."""
    if len(decision.candidates) < 2:
        return ""
    return _common_parent(decision.candidates[0].category, decision.candidates[1].category)
