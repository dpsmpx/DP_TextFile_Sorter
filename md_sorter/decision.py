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


def _first_rival(ranked: Sequence[Candidate]) -> Candidate | None:
    """Лучший кандидат, не связанный с лидером отношением предок/потомок.

    Потомок лидера — не соперник, а уточнение: спор между ``Linux`` и
    ``Linux/Arch`` решается правилом конкретности, а не порогом.
    """
    if not ranked:
        return None
    leader = ranked[0].category
    for candidate in ranked[1:]:
        if not _is_related(leader, candidate.category):
            return candidate
    return None


def decide(
    record: NoteRecord,
    candidates: Sequence[Candidate],
    categories: dict[str, Category],
    config: Config,
) -> Decision:
    """Выбирает категорию для заметки или помечает решение сомнительным.

    Решение принимается по двум независимым основаниям:

    1. **абсолютное** — оценка лидера не ниже ``threshold``;
    2. **относительное** — лидер оторвался от ближайшего несвязанного
       соперника не менее чем в ``dominance`` раз и набрал хотя бы
       ``min_evidence``.

    Второе основание существует потому, что абсолютная величина косинуса
    зависит от «толщины» профиля категории. У личных категорий пользователя
    («Хобби/Мотоциклы») профиль состоит из пары слов, и даже очевидное
    совпадение даёт низкую оценку. Отрыв от соперников от этого не зависит.
    """
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
    rival = _first_rival(ranked)
    rival_score = rival.score if rival is not None else 0.0

    confident = best.score >= config.threshold
    dominant = best.score >= max(config.min_evidence, rival_score * config.dominance)

    # Спорный случай: соперник рядом, и лидер его не подавляет.
    if rival is not None and not dominant and (best.score - rival_score) < config.margin:
        fallback = _parent_fallback(record, best, rival, ranked, categories, config)
        if fallback is not None:
            return fallback
        return Decision(
            record=record,
            status=Status.UNCERTAIN,
            score=best.score,
            reason=(
                f"неоднозначность: «{best.category}» {best.score:.2f} и "
                f"«{rival.category}» {rival_score:.2f} "
                f"(разница {best.score - rival_score:.2f} < {config.margin:.2f})"
            ),
            candidates=ranked,
        )

    if not confident and not dominant:
        return Decision(
            record=record,
            status=Status.UNCERTAIN,
            score=best.score,
            reason=(
                f"слабый сигнал: «{best.category}» набрал {best.score:.2f} "
                f"при пороге {config.threshold:.2f} и не оторвался от соперников"
            ),
            candidates=ranked,
        )

    reason = best.evidence[0] if best.evidence else "наибольшая смысловая близость"
    if not confident:
        reason = f"{reason}; лидер вне конкуренции (соперник {rival_score:.2f})"

    return Decision(
        record=record,
        status=Status.SORTED,
        category=best.category,
        score=best.score,
        reason=reason,
        candidates=ranked,
    )


def _parent_fallback(
    record: NoteRecord,
    best: Candidate,
    rival: Candidate,
    ranked: Sequence[Candidate],
    categories: dict[str, Category],
    config: Config,
) -> Decision | None:
    """Спор двух соседей решается в пользу их общего родителя, если он подходит."""
    parent_key = _common_parent(best.category, rival.category)
    if not parent_key or parent_key not in categories:
        return None
    parent_candidate = next((item for item in ranked if item.category == parent_key), None)
    if parent_candidate is None or parent_candidate.score < config.threshold:
        return None
    return Decision(
        record=record,
        status=Status.SORTED,
        category=parent_key,
        score=parent_candidate.score,
        reason=(
            f"«{best.category}» и «{rival.category}» почти равны "
            f"(разница {best.score - rival.score:.2f}); выбран общий родитель"
        ),
        candidates=list(ranked),
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
    if strategy == "best" and decision.candidates:
        best = decision.candidates[0].category
        return mapping.get(best, PurePosixPath(best))
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
