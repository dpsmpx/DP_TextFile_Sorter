"""Классификатор на основе Claude API (опциональная зависимость).

Модель не заменяет остальную программу, а работает как арбитр: лексический
алгоритм отбирает короткий список правдоподобных категорий, а модель выбирает
из него лучшую. Такая схема ограничивает объём запроса и стоимость независимо
от размера хранилища.

Установка и ключ::

    pip install anthropic
    export ANTHROPIC_API_KEY=...

Классификатор никогда не включается по умолчанию: базовая версия программы
работает полностью локально.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from ..config import Config
from ..models import Category, ParsedNote, TokenBag
from .lexical import LexicalClassifier

_IMPORT_HINT = (
    "Классификатор llm требует пакет anthropic: pip install anthropic "
    "и переменную окружения ANTHROPIC_API_KEY"
)

#: Во сколько раз понижается оценка кандидатов, которых модель не выбрала.
_NON_SELECTED_DAMPING = 0.5

#: Ограничение объёма текста заметки, отправляемого модели.
_MAX_NOTE_CHARS = 4000

_SYSTEM_PROMPT = (
    "Ты классифицируешь заметки Obsidian по УЖЕ СУЩЕСТВУЮЩЕЙ структуре каталогов "
    "пользователя. Придумывать новые категории запрещено: выбирай строго из "
    "предложенного списка. Путь вида «Linux/Termux» означает вложенную категорию "
    "Termux внутри Linux; при прочих равных предпочитай более конкретную (вложенную) "
    "категорию. Если ни одна категория не подходит, верни category = \"NONE\". "
    "Поле confidence — твоя уверенность от 0 до 1, reason — одно короткое предложение "
    "по-русски."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["category", "confidence", "reason"],
    "additionalProperties": False,
}


class LLMClassifier(LexicalClassifier):
    """Выбор категории моделью Claude из короткого списка кандидатов."""

    name = "llm"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._client = None
        self._logger = logging.getLogger("md_sorter")
        self._failures = 0

    def fit(
        self,
        notes: Sequence[ParsedNote],
        categories: Sequence[Category],
        notes_by_category: dict[str, list[int]] | None = None,
    ) -> None:
        """Готовит лексическую основу и создаёт клиента Claude API."""
        super().fit(notes, categories, notes_by_category)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(_IMPORT_HINT) from exc

        self._anthropic = anthropic
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:  # noqa: BLE001 - нет ключа/профиля
            raise RuntimeError(f"{_IMPORT_HINT}\nПричина: {exc}") from exc

        self._logger.warning(
            "Классификатор llm выполнит до %d обращений к API (модель %s)",
            len(notes),
            self.config.llm_model,
        )

    def _base_scores(
        self,
        note: ParsedNote,
        note_index: int,
        note_vector: TokenBag,
    ) -> list[float]:
        """Уточняет лексический рейтинг вердиктом модели.

        При любой ошибке обращения к API возвращается чистый лексический
        результат: программа продолжает работу без внешнего сервиса.
        """
        lexical = super()._base_scores(note, note_index, note_vector)
        shortlist = self._shortlist(lexical)
        if not shortlist:
            return lexical

        verdict = self._ask_model(note, shortlist)
        if verdict is None:
            return lexical

        chosen_key, confidence = verdict
        chosen_index = self._index_of.get(chosen_key)
        if chosen_index is None:
            return lexical

        scores = [value * _NON_SELECTED_DAMPING for value in lexical]
        scores[chosen_index] = max(0.0, min(1.0, confidence))
        return scores

    # ------------------------------------------------------------------
    def _shortlist(self, lexical: list[float]) -> list[int]:
        """Отбирает лучших кандидатов, чтобы ограничить размер запроса."""
        limit = max(2, self.config.llm_shortlist)
        ranked = sorted(range(len(lexical)), key=lexical.__getitem__, reverse=True)
        shortlist = [index for index in ranked[:limit] if lexical[index] > 0.0]
        return shortlist or ranked[:limit]

    def _ask_model(
        self,
        note: ParsedNote,
        shortlist: Sequence[int],
    ) -> tuple[str, float] | None:
        """Отправляет один запрос к модели и разбирает структурированный ответ."""
        if self._client is None:
            return None

        options = "\n".join(f"- {self._categories[index].key}" for index in shortlist)
        prompt = (
            f"Доступные категории:\n{options}\n\n"
            f"Файл: {note.record.name}\n"
            f"Исходный каталог: {note.record.source_dir.as_posix()}\n"
            f"Заголовок: {note.title or '—'}\n"
            f"Теги: {', '.join(note.tags) or '—'}\n"
            f"Языки кода: {', '.join(note.code_languages) or '—'}\n"
            f"Ключевые термины: {self._top_terms(note)}\n\n"
            "Выбери одну категорию из списка."
        )

        try:
            response = self._client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={
                    "effort": self.config.llm_effort,
                    "format": {"type": "json_schema", "schema": _RESPONSE_SCHEMA},
                },
                messages=[{"role": "user", "content": prompt[:_MAX_NOTE_CHARS]}],
            )
        except Exception as exc:  # noqa: BLE001 - сеть, лимиты, ключ
            self._failures += 1
            if self._failures <= 3:
                self._logger.warning(
                    "Обращение к модели не удалось (%s); используется локальный алгоритм", exc
                )
            return None

        if getattr(response, "stop_reason", None) == "refusal":
            return None

        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        category = str(data.get("category", "")).strip()
        if not category or category == "NONE":
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return category, confidence

    @staticmethod
    def _top_terms(note: ParsedNote, limit: int = 30) -> str:
        """Наиболее весомые термины заметки — компактная замена полного текста."""
        terms = sorted(note.tokens, key=lambda token: -note.tokens[token])[:limit]
        return ", ".join(terms)
