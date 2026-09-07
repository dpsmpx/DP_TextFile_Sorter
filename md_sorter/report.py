"""Формирование пользовательского вывода: ход работы, объяснения, итоги."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .config import Config
from .logging_setup import log_result
from .models import CopyOutcome, Decision, Stats, Status

#: Как часто печатать прогресс при большом числе заметок.
_PROGRESS_EVERY = 200
_PROGRESS_MIN_FILES = 200


class Reporter:
    """Печатает ход классификации и итоговую статистику."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.stats = Stats()
        self.uncertain: list[Decision] = []
        self.failed: list[Decision] = []

    # ------------------------------------------------------------------
    # Этапы работы
    # ------------------------------------------------------------------
    def announce_root(self, root: str) -> None:
        """Сообщает используемые каталоги и режим запуска."""
        self.logger.info("Root: %s", root)
        if self.config.split_layout:
            self.logger.info("Хранилище (структура категорий): %s", self.config.vault_dir)
        self.logger.info("Результат: %s", self.config.sorted_dir)
        if self.config.dry_run:
            self.logger.info("Режим предварительного просмотра: файлы не копируются")

    def announce_scan(self, files: int, directories: int) -> None:
        """Сообщает результаты обхода дерева."""
        self.logger.info("Found: %d files", files)
        self.logger.info("Building directory structure...")
        self.logger.info("Found %d destination directories", directories)

    def progress(self, done: int, total: int) -> None:
        """Печатает прогресс на больших хранилищах."""
        if total < _PROGRESS_MIN_FILES or done % _PROGRESS_EVERY or done == 0:
            return
        self.logger.info("Classified %d/%d ...", done, total)

    # ------------------------------------------------------------------
    # Отдельная заметка
    # ------------------------------------------------------------------
    def report_decision(self, decision: Decision) -> None:
        """Печатает решение по одной заметке и обновляет статистику."""
        self.stats.processed += 1
        name = decision.record.relative_path.as_posix()

        if decision.status is Status.ERROR:
            self.stats.errors += 1
            self.failed.append(decision)
            self.logger.error("%s\n-> %s", name, decision.error or "неизвестная ошибка")
            return

        if decision.status is Status.SORTED:
            self.stats.sorted_ok += 1
            self.logger.info(
                "Classifying:\n%s\n-> %s\nscore: %.2f", name, decision.category, decision.score
            )
        else:
            self.stats.uncertain += 1
            self.uncertain.append(decision)
            self.logger.warning("%s\n-> no confident category (%s)", name, decision.reason)

        if self.config.verbose or self.config.debug:
            self._explain(decision)

    def report_placement(self, decision: Decision) -> None:
        """Печатает результат копирования."""
        name = decision.record.relative_path.as_posix()
        destination = decision.destination.as_posix() if decision.destination else "-"

        match decision.outcome:
            case CopyOutcome.COPIED:
                self.logger.info("Copying:\n%s -> Sorted_md_files/%s", name, destination)
            case CopyOutcome.RENAMED:
                self.stats.renamed += 1
                self.logger.info(
                    "Copying (конфликт имён):\n%s -> Sorted_md_files/%s", name, destination
                )
            case CopyOutcome.IDENTICAL:
                # Копия уже существует и совпадает побайтово — это результат
                # прошлого запуска, а не пропуск заметки: считаем отдельно.
                self.stats.identical += 1
                self.logger.debug("Уже отсортирован, пропуск: %s", destination)
            case CopyOutcome.DRY_RUN:
                self.logger.debug("Would copy: %s -> Sorted_md_files/%s", name, destination)
            case CopyOutcome.FAILED:
                self.stats.errors += 1
                self.failed.append(decision)
                self.logger.error("Копирование не удалось:\n%s\n-> %s", name, decision.error)
            case None:
                self.stats.skipped += 1
                self.logger.debug("Копирование пропущено: %s", name)

    def _explain(self, decision: Decision) -> None:
        """Подробное объяснение решения для режима ``--verbose``."""
        lines = [f"Файл: {decision.record.relative_path.as_posix()}"]
        lines.append(f"Категория: {decision.category or '— не выбрана —'}")
        lines.append(f"Почему: {decision.reason}")
        for position, candidate in enumerate(decision.candidates[: self.config.top_candidates], 1):
            lines.append(f"  {position}. {candidate.category} — score {candidate.score:.2f}")
            for item in candidate.evidence:
                lines.append(f"     - {item}")
            if self.config.debug:
                components = ", ".join(
                    f"{key}={value}" for key, value in sorted(candidate.components.items())
                )
                lines.append(f"     [{components}]")
        lines.append(f"Score: {decision.score:.2f}")
        self.logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Итоги
    # ------------------------------------------------------------------
    def preview(self, decisions: Sequence[Decision]) -> None:
        """Компактная сводка «файл -> категория» для ``--dry-run``."""
        self.logger.info("Предполагаемые назначения:")
        for decision in decisions:
            if decision.destination is None:
                self.logger.info("  %s -> — не копируется —", decision.record.name)
                continue
            directory = decision.destination.parent.as_posix()
            line = f"  {decision.record.name} -> {directory}"
            if decision.destination.name != decision.record.name:
                line = f"{line} (как {decision.destination.name})"
            self.logger.info(line)

    def summary(self, scan_errors: int = 0) -> Stats:
        """Печатает финальную статистику и возвращает её."""
        self.stats.errors += scan_errors
        if self.uncertain:
            self.logger.warning("Требуют проверки: %d", len(self.uncertain))
            for decision in self.uncertain[:20]:
                self.logger.warning("  %s — %s", decision.record.name, decision.reason)
            if len(self.uncertain) > 20:
                self.logger.warning("  ... и ещё %d", len(self.uncertain) - 20)
        log_result(self.logger, "%s", self.stats.as_line())
        if self.stats.identical:
            log_result(self.logger, "Уже были отсортированы ранее: %d", self.stats.identical)
        if self.stats.renamed:
            log_result(self.logger, "Переименовано из-за конфликта имён: %d", self.stats.renamed)
        return self.stats
