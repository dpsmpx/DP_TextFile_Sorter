"""Настройка вывода: формат ``[INFO] ...`` и опциональный файл лога."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOGGER_NAME = "md_sorter"

#: Уровень итоговых сообщений: выводится даже в режиме ``--quiet``,
#: но выглядит как обычный ``[INFO]``.
RESULT_LEVEL = 25
logging.addLevelName(RESULT_LEVEL, "INFO")

_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[2;37m",
    logging.INFO: "\033[0;36m",
    logging.WARNING: "\033[0;33m",
    logging.ERROR: "\033[0;31m",
    logging.CRITICAL: "\033[1;31m",
}
_RESET = "\033[0m"


class BracketFormatter(logging.Formatter):
    """Форматтер вида ``[LEVEL] сообщение`` с опциональным цветом.

    Многострочные сообщения выравниваются отступом, чтобы блок объяснения
    в режиме ``--verbose`` читался как единое целое.
    """

    def __init__(self, *, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        head, _, tail = message.partition("\n")
        prefix = f"[{record.levelname}]"
        if self.use_color:
            color = _COLORS.get(record.levelno, "")
            prefix = f"{color}{prefix}{_RESET}"
        lines = [f"{prefix} {head}"]
        if tail:
            lines.extend(f"      {line}" for line in tail.split("\n"))
        return "\n".join(lines)


def _color_supported(stream: object, override: bool | None) -> bool:
    """Определяет, уместны ли ANSI-коды в данном потоке вывода."""
    if override is not None:
        return override
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def setup_logging(
    *,
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    log_file: Path | None = None,
    color: bool | None = None,
    stream: object | None = None,
) -> logging.Logger:
    """Создаёт и настраивает логгер программы.

    Повторный вызов безопасен: старые обработчики закрываются и заменяются.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if debug:
        level = logging.DEBUG
    elif quiet:
        level = RESULT_LEVEL
    else:
        level = logging.INFO

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    target = stream if stream is not None else sys.stdout
    console = logging.StreamHandler(target)  # type: ignore[arg-type]
    console.setLevel(level)
    console.setFormatter(BracketFormatter(use_color=_color_supported(target, color)))
    logger.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError as exc:
            logger.warning("Не удалось открыть файл лога %s: %s", log_file, exc)
        else:
            file_handler.setLevel(logging.DEBUG if debug or verbose else logging.INFO)
            file_handler.setFormatter(BracketFormatter(use_color=False))
            logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Возвращает логгер программы (уже настроенный или логгер по умолчанию)."""
    return logging.getLogger(LOGGER_NAME)


def log_result(logger: logging.Logger, message: str, *args: object) -> None:
    """Печатает итоговое сообщение, которое не скрывается в ``--quiet``."""
    logger.log(RESULT_LEVEL, message, *args)
