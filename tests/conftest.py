"""Общие фикстуры: сборка временного хранилища и запуск конвейера."""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from md_sorter.config import Config  # noqa: E402
from md_sorter.logging_setup import setup_logging  # noqa: E402


@pytest.fixture
def make_vault(tmp_path: Path) -> Callable[..., Path]:
    """Возвращает фабрику временного хранилища с заданными файлами."""

    def _factory(
        files: Mapping[str, str] | None = None,
        directories: tuple[str, ...] = (),
        name: str = "INBOX",
    ) -> Path:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (root / directory).mkdir(parents=True, exist_ok=True)
        for relative, content in (files or {}).items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return root

    return _factory


@pytest.fixture
def logger() -> logging.Logger:
    """Логгер, не засоряющий вывод тестов."""
    instance = setup_logging(quiet=True, color=False)
    instance.setLevel(logging.CRITICAL)
    return instance


@pytest.fixture
def config_factory() -> Callable[..., Config]:
    """Фабрика конфигураций с переопределением отдельных полей."""

    def _factory(root: Path, **overrides: object) -> Config:
        config = Config(root=root)
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    return _factory


def tree_digest(root: Path, *, exclude: str | None = None) -> str:
    """Хэш дерева файлов: путь + содержимое. Используется для проверок «не изменилось»."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if exclude and exclude in path.relative_to(root).parts:
            continue
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
