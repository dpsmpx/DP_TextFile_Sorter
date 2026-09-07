#!/usr/bin/env python3
"""Точка входа сортировщика Markdown-заметок.

Запуск из каталога хранилища::

    python sorter.py --dry-run
    python sorter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск как отдельного скрипта
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from md_sorter.cli import main

if __name__ == "__main__":
    sys.exit(main())
