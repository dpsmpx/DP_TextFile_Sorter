#!/usr/bin/env python3
"""Генератор демонстрационного каталога INBOX для ручной проверки сортировщика.

Использование::

    python tools/make_demo_inbox.py /tmp/INBOX
    cd /tmp/INBOX && python /путь/к/sorter.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIRECTORIES: tuple[str, ...] = (
    "Programming/Python",
    "Programming/C++",
    "Programming/JavaScript",
    "Games/Minecraft",
    "Games/Doom",
    "Games/ULTRAKILL",
    "Linux/Arch",
    "Linux/Termux",
    "Notes/Personal",
)

NOTES: dict[str, str] = {
    "Как установить пакеты в Termux.md": """---
tags:
  - termux
  - android
---

# Установка пакетов в Termux

Для установки пакетов используется `pkg install`.

```bash
pkg update && pkg upgrade
pkg install python git
```

Хранилище подключается через `termux-setup-storage`.
""",
    "network_setup.md": """# Настройка сети в Arch

Ставим пакеты через pacman, дополнительные — из AUR.

```bash
sudo pacman -S networkmanager
yay -S some-aur-package
systemctl enable NetworkManager
```

См. также [[Arch Linux установка]].
""",
    "pathlib шпаргалка.md": """---
tags: [python, programming]
aliases:
  - pathlib
---

# Работа с путями в Python

```python
import os
from pathlib import Path

for path in Path(".").rglob("*.md"):
    print(path.name)
```

Устанавливаем зависимости через pip и venv.
""",
    "redstone.md": """# Редстоун в Minecraft

Схемы автоматических ферм, поршни, наблюдатели и повторители.
Мод устанавливается через Forge. #minecraft
""",
    "ultrakill ranking.md": """# ULTRAKILL: система рангов

Style meter растёт за разнообразие убийств. Слои (layers) и боссы.
Hakita задумывал ранговую систему как способ поощрять агрессивную игру.
""",
    "cmake заметки.md": """# Сборка C++ проекта

```cpp
#include <iostream>
int main() { std::cout << "hi"; }
```

Сборка через cmake и gcc, заголовочные файлы в include/.
""",
    "промисы и async.md": """# Асинхронность в JavaScript

```javascript
const data = await fetch(url).then(r => r.json());
```

npm, node, webpack. #javascript
""",
    "случайная мысль.md": """# Мысль

Надо бы как-нибудь разобрать коробки на балконе. И позвонить в сервис.
""",
    "doom wad.md": """# Doom: свои уровни

WAD-файлы редактируются в редакторе карт. Демоны, дробовик, gzdoom.
""",
}

NESTED_NOTES: dict[str, str] = {
    "Programming/Python/декораторы.md": """# Декораторы Python

```python
from functools import wraps
```

Обёртки функций, замыкания, pip-пакеты.
""",
    "Linux/Arch/pacman.md": """# Шпаргалка pacman

`pacman -Syu`, `pacman -Ss`, чистка кэша, AUR через yay.
""",
    "Games/Doom/секреты.md": """# Секреты Doom

Скрытые комнаты на уровнях, дробовик и cacodemon.
""",
}

DUPLICATE_NOTES: dict[str, str] = {
    "A/note.md": "# Заметка A\n\nЭто про python и pip.\n",
    "B/note.md": "# Заметка B\n\nЭто тоже про python, но другое содержимое: venv.\n",
}


def build(root: Path) -> None:
    """Создаёт структуру каталогов и демонстрационные заметки."""
    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name, text in NOTES.items():
        (root / name).write_text(text, encoding="utf-8")
    for relative, text in {**NESTED_NOTES, **DUPLICATE_NOTES}.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Точка входа генератора."""
    parser = argparse.ArgumentParser(description="Создать демонстрационный INBOX")
    parser.add_argument("target", type=Path, help="каталог, который нужно создать")
    args = parser.parse_args(argv)

    target: Path = args.target.expanduser()
    target.mkdir(parents=True, exist_ok=True)
    build(target)
    print(f"Демонстрационный INBOX готов: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
