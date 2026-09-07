"""Встроенный словарь терминов, расширяющий смысл названий каталогов.

Каталог ``Linux/Arch`` сам по себе даёт только токены ``linux`` и ``arch``.
Словарь добавляет к профилю характерные термины (``pacman``, ``aur``, ...),
благодаря чему заметка находит категорию, даже если её название в тексте
не встречается ни разу.

Словарь — это подсказка, а не жёсткое правило: пользователь дополняет его
через секцию ``[aliases]`` в ``md_sorter.toml`` без правки кода.
"""

from __future__ import annotations

from collections.abc import Iterable

from .text import normalize_token, tokenize

# Ключ — нормализованный токен имени каталога, значение — родственные термины.
_RAW_LEXICON: dict[str, str] = {
    # --- Программирование -------------------------------------------------
    "programming": "код разработка программирование developer software function класс algorithm",
    "python": "pip venv django flask fastapi pandas numpy pytest conda pyproject asyncio "
              "питон requirements virtualenv poetry jupyter matplotlib",
    "cpp": "gcc clang cmake stl boost makefile pointer template header namespace qt "
           "компилятор указатель заголовочный сборка класс шаблон",
    "csharp": "dotnet nuget visual studio linq asp unity xamarin",
    "javascript": "npm node nodejs react vue angular webpack eslint typescript json "
                  "promise dom babel yarn vite промис асинхронность браузер",
    "typescript": "tsconfig npm node interface generic typing angular react",
    "java": "jvm maven gradle spring kotlin android jar class servlet",
    "rust": "cargo crate borrow lifetime tokio serde rustup clippy",
    "go": "golang goroutine module gofmt channel",
    "pascal": "delphi lazarus freepascal begin unit procedure",
    "php": "composer laravel symfony wordpress apache",
    "ruby": "gem rails bundler rspec",
    "sql": "database query select join postgres mysql sqlite index таблица запрос",
    "bash": "shell script terminal команда echo grep awk sed pipe alias скрипт оболочка конвейер",
    "web": "html css frontend backend browser http server api",
    "html": "css tag div markup browser",
    "css": "style flexbox grid selector sass scss tailwind",
    "git": "commit branch merge rebase github repository pull request clone diff "
           "коммит ветка слияние репозиторий",
    "docker": "container image compose dockerfile kubernetes volume registry "
              "контейнер образ",
    "algorithms": "sorting complexity graph tree recursion dynamic programming алгоритм сложность",
    "api": "rest endpoint request response json http token запрос ответ токен",
    "database": "sql postgres mysql sqlite mongodb index query schema база данных",
    # --- Linux и системы ---------------------------------------------------
    "linux": "kernel bash shell terminal sudo systemd distro package repository "
             "линукс ядро терминал дистрибутив пакет",
    "arch": "pacman aur yay makepkg archlinux pkgbuild rolling release manjaro "
            "арч зеркало обновление",
    "termux": "pkg android proot storage aarch64 phone smartphone "
              "термукс телефон андроид смартфон",
    "ubuntu": "apt debian deb snap ppa canonical убунту",
    "debian": "apt deb dpkg stable testing",
    "fedora": "dnf rpm redhat selinux",
    "gentoo": "portage emerge ebuild",
    "nixos": "nix flake derivation",
    "windows": "powershell registry wsl exe msi cmd реестр винда",
    "macos": "brew homebrew darwin xcode apple",
    "android": "apk adb gradle play market smartphone андроид телефон приложение",
    "network": "tcp ip dns dhcp router firewall ssh vpn wifi сеть маршрутизатор "
               "подключение интернет адрес порт",
    "security": "encryption password firewall vulnerability exploit hash certificate "
                "безопасность шифрование пароль уязвимость сертификат брандмауэр",
    "server": "nginx apache systemd deploy hosting vps ssh сервер хостинг развёртывание",
    "hardware": "cpu gpu ram ssd motherboard bios процессор видеокарта память",
    # --- Игры ---------------------------------------------------------------
    "games": "game play gameplay level boss player квест игра уровень персонаж "
             "прохождение геймплей босс оружие",
    "minecraft": "mod forge fabric redstone survival creeper nether biome craft server "
                 "майнкрафт мод редстоун крафт ферма блок выживание",
    "doom": "demon slayer wad shotgun cacodemon romero gzdoom id software "
            "дум демон дробовик ад монстр уровень секрет карта",
    "ultrakill": "hakita blood machine ranking style meter layer "
                 "ранг стиль слой кровь машина",
    "quake": "arena rocket jump id software арена ракета",
    "stalker": "zone anomaly artifact chernobyl сталкер зона аномалия",
    "witcher": "geralt ведьмак wild hunt monster геральт монстр охота",
    "roguelike": "dungeon procedural permadeath run",
    "steam": "library achievement workshop deck библиотека достижение мастерская",
    # --- Заметки, знания, быт -----------------------------------------------
    "obsidian": "vault note markdown plugin backlink graph заметка хранилище",
    "notes": "заметка note idea мысль черновик",
    "books": "книга chapter author read reading библиотека автор глава чтение",
    "music": "song album guitar chord track band музыка гитара альбом песня аккорд группа",
    "movies": "film director actor series кино фильм сериал режиссёр",
    "recipes": "рецепт ingredient cook bake кухня готовить блюдо",
    "health": "sport training workout diet здоровье тренировка питание",
    "finance": "money budget invest tax bank деньги бюджет налог инвестиции",
    "work": "task meeting project deadline работа задача проект встреча срок",
    "study": "learning course lecture exam учеба курс лекция экзамен",
    "math": "matrix integral derivative theorem математика матрица интеграл теорема",
    "physics": "quantum energy force particle физика энергия частица",
    "ai": "neural network model training llm embedding dataset нейросеть модель "
          "обучение датасет промпт",
    "design": "ui ux figma layout typography color дизайн макет шрифт",
}

# Языки кодовых блоков -> токены, которые они означают.
CODE_LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "py": ("python",),
    "python3": ("python",),
    "c++": ("cpp",),
    "cc": ("cpp",),
    "cxx": ("cpp",),
    "h": ("cpp", "c"),
    "hpp": ("cpp",),
    "cs": ("csharp",),
    "js": ("javascript",),
    "jsx": ("javascript", "react"),
    "ts": ("typescript",),
    "tsx": ("typescript", "react"),
    "sh": ("bash", "shell"),
    "zsh": ("bash", "shell"),
    "shell": ("bash", "shell"),
    "console": ("bash", "shell"),
    "ps1": ("powershell", "windows"),
    "powershell": ("powershell", "windows"),
    "yml": ("yaml",),
    "dockerfile": ("docker",),
    "psql": ("sql", "postgres"),
    "rs": ("rust",),
    "rb": ("ruby",),
    "kt": ("kotlin", "java"),
    "pas": ("pascal",),
    "delphi": ("pascal",),
}


def _build_lexicon() -> dict[str, tuple[str, ...]]:
    """Нормализует словарь один раз при импорте модуля."""
    result: dict[str, tuple[str, ...]] = {}
    for key, terms in _RAW_LEXICON.items():
        normalized_key = normalize_token(key)
        if not normalized_key:
            continue
        tokens = tuple(dict.fromkeys(tokenize(terms)))
        if tokens:
            result[normalized_key] = tokens
    return result


LEXICON: dict[str, tuple[str, ...]] = _build_lexicon()


def expand_terms(tokens: Iterable[str]) -> list[str]:
    """Возвращает родственные термины для набора токенов имени каталога."""
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(LEXICON.get(token, ()))
    return expanded


def expand_code_language(language: str) -> tuple[str, ...]:
    """Переводит идентификатор языка кодового блока в набор токенов."""
    raw = language.strip().lower()
    if not raw:
        return ()
    if raw in CODE_LANGUAGE_ALIASES:
        return CODE_LANGUAGE_ALIASES[raw]
    token = normalize_token(raw)
    return (token,) if token else ()
