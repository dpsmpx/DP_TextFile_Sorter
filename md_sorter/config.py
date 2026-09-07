"""Конфигурация сортировщика: значения по умолчанию и чтение TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .models import SORTED_DIR_NAME

#: Имена файлов конфигурации, которые ищутся в ROOT (в порядке приоритета).
CONFIG_FILENAMES: tuple[str, ...] = ("md_sorter.toml", ".md_sorter.toml")

#: Пользовательская конфигурация, общая для всех хранилищ.
USER_CONFIG_PATH = Path.home() / ".config" / "md_sorter" / "config.toml"

#: Ключи, значение которых всегда трактуется как путь.
_PATH_KEYS = frozenset({"inbox", "vault", "output", "log_file"})

#: Каталоги, которые никогда не сканируются.
DEFAULT_IGNORED_DIRECTORIES: tuple[str, ...] = (
    SORTED_DIR_NAME,
    ".git",
    ".obsidian",
    ".trash",
    ".stfolder",
    ".stversions",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
)


@dataclass(slots=True)
class ScoringWeights:
    """Веса признаков заметки и правил ранжирования категорий.

    Значения подобраны эмпирически; менять их можно через секцию
    ``[weights]`` конфигурационного файла без правки кода.
    """

    # --- вес признаков заметки ---
    filename: float = 3.0
    frontmatter_tags: float = 3.0
    inline_tags: float = 3.0
    aliases: float = 3.0
    title: float = 2.5
    heading: float = 2.0
    code_language: float = 2.5
    code_body: float = 1.5
    wiki_link: float = 1.5
    # Исходный каталог учитывается как контекст, а не как имя заметки:
    # он дополнительно даёт отдельный бонус source_path_prior.
    source_path: float = 1.0
    body: float = 1.0

    # --- вес признаков категории ---
    category_name: float = 3.0
    category_path: float = 2.0
    category_lexicon: float = 2.0
    category_alias: float = 3.0
    category_notes: float = 1.0

    # --- дополнительные источники сигнала ---
    # Голос ближайших уже разложенных заметок и ссылок [[wiki-link]] на них.
    knn_influence: float = 1.0
    link_influence: float = 0.6

    # --- правила ранжирования ---
    parent_support: float = 0.35
    depth_bonus: float = 0.06
    name_match_boost: float = 1.15
    source_path_prior: float = 0.10
    descendant_epsilon: float = 0.10


@dataclass(slots=True)
class Config:
    """Полный набор параметров запуска.

    Три пути разделены намеренно. В типичном хранилище Obsidian структура
    категорий лежит в корне, а несортированные заметки — в отдельной папке
    внутри него::

        MAIN_OBSIDIAN_PC/     <- vault: отсюда берётся система категорий
        ├── INBOX/            <- inbox: что нужно разложить
        ├── Programming/
        ├── Games/
        └── Linux/

    Заметки, уже разложенные по категориям хранилища, служат обучающим
    материалом, но сами не сортируются и не копируются.
    """

    #: Каталог с несортированными заметками (что раскладываем).
    inbox: Path = field(default_factory=Path.cwd)

    #: Корень хранилища, откуда берётся структура категорий.
    #: ``None`` означает «тот же каталог, что и inbox».
    vault: Path | None = None

    #: Куда складывать результат. ``None`` — ``<inbox>/Sorted_md_files``.
    output: Path | None = None
    dry_run: bool = False
    verbose: bool = False
    debug: bool = False
    quiet: bool = False
    log_file: Path | None = None
    color: bool | None = None

    # Абсолютный порог уверенности. Работает только для категорий с «толстым»
    # профилем: много своих заметок или попадание во встроенный словарь.
    threshold: float = 0.22

    # Относительное правило приёма. Величина косинуса зависит от размера профиля
    # категории, поэтому у личных категорий пользователя абсолютные оценки
    # всегда низкие. Если лидер оторвался от ближайшего несвязанного соперника
    # хотя бы в `dominance` раз и набрал не меньше `min_evidence`, решение
    # принимается независимо от абсолютной величины.
    dominance: float = 1.5
    min_evidence: float = 0.05

    # Минимальный отрыв лидера от соперника; иначе решение считается спорным.
    margin: float = 0.06
    uncertain_strategy: str = "review"
    classifier: str = "lexical"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "claude-opus-5"
    llm_shortlist: int = 10
    llm_max_tokens: int = 2048
    llm_effort: str = "low"
    hybrid_weights: tuple[float, float] = (0.4, 0.6)

    copy_mode: str = "copy"
    allow_move: bool = False
    manifest: bool = False

    ignored_directories: tuple[str, ...] = DEFAULT_IGNORED_DIRECTORIES
    include_hidden: bool = False
    follow_symlinks: bool = False
    max_file_size: int = 5 * 1024 * 1024
    max_analysis_chars: int = 40_000
    jobs: int = 1
    parallel_threshold: int = 500
    top_candidates: int = 3

    # Сколько раз уверенные решения возвращаются в обучающий набор.
    # Нужно для «холодного» хранилища, где ещё ничего не разложено:
    # первый проход опознаёт очевидное, второй опирается на него.
    self_training_rounds: int = 1

    # Обучаться можно только на решениях не слабее этой оценки. Иначе
    # заметка, принятая относительным правилом «за неимением лучшего»,
    # станет примером категории и потянет за собой похожие ошибки.
    self_training_min_score: float = 0.10

    # Сколько ближайших уже разложенных заметок голосует за категорию.
    knn_neighbors: int = 5

    # Сколько заметок должно лежать в категории, чтобы её голос учитывался
    # полностью. Категория с единственной заметкой иначе притягивала бы
    # всё похожее на эту одну заметку.
    knn_min_support: int = 3

    # Сколько общих терминов должно быть у заметки с соседом, чтобы его
    # голос учитывался полностью. Одно случайно совпавшее слово вроде
    # «проверил» не должно перевешивать три точных термина по теме.
    knn_min_terms: int = 2

    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    @property
    def vault_dir(self) -> Path:
        """Корень хранилища: явно заданный или совпадающий с inbox."""
        return self.vault if self.vault is not None else self.inbox

    @property
    def sorted_dir(self) -> Path:
        """Абсолютный путь к каталогу с результатом сортировки."""
        return self.output if self.output is not None else self.inbox / SORTED_DIR_NAME

    @property
    def split_layout(self) -> bool:
        """True, если структура категорий берётся не из самого inbox."""
        return self.vault_dir != self.inbox

    def effective_copy_mode(self) -> str:
        """Фактический режим переноса с учётом защиты исходных файлов."""
        if self.copy_mode == "move" and self.allow_move:
            return "move"
        return "copy"


class ConfigError(RuntimeError):
    """Ошибка чтения или валидации конфигурации."""


def find_config_file(*directories: Path) -> Path | None:
    """Ищет файл конфигурации в указанных каталогах, затем у пользователя.

    Порядок важен: настройки конкретного хранилища должны перекрывать общие,
    поэтому сначала просматривается inbox, затем корень хранилища и только
    потом ``~/.config/md_sorter/config.toml``.
    """
    seen: set[Path] = set()
    for directory in directories:
        if directory in seen:
            continue
        seen.add(directory)
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    if USER_CONFIG_PATH.is_file():
        return USER_CONFIG_PATH
    return None


def _coerce_aliases(raw: object) -> dict[str, tuple[str, ...]]:
    """Приводит секцию ``[aliases]`` к виду ``категория -> термины``."""
    if not isinstance(raw, dict):
        raise ConfigError("Секция [aliases] должна быть таблицей")
    aliases: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            terms: tuple[str, ...] = (value,)
        elif isinstance(value, (list, tuple)):
            terms = tuple(str(item) for item in value)
        else:
            raise ConfigError(f"Алиасы категории {key!r} должны быть строкой или списком")
        aliases[str(key).replace("\\", "/").strip("/")] = terms
    return aliases


def load_config_file(path: Path) -> dict[str, object]:
    """Читает TOML-конфигурацию и возвращает плоский словарь параметров."""
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"Не удалось прочитать {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Некорректный TOML в {path}: {exc}") from exc

    section = data.get("md_sorter", data)
    if not isinstance(section, dict):
        raise ConfigError(f"Ожидалась таблица параметров в {path}")
    return dict(section)


def apply_config_data(config: Config, data: dict[str, object]) -> Config:
    """Накладывает значения из файла на конфигурацию.

    Неизвестные ключи игнорируются молча — это позволяет держать в одном файле
    настройки нескольких версий программы.
    """
    known = {field_info.name for field_info in fields(Config)}
    for key, value in data.items():
        if key == "aliases":
            config.aliases = _coerce_aliases(value)
            continue
        if key == "weights":
            if not isinstance(value, dict):
                raise ConfigError("Секция [weights] должна быть таблицей")
            weight_names = {field_info.name for field_info in fields(ScoringWeights)}
            for weight_key, weight_value in value.items():
                if weight_key in weight_names:
                    setattr(config.weights, weight_key, float(weight_value))
            continue
        if key not in known:
            continue
        current = getattr(config, key)
        if isinstance(current, tuple) and isinstance(value, list):
            setattr(config, key, tuple(value))
        elif isinstance(current, Path) or key in _PATH_KEYS:
            setattr(config, key, Path(str(value)).expanduser())
        elif isinstance(current, bool):
            setattr(config, key, bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(config, key, int(value))
        elif isinstance(current, float):
            setattr(config, key, float(value))
        else:
            setattr(config, key, value)
    return config


def validate(config: Config) -> None:
    """Проверяет непротиворечивость параметров, бросает :class:`ConfigError`."""
    if not 0.0 <= config.threshold <= 1.0:
        raise ConfigError("threshold должен быть в диапазоне 0..1")
    if not 0.0 <= config.margin <= 1.0:
        raise ConfigError("margin должен быть в диапазоне 0..1")
    if config.uncertain_strategy not in {"review", "ancestor", "root", "skip", "best"}:
        raise ConfigError(
            "uncertain_strategy: допустимы review, ancestor, root, skip, best"
        )
    if config.dominance < 1.0:
        raise ConfigError("dominance не может быть меньше 1.0")
    if not 0.0 <= config.min_evidence <= 1.0:
        raise ConfigError("min_evidence должен быть в диапазоне 0..1")
    if not 0.0 <= config.self_training_min_score <= 1.0:
        raise ConfigError("self_training_min_score должен быть в диапазоне 0..1")
    if config.self_training_rounds < 0:
        raise ConfigError("self_training_rounds не может быть отрицательным")
    if config.copy_mode not in {"copy", "move"}:
        raise ConfigError("copy_mode: допустимы copy или move")
    if config.copy_mode == "move" and not config.allow_move:
        raise ConfigError(
            "Режим move требует явного флага --allow-move "
            "(по умолчанию исходные файлы неприкосновенны)"
        )
    if config.output is not None and config.output == config.inbox:
        raise ConfigError("Каталог результата не может совпадать с inbox")
    if config.max_file_size <= 0:
        raise ConfigError("max_file_size должен быть положительным")
    if config.max_analysis_chars <= 0:
        raise ConfigError("max_analysis_chars должен быть положительным")
    if config.jobs < 0:
        raise ConfigError("jobs не может быть отрицательным")


def _toml_value(value: object) -> str:
    """Сериализует значение в TOML (нужное подмножество типов)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


#: Параметры, которые не имеет смысла сохранять: это режимы одного запуска.
#: Цвет к тому же определяется автоматически по типу вывода.
_TRANSIENT_KEYS = frozenset({"dry_run", "color"})


def save_config(config: Config, path: Path) -> Path:
    """Сохраняет текущие настройки в TOML-файл.

    Записываются пути и всё, что отличается от значений по умолчанию, — чтобы
    файл оставался коротким и читаемым, а не копией всей структуры.

    Returns:
        Путь к записанному файлу.
    """
    defaults = Config()
    lines = [
        "# Создано автоматически: python sorter.py --save-config",
        "# Файл читается при запуске, поэтому пути больше вводить не нужно.",
        "",
        "[md_sorter]",
        f"inbox = {_toml_value(str(config.inbox))}",
    ]
    if config.vault is not None:
        lines.append(f"vault = {_toml_value(str(config.vault))}")
    if config.output is not None:
        lines.append(f"output = {_toml_value(str(config.output))}")

    for field_info in fields(Config):
        name = field_info.name
        if name in {"inbox", "vault", "output", "aliases", "weights"} or name in _TRANSIENT_KEYS:
            continue
        value = getattr(config, name)
        if value == getattr(defaults, name):
            continue
        if isinstance(value, Path):
            value = str(value)
        lines.append(f"{name} = {_toml_value(value)}")

    if config.aliases:
        lines.extend(["", "[md_sorter.aliases]"])
        for key, terms in sorted(config.aliases.items()):
            lines.append(f"{_toml_value(key)} = {_toml_value(list(terms))}")

    changed_weights = {
        field_info.name: getattr(config.weights, field_info.name)
        for field_info in fields(ScoringWeights)
        if getattr(config.weights, field_info.name)
        != getattr(defaults.weights, field_info.name)
    }
    if changed_weights:
        lines.extend(["", "[md_sorter.weights]"])
        for key, value in sorted(changed_weights.items()):
            lines.append(f"{key} = {_toml_value(value)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
