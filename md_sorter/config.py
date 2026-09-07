"""Конфигурация сортировщика: значения по умолчанию и чтение TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .models import SORTED_DIR_NAME

#: Имена файлов конфигурации, которые ищутся в ROOT (в порядке приоритета).
CONFIG_FILENAMES: tuple[str, ...] = ("md_sorter.toml", ".md_sorter.toml")

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

    # --- правила ранжирования ---
    parent_support: float = 0.35
    depth_bonus: float = 0.06
    name_match_boost: float = 1.15
    source_path_prior: float = 0.10
    descendant_epsilon: float = 0.10


@dataclass(slots=True)
class Config:
    """Полный набор параметров запуска."""

    root: Path = field(default_factory=Path.cwd)
    dry_run: bool = False
    verbose: bool = False
    debug: bool = False
    quiet: bool = False
    log_file: Path | None = None
    color: bool | None = None

    # Порог откалиброван по наблюдаемому разделению: уверенные решения
    # набирают 0.25-0.99, отсутствие сигнала — 0.00-0.16.
    threshold: float = 0.22
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

    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    @property
    def sorted_dir(self) -> Path:
        """Абсолютный путь к каталогу с результатом сортировки."""
        return self.root / SORTED_DIR_NAME

    def effective_copy_mode(self) -> str:
        """Фактический режим переноса с учётом защиты исходных файлов."""
        if self.copy_mode == "move" and self.allow_move:
            return "move"
        return "copy"


class ConfigError(RuntimeError):
    """Ошибка чтения или валидации конфигурации."""


def find_config_file(root: Path) -> Path | None:
    """Ищет файл конфигурации в корневом каталоге."""
    for name in CONFIG_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
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
        elif isinstance(current, Path) or key in {"root", "log_file"}:
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
    if config.uncertain_strategy not in {"review", "ancestor", "root", "skip"}:
        raise ConfigError(
            "uncertain_strategy: допустимы review, ancestor, root, skip"
        )
    if config.copy_mode not in {"copy", "move"}:
        raise ConfigError("copy_mode: допустимы copy или move")
    if config.copy_mode == "move" and not config.allow_move:
        raise ConfigError(
            "Режим move требует явного флага --allow-move "
            "(по умолчанию исходные файлы неприкосновенны)"
        )
    if config.max_file_size <= 0:
        raise ConfigError("max_file_size должен быть положительным")
    if config.max_analysis_chars <= 0:
        raise ConfigError("max_analysis_chars должен быть положительным")
    if config.jobs < 0:
        raise ConfigError("jobs не может быть отрицательным")
