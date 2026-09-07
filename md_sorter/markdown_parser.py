"""Разбор Markdown с учётом специфики Obsidian.

Собственный парсер вместо внешней библиотеки: нужны именно obsidian-признаки
(``[[wiki-links]]``, инлайн-теги, frontmatter), а не HTML-рендеринг.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from .config import ScoringWeights
from .lexicon import expand_code_language
from .models import NoteRecord, ParsedNote, TokenBag
from .text import add_tokens, tokenize, tokenize_name

try:  # PyYAML используется, только если уже установлен; зависимостью не является.
    import yaml as _yaml
except ImportError:  # pragma: no cover - зависит от окружения
    _yaml = None

_FRONTMATTER_RE = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_FENCE_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[^\r\n`]*)$")
_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_INLINE_TAG_RE = re.compile(r"(?<![\w/#&])#([^\W\d_][\w/\-]*)", re.UNICODE)
_WIKILINK_RE = re.compile(r"\[\[([^\]\[|#^]+)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_YAML_LIST_ITEM_RE = re.compile(r"^[ \t]*-[ \t]*(.*)$")
_YAML_KEY_RE = re.compile(r"^(?P<key>[A-Za-z_][\w\- ]*):[ \t]*(?P<value>.*)$")
_HEX_COLOR_RE = re.compile(r"\A[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3}(?:[0-9a-fA-F]{2})?)?\Z")


def _is_hex_color(tag: str) -> bool:
    """Отличает цвет (``#fff``, ``#ff0000``) от настоящего тега.

    Слова вроде ``facade`` состоят только из hex-символов, поэтому они
    отбрасываются лишь при наличии цифры или при длине в три символа.
    """
    if not _HEX_COLOR_RE.match(tag):
        return False
    return len(tag) == 3 or any(char.isdigit() for char in tag)

#: Ключи frontmatter, из которых берутся теги.
_TAG_KEYS = ("tags", "tag", "keywords")
#: Ключи frontmatter, из которых берутся алиасы.
_ALIAS_KEYS = ("aliases", "alias", "title", "category", "type", "topic")


def _strip_quotes(value: str) -> str:
    """Убирает окружающие кавычки у скалярного значения YAML."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_frontmatter_fallback(block: str) -> dict[str, list[str]]:
    """Минимальный парсер подмножества YAML, достаточного для Obsidian.

    Поддерживаются скаляры, инлайн-списки ``[a, b]`` и блочные списки ``- a``.
    Вложенные структуры игнорируются: они не влияют на классификацию.
    """
    result: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        list_item = _YAML_LIST_ITEM_RE.match(line)
        if list_item and current_key is not None and (len(line) - len(line.lstrip())) > 0:
            value = _strip_quotes(list_item.group(1))
            if value:
                result.setdefault(current_key, []).append(value)
            continue
        if list_item and current_key is not None and line.lstrip().startswith("-"):
            value = _strip_quotes(list_item.group(1))
            if value:
                result.setdefault(current_key, []).append(value)
            continue
        key_match = _YAML_KEY_RE.match(line)
        if not key_match:
            continue
        current_key = key_match.group("key").strip().lower()
        raw_value = key_match.group("value").strip()
        result.setdefault(current_key, [])
        if not raw_value:
            continue
        if raw_value.startswith("[") and raw_value.endswith("]"):
            items = [_strip_quotes(part) for part in raw_value[1:-1].split(",")]
            result[current_key].extend(item for item in items if item)
        else:
            value = _strip_quotes(raw_value)
            if value:
                result[current_key].append(value)
    return result


def _flatten_yaml_value(value: object) -> list[str]:
    """Разворачивает произвольное значение YAML в список строк."""
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for key, nested in value.items():
            flattened.append(str(key))
            flattened.extend(_flatten_yaml_value(nested))
        return flattened
    if isinstance(value, Iterable):
        flattened = []
        for item in value:
            flattened.extend(_flatten_yaml_value(item))
        return flattened
    return []


def parse_frontmatter(text: str) -> tuple[dict[str, list[str]], str]:
    """Извлекает YAML frontmatter и возвращает ``(данные, остаток текста)``."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    block = match.group(1)
    body = text[match.end():]
    if _yaml is not None:
        try:
            loaded = _yaml.safe_load(block)
        except Exception:  # noqa: BLE001 - битый YAML не должен ломать разбор
            loaded = None
        if isinstance(loaded, Mapping):
            return (
                {str(key).lower(): _flatten_yaml_value(value) for key, value in loaded.items()},
                body,
            )
    return _parse_frontmatter_fallback(block), body


def split_code_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Разделяет текст на обычную часть и список ``(язык, код)``.

    Кодовые блоки исключаются из поиска тегов и заголовков, но их содержимое
    сохраняется: импорты и команды — сильный признак категории.
    """
    lines = text.splitlines()
    plain: list[str] = []
    blocks: list[tuple[str, str]] = []
    fence: str | None = None
    language = ""
    buffer: list[str] = []

    for line in lines:
        if fence is None:
            match = _FENCE_RE.match(line)
            if match:
                fence = match.group("fence")[0] * 3
                raw_language = match.group("lang").strip().strip("{}")
                language = raw_language.split()[0] if raw_language else ""
                buffer = []
                continue
            plain.append(line)
        else:
            stripped = line.strip()
            if stripped.startswith(fence) and set(stripped) == {fence[0]}:
                blocks.append((language, "\n".join(buffer)))
                fence = None
                language = ""
                buffer = []
                continue
            buffer.append(line)

    if fence is not None:  # незакрытый блок — считаем кодом до конца файла
        blocks.append((language, "\n".join(buffer)))

    return "\n".join(plain), blocks


def extract_headings(text: str) -> list[tuple[int, str]]:
    """Возвращает список ``(уровень, текст)`` для ATX-заголовков."""
    headings: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            title = match.group(2).strip()
            if title:
                headings.append((len(match.group(1)), title))
    return headings


def extract_inline_tags(text: str) -> list[str]:
    """Находит инлайн-теги Obsidian, включая вложенные (``#linux/arch``)."""
    tags: list[str] = []
    for raw in _INLINE_TAG_RE.findall(text):
        tag = raw.rstrip("/-")
        if not tag or tag.isdigit():
            continue
        if _is_hex_color(tag):  # «#fff» — это цвет, а не тег
            continue
        tags.append(tag)
    return tags


def extract_wiki_links(text: str) -> list[str]:
    """Находит цели ``[[wiki-links]]`` (без алиасов и якорей)."""
    return [link.strip() for link in _WIKILINK_RE.findall(text) if link.strip()]


def _tag_tokens(tag: str) -> list[str]:
    """Токенизирует тег с учётом вложенности ``parent/child``."""
    return tokenize(tag.replace("/", " ").replace("-", " ").replace("_", " "))


def parse_note(
    record: NoteRecord,
    content: str,
    weights: ScoringWeights,
    *,
    truncated: bool = False,
) -> ParsedNote:
    """Строит взвешенный набор признаков заметки.

    Args:
        record: метаданные файла.
        content: текст заметки (возможно, усечённый).
        weights: веса признаков.
        truncated: был ли текст обрезан по ``max_analysis_chars``.

    Returns:
        Заполненный :class:`~md_sorter.models.ParsedNote`.
    """
    frontmatter, body_with_code = parse_frontmatter(content)
    plain_body, code_blocks = split_code_blocks(body_with_code)

    headings = extract_headings(plain_body)
    inline_tags = extract_inline_tags(plain_body)
    wiki_links = extract_wiki_links(plain_body)

    frontmatter_tags: list[str] = []
    for key in _TAG_KEYS:
        frontmatter_tags.extend(frontmatter.get(key, []))
    aliases: list[str] = []
    for key in _ALIAS_KEYS:
        aliases.extend(frontmatter.get(key, []))

    code_languages: list[str] = []
    for language, _ in code_blocks:
        code_languages.extend(expand_code_language(language))

    # «Сильные» признаки собираются отдельно: по ним классификатор делает
    # второй проход, не размывая сигнал основным текстом заметки.
    strong: TokenBag = {}
    add_tokens(strong, tokenize_name(record.stem), weights.filename)
    for tag in frontmatter_tags:
        add_tokens(strong, _tag_tokens(tag), weights.frontmatter_tags)
    for tag in inline_tags:
        add_tokens(strong, _tag_tokens(tag), weights.inline_tags)
    for alias in aliases:
        add_tokens(strong, tokenize(alias), weights.aliases)

    title = ""
    body_headings: list[str] = []
    for level, heading in headings:
        if level == 1 and not title:
            title = heading
            add_tokens(strong, tokenize(heading), weights.title)
        elif level <= 3:
            add_tokens(strong, tokenize(heading), weights.heading)
        else:
            body_headings.append(heading)
    if not title and headings:
        title = headings[0][1]

    add_tokens(strong, code_languages, weights.code_language)
    for link in wiki_links:
        add_tokens(strong, tokenize_name(link), weights.wiki_link)

    bag: TokenBag = dict(strong)

    # Исходный каталог — подсказка, но не «имя, которое дал заметке автор»:
    # он попадает только в полный мешок, иначе заметка в каталоге Games
    # считалась бы названной «Games» и получала бы бонус за совпадение имени.
    source_dir = record.source_dir
    if source_dir.as_posix() not in (".", ""):
        for part in source_dir.parts:
            add_tokens(bag, tokenize_name(part), weights.source_path)

    for heading in body_headings:
        add_tokens(bag, tokenize(heading), weights.body)
    for _, code in code_blocks:
        add_tokens(bag, tokenize(code), weights.code_body)

    text_for_body = _MD_LINK_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", plain_body)
    add_tokens(bag, tokenize(text_for_body), weights.body)

    return ParsedNote(
        record=record,
        tokens=bag,
        strong_tokens=strong,
        title=title,
        tags=tuple(dict.fromkeys([*frontmatter_tags, *inline_tags])),
        aliases=tuple(dict.fromkeys(aliases)),
        headings=tuple(heading for _, heading in headings),
        wiki_links=tuple(dict.fromkeys(wiki_links)),
        code_languages=tuple(dict.fromkeys(code_languages)),
        truncated=truncated,
    )
