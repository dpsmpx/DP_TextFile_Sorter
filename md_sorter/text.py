"""Токенизация текста и операции над разреженными векторами.

Модуль намеренно не зависит от внешних библиотек: базовый классификатор
должен работать в любом окружении, включая Termux на Android.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Container, Iterable, Mapping, Sequence

from .models import TokenBag

# Термины, которые обычная токенизация разрушила бы («C++» -> «c»).
_SPECIAL_TERMS: dict[str, str] = {
    "c/c++": "cpp",
    "c++": "cpp",
    "g++": "cpp",
    "c#": "csharp",
    "f#": "fsharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "vue.js": "vuejs",
    "next.js": "nextjs",
    "objective-c": "objectivec",
}

_SPECIAL_RE = re.compile(
    "|".join(re.escape(term) for term in sorted(_SPECIAL_TERMS, key=len, reverse=True)),
    re.IGNORECASE,
)

# Буквы и цифры любого алфавита; «_» служит разделителем (snake_case).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CAMEL_RE = re.compile(r"[A-ZА-ЯЁ]?[a-zа-яё0-9]+|[A-ZА-ЯЁ]+(?![a-zа-яё])")
_HAS_CYRILLIC_RE = re.compile(r"[а-яё]")

_STOPWORDS_EN = frozenset(
    """
    a an and are as at be but by can do does for from had has have how i if in
    into is it its of on or our so than that the their then there these they
    this to was were what when where which who why will with you your not no
    all any more most other some such only own same too very just about after
    before also here over under again once each few many much both
    """.split()
)

_STOPWORDS_RU = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за
    бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну
    вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там
    потом себя ничего ей может они тут где есть надо ней для мы тебя их чем
    была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того
    потому этого какой совсем ним здесь этом один почти мой тем чтобы нее были
    куда зачем всех никогда можно при наконец два об другой хоть после над
    больше тот через эти нас про всего них какая много разве три эту моя
    впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более
    всегда конечно всю между это как-то который которые которая
    """.split()
)

STOPWORDS = _STOPWORDS_EN | _STOPWORDS_RU

# Короткие токены, которые всё же несут смысл.
_SHORT_KEEP = frozenset(
    {"c", "r", "go", "js", "ts", "vm", "os", "ai", "ml", "db", "3d", "ci", "cd"}
)

_EN_SUFFIXES: tuple[str, ...] = ("ings", "ing", "ies", "ers", "er", "ed", "es", "s")

# --- Русский стеммер (алгоритм Snowball / Porter для русского языка) ---------
# Без него «секреты» и «секрет», «пакеты» и «пакет» считались бы разными
# токенами, и русскоязычные заметки почти не совпадали бы с категориями.
_RU_VOWELS = frozenset("аеиоуыэюяё")

_RU_PERFECTIVE_GERUND_1 = ("вшись", "вши", "в")
_RU_PERFECTIVE_GERUND_2 = ("ывшись", "ившись", "ывши", "ивши", "ыв", "ив")
_RU_ADJECTIVE = (
    "ими", "ыми", "его", "ого", "ему", "ому", "ее", "ие", "ые", "ое", "ей",
    "ий", "ый", "ой", "ем", "им", "ым", "ом", "их", "ых", "ую", "юю", "ая",
    "яя", "ою", "ею",
)
_RU_PARTICIPLE_1 = ("ющ", "нн", "вш", "ем", "щ")
_RU_PARTICIPLE_2 = ("ующ", "ивш", "ывш")
_RU_REFLEXIVE = ("ся", "сь")
_RU_VERB_1 = (
    "ешь", "нно", "ете", "йте", "ла", "на", "ли", "ем", "ло", "но", "ет",
    "ют", "ны", "ть", "й", "л", "н",
)
_RU_VERB_2 = (
    "ейте", "уйте", "ила", "ыла", "ена", "ите", "или", "ыли", "ило", "ыло",
    "ено", "ует", "уют", "ены", "ить", "ыть", "ишь", "ей", "уй", "ил", "ыл",
    "им", "ым", "ен", "ят", "ит", "ыт", "ую", "ю",
)
_RU_NOUN = (
    "иями", "ями", "ами", "иях", "ией", "иям", "ием", "иях", "ях", "ах", "ию",
    "ью", "ия", "ья", "ев", "ов", "ие", "ье", "еи", "ии", "ей", "ой", "ий",
    "ям", "ем", "ам", "ом", "а", "е", "и", "й", "о", "у", "ы", "ь", "ю", "я",
)
_RU_SUPERLATIVE = ("ейше", "ейш")
_RU_DERIVATIONAL = ("ость", "ост")


def _sorted_by_length(endings: tuple[str, ...]) -> tuple[str, ...]:
    """Сортирует окончания по убыванию длины: сначала пробуем самое длинное."""
    return tuple(sorted(endings, key=len, reverse=True))


_RU_GROUPS: dict[str, tuple[str, ...]] = {
    name: _sorted_by_length(endings)
    for name, endings in {
        "gerund1": _RU_PERFECTIVE_GERUND_1,
        "gerund2": _RU_PERFECTIVE_GERUND_2,
        "adjective": _RU_ADJECTIVE,
        "participle1": _RU_PARTICIPLE_1,
        "participle2": _RU_PARTICIPLE_2,
        "reflexive": _RU_REFLEXIVE,
        "verb1": _RU_VERB_1,
        "verb2": _RU_VERB_2,
        "noun": _RU_NOUN,
        "superlative": _RU_SUPERLATIVE,
        "derivational": _RU_DERIVATIONAL,
    }.items()
}


def _ru_regions(word: str) -> tuple[int, int]:
    """Возвращает границы областей ``RV`` и ``R2`` по определению Snowball."""
    rv = len(word)
    r2 = len(word)
    state = 0
    for index, char in enumerate(word):
        is_vowel = char in _RU_VOWELS
        if state == 0 and is_vowel:
            rv = index + 1
            state = 1
        elif state == 1 and not is_vowel:
            state = 2
        elif state == 2 and is_vowel:
            state = 3
        elif state == 3 and not is_vowel:
            r2 = index + 1
            state = 4
    return rv, r2


def _cut(word: str, rv: int, group: str, *, preceded_by: str = "") -> str | None:
    """Пробует отсечь окончание из группы, если оно лежит в области RV.

    Args:
        preceded_by: если задано, окончанию должна предшествовать одна из
            указанных букв (правило Snowball для групп «после а/я»).

    Returns:
        Усечённое слово либо ``None``, если подходящего окончания нет.
    """
    for ending in _RU_GROUPS[group]:
        if not word.endswith(ending):
            continue
        start = len(word) - len(ending)
        if start < rv:
            continue
        if preceded_by and (start == 0 or word[start - 1] not in preceded_by):
            continue
        # По правилам Snowball удаляется только само окончание;
        # предшествующая «а»/«я» — условие, а не часть суффикса.
        return word[:start]
    return None


def _stem_russian(word: str) -> str:
    """Стеммер русского языка по алгоритму Snowball (Porter)."""
    word = word.replace("ё", "е")
    rv, r2 = _ru_regions(word)

    # Шаг 1: деепричастие, иначе возвратная частица + прилагательное/глагол/сущ.
    step1 = _cut(word, rv, "gerund1", preceded_by="ая") or _cut(word, rv, "gerund2")
    if step1 is None:
        stripped = _cut(word, rv, "reflexive")
        if stripped is not None:
            word = stripped
        adjectival = _cut(word, rv, "adjective")
        if adjectival is not None:
            participle = (
                _cut(adjectival, rv, "participle1", preceded_by="ая")
                or _cut(adjectival, rv, "participle2")
            )
            word = participle if participle is not None else adjectival
        else:
            verb = _cut(word, rv, "verb1", preceded_by="ая") or _cut(word, rv, "verb2")
            word = verb if verb is not None else (_cut(word, rv, "noun") or word)
    else:
        word = step1

    # Шаг 2: отсечение «и».
    if word.endswith("и") and len(word) - 1 >= rv:
        word = word[:-1]

    # Шаг 3: словообразовательный суффикс в области R2.
    for ending in _RU_GROUPS["derivational"]:
        if word.endswith(ending) and len(word) - len(ending) >= r2:
            word = word[: -len(ending)]
            break

    # Шаг 4: превосходная степень, удвоенное «н», мягкий знак.
    if word.endswith("нн"):
        word = word[:-1]
    else:
        superlative = _cut(word, rv, "superlative")
        if superlative is not None:
            word = superlative
            if word.endswith("нн"):
                word = word[:-1]
    if word.endswith("ь"):
        word = word[:-1]

    return word


def _apply_special_terms(text: str) -> str:
    """Заменяет термины вида ``C++`` на токенизируемые эквиваленты."""

    def _sub(match: re.Match[str]) -> str:
        return f" {_SPECIAL_TERMS[match.group(0).lower()]} "

    return _SPECIAL_RE.sub(_sub, text)


def _split_camel(raw: str) -> list[str]:
    """Разбивает ``camelCase``/``PascalCase`` на части, сохраняя исходный токен."""
    parts = _CAMEL_RE.findall(raw)
    if len(parts) <= 1:
        return [raw]
    return [raw, *parts]


def _stem(token: str) -> str:
    """Приводит токен к основе: Snowball для русского, лёгкое усечение для английского.

    Стемминг применяется одинаково к заметкам и к профилям категорий, поэтому
    важна не лингвистическая точность, а согласованность обеих сторон.
    """
    if _HAS_CYRILLIC_RE.search(token):
        return _stem_russian(token)
    for suffix in _EN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def normalize_token(raw: str) -> str:
    """Приводит токен к канонической форме или возвращает пустую строку."""
    token = raw.lower()
    if not token:
        return ""
    if token.isdigit():
        return ""
    if len(token) < 2 and token not in _SHORT_KEEP:
        return ""
    if token in STOPWORDS:
        return ""
    stemmed = _stem(token)
    if len(stemmed) < 2 and stemmed not in _SHORT_KEEP:
        return token
    return stemmed


def tokenize(text: str) -> list[str]:
    """Разбивает произвольный текст на нормализованные токены."""
    if not text:
        return []
    prepared = _apply_special_terms(text)
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(prepared):
        for part in _split_camel(raw):
            token = normalize_token(part)
            if token:
                tokens.append(token)
    return tokens


def tokenize_name(name: str) -> list[str]:
    """Токенизирует имя файла или каталога (``Dev-Tools`` -> ``dev``, ``tools``).

    Если обычная токенизация ничего не даёт (имя вроде ``A`` или ``42``),
    возвращается само имя: у каталога всегда должен быть свой токен, иначе
    его профиль будет состоять только из чужого содержимого.
    """
    tokens = tokenize(name.replace("-", " ").replace(".", " "))
    if tokens:
        return tokens
    fallback = name.strip().lower()
    return [fallback] if fallback else []


def add_tokens(bag: TokenBag, tokens: Iterable[str], weight: float) -> None:
    """Добавляет токены в мешок с указанным весом."""
    if weight <= 0.0:
        return
    for token in tokens:
        bag[token] = bag.get(token, 0.0) + weight


def merge_bag(target: TokenBag, source: Mapping[str, float], factor: float = 1.0) -> None:
    """Прибавляет содержимое ``source`` к ``target`` с масштабом ``factor``."""
    if factor == 0.0:
        return
    for token, value in source.items():
        target[token] = target.get(token, 0.0) + value * factor


def subtract_bag(
    source: Mapping[str, float],
    other: Mapping[str, float],
    factor: float,
) -> TokenBag:
    """Возвращает копию ``source`` за вычетом ``other * factor`` (без отрицательных)."""
    result: TokenBag = dict(source)
    for token, value in other.items():
        if token not in result:
            continue
        remainder = result[token] - value * factor
        if remainder > 1e-9:
            result[token] = remainder
        else:
            del result[token]
    return result


def compute_idf(documents: Sequence[Iterable[str]]) -> tuple[dict[str, float], float]:
    """Считает IDF по корпусу документов.

    Возвращает пару ``(таблица IDF, значение по умолчанию)``; значение по
    умолчанию используется для токенов, не встречавшихся в корпусе.
    """
    total = max(1, len(documents))
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    idf = {
        token: math.log((total + 1) / (freq + 1)) + 1.0
        for token, freq in document_frequency.items()
    }
    return idf, math.log(total + 1) + 1.0


def sublinear(weight: float) -> float:
    """Сжимает вес признака по правилу ``1 + ln(w)``.

    Без этого один редкий токен способен занять почти всю норму вектора: слово
    из имени файла получает вес 3.0, вместе с заголовком и текстом — 6.5, и
    после умножения на максимальный IDF оно вытесняет содержательные термины.
    Логарифм сохраняет порядок важности признаков, но сокращает разрыв.
    """
    return 1.0 + math.log(weight) if weight > 1.0 else weight


def to_vector(
    bag: Mapping[str, float],
    idf: Mapping[str, float],
    default_idf: float,
    vocabulary: Container[str] | None = None,
) -> TokenBag:
    """Строит L2-нормализованный IDF-взвешенный вектор из мешка токенов.

    Args:
        vocabulary: если задан, токены вне его отбрасываются до нормировки.
            Слово, которого нет ни в одной категории и ни в одной разложенной
            заметке, не может повлиять на выбор категории, но при нормировке
            съедает долю нормы и глушит содержательные термины.
    """
    vector: TokenBag = {}
    for token, weight in bag.items():
        if weight <= 0.0:
            continue
        if vocabulary is not None and token not in vocabulary:
            continue
        vector[token] = sublinear(weight) * idf.get(token, default_idf)
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 0.0:
        return {}
    return {token: value / norm for token, value in vector.items()}


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Косинусная близость двух уже нормализованных векторов."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    total = 0.0
    for token, value in left.items():
        other = right.get(token)
        if other is not None:
            total += value * other
    return max(0.0, min(1.0, total))


def top_shared_terms(
    left: Mapping[str, float],
    right: Mapping[str, float],
    limit: int = 5,
) -> list[tuple[str, float]]:
    """Возвращает токены с наибольшим вкладом в косинусную близость."""
    if not left or not right:
        return []
    if len(left) > len(right):
        left, right = right, left
    shared = [
        (token, value * right[token])
        for token, value in left.items()
        if token in right
    ]
    shared.sort(key=lambda item: item[1], reverse=True)
    return shared[:limit]
