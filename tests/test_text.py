"""Тесты токенизации и векторных операций."""

from __future__ import annotations

import pytest

from md_sorter.text import (
    add_tokens,
    normalize_token,
    compute_idf,
    cosine,
    subtract_bag,
    to_vector,
    tokenize,
    tokenize_name,
)


def test_tokenize_handles_russian_and_english() -> None:
    tokens = tokenize("Установка пакетов в Termux через pkg install")
    assert "termux" in tokens
    assert "pkg" in tokens
    assert "в" not in tokens  # стоп-слово


def test_tokenize_preserves_special_terms() -> None:
    assert "cpp" in tokenize("Пишем на C++ с CMake")
    assert "csharp" in tokenize("Проект на C# и .NET")
    assert "dotnet" in tokenize("Проект на C# и .NET")


def test_tokenize_splits_camel_case_and_keeps_whole() -> None:
    tokens = tokenize("camelCaseName")
    assert "camel" in tokens
    assert "case" in tokens
    assert "camelcasenam" in tokens or "camelcasename" in tokens


def test_tokenize_drops_digits_and_short_noise() -> None:
    tokens = tokenize("2024 x qq")
    assert "2024" not in tokens
    assert "x" not in tokens
    assert "qq" in tokens


def test_tokenize_name_splits_separators() -> None:
    assert set(tokenize_name("Dev-Tools_And.Stuff")) == {"dev", "tool", "stuff"}


def test_tokenize_name_falls_back_to_raw_name() -> None:
    """Односимвольный каталог обязан иметь собственный токен."""
    assert tokenize_name("A") == ["a"]
    assert tokenize_name("42") == ["42"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("секреты", "секрет"),
        ("демоны", "демон"),
        ("пакеты", "пакет"),
        ("заметки", "заметка"),
        ("настройка", "настройки"),
        ("установки", "установка"),
        ("команды", "команда"),
        ("сделав", "сделать"),
        ("файлы", "файл"),
    ],
)
def test_russian_forms_share_a_stem(first: str, second: str) -> None:
    """Без этого русскоязычные заметки почти не совпадали бы с категориями."""
    assert normalize_token(first) == normalize_token(second)


@pytest.mark.parametrize("word", ["linux", "termux", "python", "pacman", "minecraft"])
def test_latin_terms_survive_stemming(word: str) -> None:
    """Ключевые технические термины не должны искажаться."""
    assert normalize_token(word) == word


def test_add_tokens_accumulates_weight() -> None:
    bag: dict[str, float] = {}
    add_tokens(bag, ["python", "python"], 2.0)
    assert bag["python"] == 4.0


def test_subtract_bag_removes_exhausted_tokens() -> None:
    result = subtract_bag({"a": 2.0, "b": 1.0}, {"a": 2.0, "b": 0.25}, 1.0)
    assert "a" not in result
    assert result["b"] == 0.75


def test_cosine_of_identical_vectors_is_one() -> None:
    idf, default = compute_idf([["a", "b"], ["b", "c"]])
    vector = to_vector({"a": 1.0, "b": 2.0}, idf, default)
    assert cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_of_disjoint_vectors_is_zero() -> None:
    idf, default = compute_idf([["a"], ["b"]])
    left = to_vector({"a": 1.0}, idf, default)
    right = to_vector({"b": 1.0}, idf, default)
    assert cosine(left, right) == 0.0


def test_to_vector_of_empty_bag_is_empty() -> None:
    idf, default = compute_idf([["a"]])
    assert to_vector({}, idf, default) == {}
