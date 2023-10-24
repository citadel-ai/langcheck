from __future__ import annotations

import random

import pytest

from langcheck.augment.en import synonym


@pytest.mark.parametrize(
    "texts, expected",
    [
        (["Hello, world!"], ["Hullo, world!"]),
        (["Hello, world!", "I have a pen. I have an apple"
         ], ["Hullo, world!", "I have a pen. I have an malus pumila"]),
    ],
)
def test_synonym(texts: list[str], expected: list[str]):
    seed = 42
    random.seed(seed)
    actual = synonym(texts)
    print(actual)
    assert actual == expected
