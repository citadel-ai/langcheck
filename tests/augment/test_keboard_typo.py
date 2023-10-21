import random
from typing import List

import pytest

from langcheck.augment.en import keyboard_typo


@pytest.mark.parametrize(
    "texts, expected",
    [
        (["Hello, world!"], ["HF;lo, eorlE!"]),
        (["Hello, world!", "I'm hungry"], ["HF;lo, eorlE!", "I ' m Y Tngry"]),
    ],
)
def test_keyboard_typo(texts: List[str], expected: List[str]):
    seed = 42
    random.seed(seed)
    actual = keyboard_typo(texts)
    assert actual == expected
