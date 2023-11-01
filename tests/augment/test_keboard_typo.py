import random
from typing import List

import pytest

from langcheck.augment.en import keyboard_typo


@pytest.mark.parametrize(
    "texts, expected",
    [
        (["Hello, world!"], ["HePlo, wLrld!"]),
        (["Hello, world!", "I'm hungry"], ["HePlo, wLrld!", "I ' m hungrt"]),
    ],
)
def test_keyboard_typo(texts: List[str], expected: List[str]):
    seed = 42
    random.seed(seed)
    actual = keyboard_typo(texts)
    assert actual == expected
