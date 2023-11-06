import random
from typing import List

import pytest

from langcheck.augment.en import ocr_typo


@pytest.mark.parametrize(
    "texts, expected",
    [
        (["Hello, world!"], ["Hel1u, world!"]),
        (["Hello, world!", "I'm hungry"], ["Hel1u, world!", "I ' m hungry"]),
    ],
)
def test_ocr_typo(texts: List[str], expected: List[str]):
    seed = 42
    random.seed(seed)
    actual = ocr_typo(texts)
    assert actual == expected
