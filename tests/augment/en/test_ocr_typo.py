from __future__ import annotations

import random

import pytest

from langcheck.augment.en import ocr_typo


@pytest.mark.parametrize(
    "instances, num_perturbations, expected",
    [
        ("Hello, world!", 1, ["Hel1u, world!"]),
        ("Hello, world!", 2, ["Hel1u, world!", "Hello, w0r1d!"]),
        (["Hello, world!"], 1, ["Hel1u, world!"]),
        (["Hello, world!"], 2, ["Hel1u, world!", "Hello, w0r1d!"]),
        (["Hello, world!", "I'm hungry"], 1, ["Hel1u, world!", "I ' m hungry"]),
        (
            ["Hello, world!", "I'm hungry"],
            2,
            ["Hel1u, world!", "Hello, w0r1d!", "1 ' m hongky", "I ' m hun9ky"],
        ),
    ],
)
def test_ocr_typo(
    instances: list[str] | str, num_perturbations: int, expected: list[str]
):
    seed = 42
    random.seed(seed)
    actual = ocr_typo(instances, num_perturbations=num_perturbations)
    assert actual == expected
