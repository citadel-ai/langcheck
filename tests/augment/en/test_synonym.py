from __future__ import annotations

import random

import pytest

from langcheck.augment.en import synonym


@pytest.mark.parametrize(
    "instances, num_perturbations, expected",
    [
        ("Hello, world!", 1, ["Hullo, world!"]),
        ("Hello, world!", 2, ["Hullo, world!", "Hello, earth!"]),
        (["Hello, world!"], 1, ["Hullo, world!"]),
        ("Hello, world!", 2, ["Hullo, world!", "Hello, earth!"]),
        (
            ["Hello, world!", "I have a pen. I have an apple"],
            1,
            ["Hullo, world!", "I have a pen. I have an malus pumila"],
        ),
        (
            ["Hello, world!", "I have a pen. I have an apple"],
            2,
            [
                "Hullo, world!",
                "Hello, earth!",
                "I get a pen. I have an apple",
                "I have a pen. I have an orchard apple tree",
            ],
        ),
    ],
)
def test_synonym(
    instances: list[str] | str, num_perturbations: int, expected: list[str]
):
    seed = 42
    random.seed(seed)
    actual = synonym(instances, num_perturbations=num_perturbations)
    assert actual == expected
