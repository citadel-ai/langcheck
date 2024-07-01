from __future__ import annotations

import random
from typing import List

import pytest
from langcheck.augment.en import keyboard_typo


@pytest.mark.parametrize(
    "instances, num_perturbations, expected",
    [
        ("Hello, world!", 1, ["HePlo, wLrld!"]),
        ("Hello, world!", 2, ["HePlo, wLrld!", "Helll, Aorld!"]),
        (["Hello, world!"], 1, ["HePlo, wLrld!"]),
        (["Hello, world!"], 2, ["HePlo, wLrld!", "Helll, Aorld!"]),
        (["Hello, world!", "I'm hungry"], 1, ["HePlo, wLrld!", "I ' m hungrt"]),
        (["Hello, world!", "I'm hungry"], 2,
         ["HePlo, wLrld!", "Helll, Aorld!", "I ' m hKngry", "I ' m hungGy"]),
    ],
)
def test_keyboard_typo(instances: List[str] | str, num_perturbations: int,
                       expected: List[str]):
    seed = 42
    random.seed(seed)
    actual = keyboard_typo(instances, num_perturbations=num_perturbations)
    assert actual == expected
