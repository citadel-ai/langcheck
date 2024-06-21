from __future__ import annotations

import random
from typing import List

import pytest
from langcheck.augment.en import change_case


@pytest.mark.parametrize(
    "instances, num_perturbations, aug_char_p, to_case, expected",
    [
        ########################################################################
        # To uppercase, single input
        ########################################################################
        ("Hello, world!", 1, 0.9, 'uppercase', ["HELLO, WORLD!"]),
        ("Hello, world!", 2, 0.9, 'uppercase',
         ["HELLO, WORLD!", "HELLO, WORLd!"]),
        (["Hello, world!"], 1, 0.9, 'uppercase', ["HELLO, WORLD!"]),
        (["Hello, world!"
         ], 2, 0.9, 'uppercase', ["HELLO, WORLD!", "HELLO, WORLd!"]),
        ("Hello, world!", 1, 0.1, 'uppercase', ["HEllo, WoRld!"]),
        ("Hello, world!", 2, 0.1, 'uppercase',
         ["HEllo, WoRld!", "Hello, world!"]),
        (["Hello, world!"], 1, 0.1, 'uppercase', ["HEllo, WoRld!"]),
        (["Hello, world!"
         ], 2, 0.1, 'uppercase', ["HEllo, WoRld!", "Hello, world!"]),
        ########################################################################
        # To lowercase, single input
        ########################################################################
        ("HELLO, world!", 1, 0.9, 'lowercase', ["hello, world!"]),
        ("HELLO, world!", 2, 0.9, 'lowercase',
         ["hello, world!", "hello, world!"]),
        (["HELLO, world!"], 1, 0.9, 'lowercase', ["hello, world!"]),
        (["HELLO, world!"
         ], 2, 0.9, 'lowercase', ["hello, world!", "hello, world!"]),
        ("HELLO, world!", 1, 0.1, 'lowercase', ["HeLLO, world!"]),
        ("HELLO, world!", 2, 0.1, 'lowercase',
         ["HeLLO, world!", "HELLO, world!"]),
        (["HELLO, world!"], 1, 0.1, 'lowercase', ["HeLLO, world!"]),
        (["HELLO, world!"
         ], 2, 0.1, 'lowercase', ["HeLLO, world!", "HELLO, world!"]),
        ########################################################################
        # Multiple inputs
        ########################################################################
        (["HELLO, world!", "I'm hungry"
         ], 1, 0.9, 'lowercase', ["hello, world!", "i'm hungry"]),
        (["HELLO, world!", "I'm hungry"], 2, 0.9, 'lowercase',
         ['hello, world!', 'hello, world!', "i'm hungry", "i'm hungry"]),
        (["HELLO, world!", "I'm hungry"
         ], 1, 0.1, 'uppercase', ["HELLO, WoRld!", "I'm huNgry"]),
        (["HELLO, world!", "I'm hungry"], 2, 0.1, 'uppercase',
         ["HELLO, WoRld!", "HELLO, world!", "I'm hungry", "I'm hUngRy"])
    ],
)
def test_change_case(instances: List[str] | str, num_perturbations: int,
                     aug_char_p: float, to_case: str, expected: List[str]):
    seed = 42
    random.seed(seed)
    actual = change_case(instances,
                         to_case=to_case,
                         aug_char_p=aug_char_p,
                         num_perturbations=num_perturbations)
    assert actual == expected
