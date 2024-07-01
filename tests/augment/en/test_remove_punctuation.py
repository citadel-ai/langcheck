from __future__ import annotations

import random
from typing import List

import pytest
from langcheck.augment.en import remove_punctuation


@pytest.mark.parametrize(
    "instances, num_perturbations, aug_char_p, expected",
    [
        ("Hello, world...!?", 1, 0.5, ["Hello, world!?"]),
        ("Hello, world...!?", 2, 0.5, ["Hello, world!?", "Hello, world?"]),
        (["Hello, world...!?"], 1, 0.5, ["Hello, world!?"]),
        (["Hello, world...!?"], 2, 0.5, ["Hello, world!?", "Hello, world?"]),
        (["Hello, world...!?", "!@#$%^&*()_+,./"
         ], 1, 0.5, ["Hello, world!?", "!^()+,/"]),
        (["Hello, world...!?", "!@#$%^&*()_+,./"], 2, 0.5,
         ["Hello, world!?", "Hello, world?", "#$^&(),", "@#$%^&()_+,."]),
    ],
)
def test_remove_punctuation(instances: List[str] | str, num_perturbations: int,
                            aug_char_p: float, expected: List[str]):
    seed = 42
    random.seed(seed)
    actual = remove_punctuation(instances,
                                aug_char_p=aug_char_p,
                                num_perturbations=num_perturbations)
    assert actual == expected
