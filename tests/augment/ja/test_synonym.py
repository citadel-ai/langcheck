from __future__ import annotations

import random

import pytest
from langcheck.augment.ja import synonym


@pytest.mark.parametrize(
    "instances, num_perturbations, expected",
    [
        ("地球は青かった。", 1, ["アースは青かった。"]),
        ("地球は青かった。", 2, ["アースは青かった。", "アースは青かった。"]),
        (["地球は青かった。"], 1, ["アースは青かった。"]),
        ("地球は青かった。", 2, ["アースは青かった。", "アースは青かった。"]),
        (["地球は青かった。", "人間にとっては小さな一歩だが人類にとっては偉大な一歩だ。"], 1, [
            "アースは青かった。", "人間にとっては小さな一歩だが人類にとってはグレイトな一歩だ。"]),
        (["地球は青かった。", "人間にとっては小さな一歩だが人類にとっては偉大な一歩だ。"], 2, [
            "アースは青かった。", "アースは青かった。", "人間にとっては小さな一歩だが人類にとってはグレートな一歩だ。",
            "人間にとっては小さな一歩だが人類にとってはグレートな一歩だ。"
        ]),
    ],
)
def test_synonym(instances: list[str] | str, num_perturbations: int,
                 expected: list[str]):
    seed = 42
    random.seed(seed)
    actual = synonym(instances, num_perturbations=num_perturbations)
    print(actual)
    assert actual == expected
