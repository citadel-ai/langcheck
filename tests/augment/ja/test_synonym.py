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
        (["地球は青かった。", "先行きが不安だ。"
         ], 1, ["アースは青かった。", "将来が畏れだ。"]),
        (["地球は青かった。", "先行きが不安だ。"], 2, [
            "アースは青かった。", "アースは青かった。", "フューチャーが気掛かりだ。",
            "行く先が畏れだ。"
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
