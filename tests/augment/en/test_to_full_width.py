from __future__ import annotations

import random

import pytest

from langcheck.augment.en import to_full_width


@pytest.mark.parametrize(
    "instances, num_perturbations, aug_char_p, expected",
    [
        ########################################################################
        # Single input
        ########################################################################
        (
            "Hello, world!",
            1,
            0.9,
            ["Ｈｅｌｌｏ，\u3000ｗｏｒｌｄ！"],
        ),
        (
            "Hello, world!",
            2,
            0.9,
            ["Ｈｅｌｌｏ，\u3000ｗｏｒｌｄ！", "Ｈｅｌｌｏ，\u3000ｗｏｒｌd！"],
        ),
        (["Hello, world!"], 1, 0.9, ["Ｈｅｌｌｏ，\u3000ｗｏｒｌｄ！"]),
        (
            ["Hello, world!"],
            2,
            0.9,
            ["Ｈｅｌｌｏ，\u3000ｗｏｒｌｄ！", "Ｈｅｌｌｏ，\u3000ｗｏｒｌd！"],
        ),
        ("Hello, world!", 1, 0.1, ["Hｅllo, ｗoｒld！"]),
        ("Hello, world!", 2, 0.1, ["Hｅllo, ｗoｒld！", "Hello,\u3000world!"]),
        (["Hello, world!"], 1, 0.1, ["Hｅllo, ｗoｒld！"]),
        (
            ["Hello, world!"],
            2,
            0.1,
            ["Hｅllo, ｗoｒld！", "Hello,\u3000world!"],
        ),
        ########################################################################
        # Multiple inputs
        ########################################################################
        (
            ["HELLO, world!", "I'm hungry"],
            1,
            0.9,
            ["ＨＥＬＬＯ，\u3000ｗｏｒｌｄ！", "Ｉ＇ｍ\u3000ｈｕｎｇｒｙ"],
        ),
        (
            ["HELLO, world!", "I'm hungry"],
            2,
            0.9,
            [
                "ＨＥＬＬＯ，\u3000ｗｏｒｌｄ！",
                "ＨＥＬＬＯ，\u3000ｗｏｒｌd！",
                "Ｉ＇ｍ\u3000ｈｕｎgｒｙ",
                "Ｉ＇ｍ\u3000ｈｕｎｇｒｙ",
            ],
        ),
        (
            ["HELLO, world!", "I'm hungry"],
            1,
            0.1,
            ["HＥLLO, ｗoｒld！", "I'm huｎgry"],
        ),
        (
            ["HELLO, world!", "I'm hungry"],
            2,
            0.1,
            [
                "HＥLLO, ｗoｒld！",
                "HELLO,\u3000world!",
                "Ｉ＇m hungry",
                "I'm hｕngｒy",
            ],
        ),
    ],
)
def test_to_ful_width(
    instances: list[str] | str,
    num_perturbations: int,
    aug_char_p: float,
    expected: list[str],
):
    seed = 42
    random.seed(seed)
    actual = to_full_width(
        instances,
        aug_char_p=aug_char_p,
        num_perturbations=num_perturbations,
    )
    assert actual == expected
