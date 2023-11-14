import random
from typing import List, Optional

import pytest

from langcheck.augment.en import gender


def test_invalid_to_gender():
    with pytest.raises(ValueError):
        gender("text", to_gender="invalid option")


def test_invalid_input():
    with pytest.raises(TypeError):
        gender(1)  # type: ignore[reportGeneralTypeIssues]


@pytest.mark.parametrize(
    "texts",
    [
        (["He cooks by himself.", "This is his dog.", "I gave him a book."]),
        (["She cooks by herself.", "This is her dog.", "I gave her a book."]),
        ([
            "They cooks by themselves.", "This is their dog.",
            "I gave them a book."
        ]),
    ],
)
@pytest.mark.parametrize("to_gender, expected", [
    (None, [
        "They cooks by themselves.", "This is their dog.", "I gave them a book."
    ]),
    ("female",
     ["She cooks by herself.", "This is her dog.", "I gave her a book."]),
    ("male", ["He cooks by himself.", "This is his dog.", "I gave him a book."
             ]),
    ("neutral",
     ['Xe cooks by xyrself.', 'This is xyr dog.', 'I gave xem a book.']),
    ("plural", [
        "They cooks by themselves.", "This is their dog.", "I gave them a book."
    ]),
])
def test_gender(
    texts: List[str],
    to_gender: Optional[str],
    expected: List[str],
):
    seed = 42
    random.seed(seed)
    actual = gender(texts) if to_gender is None else gender(texts,
                                                            to_gender=to_gender)
    assert actual == expected
