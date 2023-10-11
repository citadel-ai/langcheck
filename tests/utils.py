from __future__ import annotations

import math
from typing import List

################################################################################
# Utility functions
################################################################################


def is_close(a: List, b: List) -> bool:
    '''Returns True if two lists of numbers are element-wise close.'''
    assert len(a) == len(b)
    return all(math.isclose(x, y) for x, y in zip(a, b))


def lists_are_equal(a: List[str] | str, b: List[str] | str) -> bool:
    '''Returns True if two lists of strings are equal. If either argument is a
    single string, it's automatically converted to a list.
    '''
    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]
    return a == b
