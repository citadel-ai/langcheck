import math
from typing import List

################################################################################
# Utility functions
################################################################################


def is_close(a: List, b: List) -> bool:
    '''Returns True if two lists of numbers are element-wise close.'''
    assert len(a) == len(b)
    return all(math.isclose(x, y) for x, y in zip(a, b))