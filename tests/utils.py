from __future__ import annotations

import math
from typing import Iterable, List

from langcheck.metrics.eval_clients import EvalClient


################################################################################
# Utility Classes
################################################################################
class MockEvalClient(EvalClient):
    '''A mock evaluation client for testing purposes.'''

    def _unstructured_assessment(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        return ['Okay'] * len(list(prompts))

    def _get_float_score(
            self,
            metric_name: str,
            language: str,
            unstructured_assessment_result: list[str | None],
            score_map: dict[str, float],
            *,
            tqdm_description: str | None = None) -> list[float | None]:
        return [0.5] * len(unstructured_assessment_result)


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
