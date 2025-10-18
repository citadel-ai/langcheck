from __future__ import annotations

import math
from collections.abc import Iterable

from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.eval_clients.extractor import Extractor
from src.langcheck.metrics.eval_clients.eval_response import (
    ResponsesWithTokenUsage,
)


################################################################################
# Utility Classes
################################################################################
class MockEvalClient(EvalClient):
    """A mock evaluation client for testing purposes."""

    def __init__(self, evaluation_result: str | None = None) -> None:
        """You can set the constant evaluation result for the mock client.
        the score is returned based on that value and the score_map passed from
        each metric function.
        """
        self.evaluation_result = evaluation_result
        self._extractor = MockExtractor()

    def get_text_responses(
        self, prompts: Iterable[str], *, tqdm_description: str | None = None
    ) -> ResponsesWithTokenUsage[str]:
        return ResponsesWithTokenUsage(
            [self.evaluation_result] * len(list(prompts)),
            None,
        )


class MockExtractor(Extractor):
    def get_float_score(
        self,
        metric_name: str,
        language: str,
        unstructured_assessment_result: list[str | None],
        score_map: dict[str, float],
        *,
        tqdm_description: str | None = None,
    ) -> list[float | None]:
        eval_results = []
        # Assume that the evaluation result is actually structured and it can be
        # put into the score_map directly
        for assessment in unstructured_assessment_result:
            if assessment is None or assessment not in score_map:
                eval_results.append(None)
            else:
                eval_results.append(score_map[assessment])

        return eval_results


################################################################################
# Utility functions
################################################################################


def is_close(a: list, b: list) -> bool:
    """Returns True if two lists of numbers are element-wise close."""
    assert len(a) == len(b)
    return all(math.isclose(x, y) for x, y in zip(a, b))


def lists_are_equal(a: list[str] | str, b: list[str] | str) -> bool:
    """Returns True if two lists of strings are equal. If either argument is a
    single string, it's automatically converted to a list.
    """
    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]
    return a == b
