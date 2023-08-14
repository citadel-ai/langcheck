import math
from typing import List

import pytest

from langcheck.eval.text_structure import is_float, is_int


def is_close(a: List, b: List) -> bool:
    '''Returns True if two lists of numbers are element-wise close.'''
    assert len(a) == len(b)
    return all(math.isclose(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("generated_outputs,domain,metric_values", [
    (['-100', '-1', '0', '1', '100'], None, [1, 1, 1, 1, 1]),
    (['-100', '-1', '0', '1', '100'], range(-5, 6), [0, 1, 1, 1, 0]),
    (['-100', '-1', '0', '1', '100'], {0, 1, 2}, [0, 0, 1, 1, 0]),
    (['lorem', 'ipsum', '13.14', '-999.999', 'true', 'True', 'false', 'False'
      ], None, [0, 0, 0, 0, 0, 0, 0, 0]),
])
def test_is_int(generated_outputs, domain, metric_values):
    eval_value = is_int(generated_outputs, domain)
    assert eval_value.metric_name == 'is_int'
    assert eval_value.prompts is None
    assert eval_value.generated_outputs == generated_outputs
    assert is_close(eval_value.metric_values, metric_values)


@pytest.mark.parametrize("generated_outputs,min,max,metric_values", [
    (['-100.5', '-1', '0', '1', '100.5'], None, None, [1, 1, 1, 1, 1]),
    (['-100.5', '-1', '0', '1', '100.5'], None, 5, [1, 1, 1, 1, 0]),
    (['-100.5', '-1', '0', '1', '100.5'], -5, None, [0, 1, 1, 1, 1]),
    (['-100.5', '-1', '0', '1', '100.5'], -5, 5, [0, 1, 1, 1, 0]),
    (['lorem', 'ipsum', '13.14', '-999.999', 'true', 'True', 'false', 'False'
      ], None, None, [0, 0, 1, 1, 0, 0, 0, 0]),
])
def test_is_float(generated_outputs, min, max, metric_values):
    eval_value = is_float(generated_outputs, min, max)
    assert eval_value.metric_name == 'is_float'
    assert eval_value.prompts is None
    assert eval_value.generated_outputs == generated_outputs
    assert is_close(eval_value.metric_values, metric_values)
