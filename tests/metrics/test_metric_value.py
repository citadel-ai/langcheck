from typing import Optional

import pytest

from langcheck.metrics import is_float
from langcheck.metrics.metric_value import MetricValue


def test_metric_value():
    metric_value = is_float(['1', '2', '3'])
    assert metric_value.all()
    assert metric_value.any()

    metric_value = is_float(['1', 'a', 'b'])
    assert not metric_value.all()
    assert metric_value.any()

    metric_value = is_float(['a', 'b', 'c'])
    assert not metric_value.all()
    assert not metric_value.any()

    with pytest.raises(ValueError):
        bool(metric_value)


def test_metric_value_comparisons():
    metric_value = is_float(['1', '2', '3'])

    # Test all comparisons: ==, !=, <, >, <=, >=
    assert metric_value == 1
    assert (metric_value == 1).all()
    assert all((metric_value == 1).threshold_results)
    assert (metric_value == 1).pass_rate == 1

    assert not metric_value != 1
    assert not (metric_value != 1).any()
    assert not any((metric_value != 1).threshold_results)
    assert (metric_value != 1).pass_rate == 0

    assert metric_value < 1.5
    assert (metric_value < 1.5).all()
    assert all((metric_value < 1.5).threshold_results)
    assert (metric_value < 1.5).pass_rate == 1

    assert metric_value > 0.5
    assert (metric_value > 0.5).all()
    assert all((metric_value > 0.5).threshold_results)
    assert (metric_value > 0.5).pass_rate == 1

    assert metric_value <= 1
    assert (metric_value <= 1).all()
    assert all((metric_value <= 1).threshold_results)
    assert (metric_value <= 1).pass_rate == 1

    assert metric_value >= 0
    assert (metric_value >= 0).all()
    assert all((metric_value >= 0).threshold_results)
    assert (metric_value >= 0).pass_rate == 1


def test_optional_metric_values():
    score_list = [1.0, None]
    dummy_generated_outputs = ['foo', 'bar']
    metric_value: MetricValue[Optional[float]] = MetricValue(
        metric_name='test',
        prompts=None,
        generated_outputs=dummy_generated_outputs,
        reference_outputs=None,
        sources=None,
        explanations=None,
        metric_values=score_list,
        language='en')

    assert (metric_value > 0).pass_rate == 0.5
    assert (metric_value == 1).pass_rate == 0.5
    assert (metric_value == 0).pass_rate == 0
