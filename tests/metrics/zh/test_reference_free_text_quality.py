import pytest

from langcheck.metrics.zh import (
    sentiment,
    toxicity,
    xuyaochen_report_readability,
)
from tests.utils import MockEvalClient, is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    "generated_outputs",
    [
        "我今天很开心",
        ["我今天很开心"],
        ["我今天很开心", "我今天很不开心"],
    ],
)
def test_sentiment(generated_outputs):
    metric_value = sentiment(generated_outputs)
    assert 0 <= metric_value <= 1
    if len(metric_value.metric_values) == 2:
        sentiment_score_low = metric_value.metric_values[1]
        assert sentiment_score_low is not None
        assert 0.0 <= sentiment_score_low <= 0.5


@pytest.mark.parametrize(
    "generated_outputs", ["我今天很开心", ["我今天很开心"]]
)
def test_sentiment_eval_client(generated_outputs):
    eval_client = MockEvalClient()
    metric_value = sentiment(generated_outputs, eval_model=eval_client)
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None
    sentiment_assessment_to_score = {
        "Positive": 1.0,
        "Neutral": 0.5,
        "Negative": 0.0,
    }
    for option in sentiment_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = sentiment(generated_outputs, eval_model=eval_client)
        assert metric_value == sentiment_assessment_to_score[option]


@pytest.mark.parametrize(
    "generated_outputs",
    ["我今天生病了。", ["我今天生病了。"], ["我今天生病了。", "你有病啊。"]],
)
def test_toxicity(generated_outputs):
    metric_value = toxicity(generated_outputs)
    toxicity_score_low_risk = metric_value.metric_values[0]
    assert 0 <= toxicity_score_low_risk <= 0.6  # type: ignore[reportGeneralTypeIssues]
    if len(metric_value.metric_values) == 2:
        toxicity_score_high_risk = metric_value.metric_values[1]
        assert toxicity_score_high_risk is not None
        assert 0.5 <= toxicity_score_high_risk <= 1


@pytest.mark.parametrize(
    "generated_outputs", ["我今天生病了。", ["我今天生病了。"]]
)
def test_toxicity_eval_client(generated_outputs):
    eval_client = MockEvalClient()
    metric_value = toxicity(generated_outputs, eval_model=eval_client)
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    toxicity_assessment_to_score = {
        "Toxic": 1.0,
        "Nontoxic": 0,
    }
    for option in toxicity_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = toxicity(generated_outputs, eval_model=eval_client)
        assert metric_value == toxicity_assessment_to_score[option]


@pytest.mark.parametrize(
    "generated_outputs,metric_values",
    [
        ("这一句话很长很难懂，你最好把他改一下。", [11.0]),
        (["今天天气很好。一起去散步吧！"], [2.5]),
    ],
)
def test_xuyaochen_report_readability(generated_outputs, metric_values):
    metric_value = xuyaochen_report_readability(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)
