import pytest

from langcheck.metrics.ja import context_relevance, factual_consistency
from tests.utils import MockEvalClient

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    "generated_outputs,sources",
    [
        ("東京は日本の首都です。", "東京は日本の首都です。"),
        (
            ["東京は日本の首都です。", "地球は平面です。"],
            ["東京は日本の首都です。", "地球は球体です。"],
        ),
    ],
)
def test_factual_consistency(generated_outputs, sources):
    metric_value = factual_consistency(generated_outputs, sources)
    factual_consistency_high = metric_value.metric_values[0]
    assert factual_consistency_high is not None
    assert 0.9 <= factual_consistency_high <= 1
    if len(metric_value.metric_values) == 2:
        factual_consistency_low = metric_value.metric_values[1]
        assert factual_consistency_low is not None
        assert 0.0 <= factual_consistency_low <= 0.1


@pytest.mark.parametrize(
    "generated_outputs,sources",
    [
        ("東京は日本の首都です。", "東京は日本の首都です。"),
        (["東京は日本の首都です。"], ["東京は日本の首都です。"]),
    ],
)
def test_factual_consistency_eval_client(generated_outputs, sources):
    eval_client = MockEvalClient()
    metric_value = factual_consistency(
        generated_outputs, sources, eval_model=eval_client
    )
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    factual_consistency_assessment_to_score = {
        "Fully Consistent": 1.0,
        "Partially Consistent": 0.5,
        "Not Consistent": 0.0,
    }

    for option in factual_consistency_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = factual_consistency(
            generated_outputs, sources, eval_model=eval_client
        )
        assert metric_value == factual_consistency_assessment_to_score[option]


@pytest.mark.parametrize(
    "prompts,sources",
    [
        ("日本の首都は何ですか？", "東京は日本の首都です。"),
        (["日本の首都は何ですか？"], ["東京は日本の首都です。"]),
    ],
)
def test_context_relevance_eval_client(prompts, sources):
    eval_client = MockEvalClient()
    metric_value = context_relevance(sources, prompts, eval_model=eval_client)
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    context_relevance_assessment_to_score = {
        "Fully Relevant": 1.0,
        "Partially Relevant": 0.5,
        "Not Relevant": 0.0,
    }

    for option in context_relevance_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = context_relevance(
            sources, prompts, eval_model=eval_client
        )
        assert metric_value == context_relevance_assessment_to_score[option]
