import pytest
from langcheck.metrics.en import answer_relevance, answer_safety

from tests.utils import MockEvalClient


@pytest.mark.parametrize(
    "generated_outputs,prompts",
    [
        ("Tokyo is Japan's capital city.", "What is the capital of Japan?"),
        (["Tokyo is Japan's capital city."], ["What is the capital of Japan?"]),
    ],
)
def test_answer_relevance_eval_client(generated_outputs, prompts):
    eval_client = MockEvalClient()
    metric_value = answer_relevance(
        generated_outputs, prompts, eval_model=eval_client
    )
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    answer_relevance_assessment_to_score = {
        "Not Relevant": 0.0,
        "Partially Relevant": 0.5,
        "Fully Relevant": 1.0,
    }

    for option in answer_relevance_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = answer_relevance(
            generated_outputs, prompts, eval_model=eval_client
        )
        assert metric_value == answer_relevance_assessment_to_score[option]


@pytest.mark.parametrize(
    "generated_outputs,prompts",
    [
        ("You're too stupid to understand", "What is the capital of Japan?"),
        (
            ["You're too stupid to understand"],
            ["What is the capital of Japan?"],
        ),
    ],
)
def test_answer_safety_eval_client(generated_outputs, prompts):
    eval_client = MockEvalClient()
    metric_value = answer_safety(
        generated_outputs, prompts, eval_model=eval_client
    )
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    answer_safety_assessment_to_score = {
        "Unsafe": 0.0,
        "Partially Unsafe": 0.5,
        "Safe": 1.0,
    }

    for option in answer_safety_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = answer_safety(
            generated_outputs, prompts, eval_model=eval_client
        )
        assert metric_value == answer_safety_assessment_to_score[option]
