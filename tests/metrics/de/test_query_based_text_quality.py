import pytest

from langcheck.metrics.de import answer_relevance
from tests.utils import MockEvalClient


@pytest.mark.parametrize(
    "generated_outputs,prompts",
    [
        (
            "Tokio ist die Hauptstadt von Japan.",
            "Was ist die Hauptstadt von Japan?",
        ),
        (
            ["Tokio ist die Hauptstadt von Japan."],
            ["Was ist die Hauptstadt von Japan?"],
        ),
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
