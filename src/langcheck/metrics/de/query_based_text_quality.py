from __future__ import annotations

from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import get_metric_inputs
from langcheck.metrics.metric_value import MetricValue

LANG = "de"


def answer_relevance(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates the relevance of generated outputs to the prompt. This metric
    takes on float values of either 0.0 (Not Relevant), 0.5 (Partially
    Relevant), or 1.0 (Fully Relevant). The score may also be `None` if it could
    not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "prompts"],
    )

    metric_name = "answer_relevance"

    answer_relevance_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    answer_relevance_assessment_to_score = {
        "Fully Relevant": 1.0,
        "Partially Relevant": 0.5,
        "Not Relevant": 0.0,
    }

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map=answer_relevance_assessment_to_score,
    )
