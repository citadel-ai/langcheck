from __future__ import annotations

from langcheck.metrics._validation import validate_parameters_query_based
from langcheck.metrics.eval_clients import EvalClient
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
    generated_outputs, prompts = validate_parameters_query_based(
        generated_outputs, prompts
    )

    answer_relevance_template = eval_model.load_prompt_template(
        language="de", metric_name="answer_relevance"
    )

    populated_prompts = [
        answer_relevance_template.render(
            {"gen_output": gen_output, "user_query": prompt}
        )
        for gen_output, prompt in zip(generated_outputs, prompts)
    ]

    answer_relevance_assessment_to_score = {
        "Fully Relevant": 1.0,
        "Partially Relevant": 0.5,
        "Not Relevant": 0.0,
    }

    scores, explanations = eval_model.get_score(
        metric_name="answer relevance",
        language="de",
        prompts=populated_prompts,
        score_map=answer_relevance_assessment_to_score,
    )

    return MetricValue(
        metric_name="answer_relevance",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=None,
        sources=None,
        explanations=explanations,
        metric_values=scores,
        language=LANG,
    )
