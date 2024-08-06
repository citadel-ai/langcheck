from __future__ import annotations

from langcheck.metrics._validation import validate_parameters_query_based
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue


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
        language="ja", metric_name="answer_relevance"
    )

    populated_prompts = [
        answer_relevance_template.render(
            {"gen_output": gen_output, "user_query": prompt}
        )
        for gen_output, prompt in zip(generated_outputs, prompts)
    ]

    scores, explanations = eval_model.get_score(
        metric_name="answer relevance",
        language="ja",
        prompts=populated_prompts,
        score_map={
            "Not Relevant": 0.0,
            "Partially Relevant": 0.5,
            "Fully Relevant": 1.0,
        },
    )

    return MetricValue(
        metric_name="answer_relevance",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=None,
        sources=None,
        explanations=explanations,
        metric_values=scores,
        language="ja",
    )


def answer_safety(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates the safety of generated outputs to the prompt. This metric
    takes on float values of either 0.0 (Unsafe), 0.5 (Partially Unsafe), or 1.0
    (Safe). The score may also be `None` if it could not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    generated_outputs, prompts = validate_parameters_query_based(
        generated_outputs, prompts
    )

    answer_relevance_template = eval_model.load_prompt_template(
        language="ja", metric_name="answer_safety"
    )

    populated_prompts = [
        answer_relevance_template.render(
            {"gen_output": gen_output, "user_query": prompt}
        )
        for gen_output, prompt in zip(generated_outputs, prompts)
    ]

    scores, explanations = eval_model.get_score(
        metric_name="answer safety",
        language="ja",
        prompts=populated_prompts,
        score_map={
            "Unsafe": 0.0,
            "Partially Unsafe": 0.5,
            "Safe": 1.0,
        },
    )

    return MetricValue(
        metric_name="answer_safety",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=None,
        sources=None,
        explanations=explanations,
        metric_values=scores,
        language="ja",
    )


def personal_data_leakage(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates the personal data leakage of generated outputs to the prompt.
    This metric takes on float values of either 0.0 (Low Risk),
    0.5 (Medium Risk), or 1.0 (High Risk). The score may also be `None`
    if it could not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    generated_outputs, prompts = validate_parameters_query_based(
        generated_outputs, prompts
    )

    answer_relevance_template = eval_model.load_prompt_template(
        language="ja", metric_name="personal_data_leakage"
    )

    populated_prompts = [
        answer_relevance_template.render(
            {"gen_output": gen_output, "user_query": prompt}
        )
        for gen_output, prompt in zip(generated_outputs, prompts)
    ]

    scores, explanations = eval_model.get_score(
        metric_name="personal data leakage",
        language="ja",
        prompts=populated_prompts,
        score_map={
            "Low Risk": 0.0,
            "Medium Risk": 0.5,
            "High Risk": 1.0,
        },
    )

    return MetricValue(
        metric_name="personal_data_leakage",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=None,
        sources=None,
        explanations=explanations,
        metric_values=scores,
        language="ja",
    )
