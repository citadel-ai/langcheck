from __future__ import annotations

from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import get_metric_inputs
from langcheck.metrics.metric_value import MetricValue

LANG = "ja"


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

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Not Relevant": 0.0,
            "Partially Relevant": 0.5,
            "Fully Relevant": 1.0,
        },
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
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "prompts"],
    )

    metric_name = "answer_safety"

    answer_relevance_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Unsafe": 0.0,
            "Partially Unsafe": 0.5,
            "Safe": 1.0,
        },
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
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "prompts"],
    )

    metric_name = "personal_data_leakage"

    answer_relevance_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Low Risk": 0.0,
            "Medium Risk": 0.5,
            "High Risk": 1.0,
        },
    )


def hate_speech(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates whether hate speech is included in the generated outputs to
    the prompt.  This metric takes on float values of either 0.0 (Low Risk),
    0.5 (Medium Risk), or 1.0 (High Risk). The score may also be `None`
    if it could not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "prompts"],
    )

    metric_name = "hate_speech"

    answer_relevance_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Low Risk": 0.0,
            "Medium Risk": 0.5,
            "High Risk": 1.0,
        },
    )


def adult_content(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates whether adult content is included in the generated outputs to
    the prompt.  This metric takes on float values of either 0.0 (Low Risk),
    0.5 (Medium Risk), or 1.0 (High Risk). The score may also be `None`
    if it could not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "prompts"],
    )

    metric_name = "adult_content"

    answer_relevance_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Low Risk": 0.0,
            "Medium Risk": 0.5,
            "High Risk": 1.0,
        },
    )


def harmful_activity(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates whether the suggestion or instruction for harmful activity is
    included in the generated outputs to the prompt.  This metric takes on float
    values of either 0.0 (Low Risk), 0.5 (Medium Risk), or 1.0 (High Risk).
    The score may also be `None` if it could not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "prompts"],
    )

    metric_name = "harmful_activity"

    answer_relevance_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Low Risk": 0.0,
            "Medium Risk": 0.5,
            "High Risk": 1.0,
        },
    )


def summarization_quality(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates the summarization quality of the generated outputs. This
    metric takes on float values of either 0.0 (Not Good), 0.5 (Somewhat Good),
    or 1.0 (Good). The score may also be `None` if it could not be computed.

    We currently only support the evaluation based on an EvalClient.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s)
        eval_model: The EvalClient instance used for the evaluation

    Returns:
        A :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "prompts"],
    )
    metric_name = "summarization_quality"

    summarization_quality_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=summarization_quality_template,
        metric_name=metric_name,
        language=LANG,
        score_map={"0": 0.0, "0.5": 0.5, "1": 1.0},
    )


def system_prompt_adherence(
    generated_outputs: list[str] | str,
    prompts: list[str] | str,
    system_prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates how well the system prompt is followed when the LLM generates
    outputs. This metric takes on float values of either 0.0 (Not Adherent),
    0.5 (Partially Adherent), or 1.0 (Adherent). The score may also be `None` if
    it could not be computed.

    We currently only support the evaluation based on an EvalClient.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s)
        system_prompts: The system prompts used to generate the output(s)
        eval_model: The EvalClient instance used for the evaluation

    Returns:
        A :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        additional_inputs={
            "system_prompts": system_prompts,
        },
        additional_input_name_to_prompt_var_mapping={
            "system_prompts": "system_prompt",
        },
        required_params=["generated_outputs", "prompts", "system_prompts"],
    )
    metric_name = "system_prompt_adherence"

    system_prompt_adherence_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=system_prompt_adherence_template,
        metric_name=metric_name,
        language=LANG,
        score_map={"0": 0.0, "0.5": 0.5, "1": 1.0},
    )


def user_frustration(
    prompts: list[str] | str,
    history: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates the user frustration from the interaction history between the
    user and the LLM. This metric takes on float values of either
    0.0 (Not Frustrated), 0.5 (Somewhat Frustrated), or 1.0 (Frustrated).
    The score may also be `None` if it could not be computed.

    We currently only support the evaluation based on an EvalClient.

    Args:
        prompts: The prompts used to generate the output(s)
        history: The interaction history between the user and the LLM
        eval_model: The EvalClient instance used for the evaluation

    Returns:
        A :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs = get_metric_inputs(
        prompts=prompts,
        additional_inputs={
            "history": history,
        },
        additional_input_name_to_prompt_var_mapping={
            "history": "history",
        },
        required_params=["prompts", "history"],
    )
    metric_name = "user_frustration"

    user_frustration_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=user_frustration_template,
        metric_name=metric_name,
        language=LANG,
        score_map={"0": 0.0, "0.5": 0.5, "1": 1.0},
    )
