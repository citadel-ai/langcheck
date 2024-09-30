from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from langcheck.metrics._pairwise_text_quality_utils import (
    compute_pairwise_comparison_metric_values_with_consistency,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import (
    IndividualInputType,
    get_metric_inputs,
)
from langcheck.metrics.metric_value import MetricValue


def custom_evaluator(
    generated_outputs: list[str] | str | None,
    prompts: list[str] | str | None,
    sources: list[str] | str | None,
    reference_outputs: list[str] | str | None,
    eval_model: EvalClient,
    metric_name: str,
    score_map: dict[str, float],
    template_path: str,
    language: str,
    *,
    additional_inputs: dict[str, IndividualInputType] | None = None,
    additional_input_name_to_prompt_var_mapping: dict[str, str] | None = None,
) -> MetricValue[float | None]:
    """Calculates the scores of a custom evaluator. The EvalClient will first
    assess the provided inputs using the prompt template, and then convert those
    assessments into scores using the score map.

    The prompt template should be a Jinja2 file (file extension .j2) that
    specifies the criteria that an LLM (as configured in the Eval Client) should
    follow when evaluating an instance. The template is allowed to have
    placeholders for the following variables (NOTE: not all are required):
    - `gen_output`: The generated output
    - `user_query`: The prompt
    - `src`: The source text
    - `ref_output`: The reference output

    By specifying additional inputs, the prompt template can be more flexible.
    The additional inputs should be passed as a dictionary, where the keys are
    the input names and the values are the corresponding values. The additional
    inputs can be mapped to variable names in the prompt template using the
    `additional_input_name_to_prompt_var_mapping` dictionary.

    The prompt template should also specify the final available assessments for
    the LLM evaluator, e.g. "Good", "Bad", "Neutral", etc. The score map should
    then map each of those available assessments to a numerical score. E.g. if
    the available assessments in the prompt template are "Good", "Bad", and
    "Neutral", the score map should be something like:
    ``score_map = {'Good': 1.0, 'Neutral': 0.5, 'Bad': 0.0}``

    NOTE: We have found that LLM models sometimes behave weirdly when the
    assessments are non-ascii characters (see
    https://github.com/citadel-ai/langcheck/pull/84 as an example). So, we
    recommend making the final assessments ascii characters, even when the rest
    of the prompt template contains non-ascii characters (e.g. Japanese).

    Args:
        generated_outputs: The model generated output(s)
        prompts: The prompts used to generate the output(s)
        sources: The source(s) of the generated output(s)
        reference_outputs: The reference output(s)
        eval_model: The EvalClient instance used for the evaluation
        metric_name: The name of the metric
        score_map: A dictionary mapping the evaluator's assessments to scores
        template_path: The path to the prompt template file. This should be a
            Jinja2 file (file extension .j2).
        language: The language that the evaluator will use ('en', 'ja', or 'de')
        additional_inputs: Additional inputs other than the standard ones.
        additional_input_name_to_prompt_var_mapping: A dictionary that maps the
            additional input names to the variable names in the prompt template.

    Returns:
        A MetricValue object
    """
    if language not in ["en", "ja", "de"]:
        raise ValueError(f"Unsupported language: {language}")

    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        sources=sources,
        reference_outputs=reference_outputs,
        additional_inputs=additional_inputs,
        additional_input_name_to_prompt_var_mapping=additional_input_name_to_prompt_var_mapping,
        required_params=[],
    )

    assert Path(
        template_path
    ).exists(), f"Prompt template file {template_path} does not exist."
    assert template_path.endswith(
        ".j2"
    ), 'The prompt template file must be a Jinja2 template file with the extension ".j2"'

    prompt_template_source = Path(template_path).read_text(encoding="utf-8")
    metric_inputs.validate_template(prompt_template_source)
    prompt_template = Template(prompt_template_source)

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=prompt_template,
        metric_name=metric_name,
        language=language,
        score_map=score_map,
    )


def custom_pairwise_evaluator(
    generated_outputs_a: list[str] | str | None,
    generated_outputs_b: list[str] | str | None,
    prompts: list[str] | str | None,
    sources_a: list[str] | str | None,
    sources_b: list[str] | str | None,
    reference_outputs: list[str] | str | None,
    eval_model: EvalClient,
    metric_name: str,
    score_map: dict[str, float],
    template_path: str,
    language: str,
    enforce_consistency: bool = True,
) -> MetricValue[float | None]:
    """Calculates the scores of a custom pairwise evaluator, where "pairwise"
    means that the Responses and/or Sources of two systems will be compared
    against each other. The EvalClient will first assess the provided inputs
    using the prompt template, and then convert those assessments into scores
    using the score map.

    The prompt template should be a Jinja2 file (file extension .j2) that
    specifies the criteria that an LLM (as configured in the Eval Client) should
    follow when evaluating an instance. The template is allowed to have
    placeholders for the following variables (NOTE: not all are required):
    - `gen_output_a`: Model A's generated output
    - `gen_output_b`: Model B's generated output
    - `user_query`: The prompt
    - `src_a`: The source text for Model A
    - `src_b`: The source text for Model B
    - `ref_output`: The reference output

    The prompt template should also specify the final available assessments for
    the LLM evaluator, e.g. "Response A", "Response B", "Tie", etc. The score
    map should then map each of those available assessments to a numerical
    score. E.g. if the available assessments in the prompt template are
    "Response A", "Response B", and "Tie", the score map should be something
    like:
    ``score_map = {'Response A': 0.0, 'Response B': 1.0, 'Tie': 0.5}``

    NOTE: If `enforce_consistency` is True, please make sure that the score map
    is symmetric, in the sense that swapping Model A and Model B should result
    in inverse scores. See the code below for more details.

    NOTE: We have found that LLM models sometimes behave weirdly when the
    assessments are non-ascii characters (see
    https://github.com/citadel-ai/langcheck/pull/84 as an example). So, we
    recommend making the final assessments ascii characters, even when the rest
    of the prompt template contains non-ascii characters (e.g. Japanese).

    Args:
        generated_outputs_a: Model A's generated output(s)
        generated_outputs_b: Model B's generated output(s)
        prompts: The prompts used to generate the output(s)
        sources_a: The source(s) for Model A's generated output(s)
        sources_b: The source(s) for Model B's generated output(s)
        reference_outputs: The reference output(s)
        eval_model: The EvalClient instance used for the evaluation
        metric_name: The name of the metric
        score_map: A dictionary mapping the evaluator's assessments to scores
        template_path: The path to the prompt template file. This should be a
            Jinja2 file (file extension .j2).
        language: The language that the evaluator will use ('en', 'ja', or 'de')
        enforce_consistency: When this is True, we will only return a score if
            the score is the same when Model A and Model B are swapped. This is
            useful for ensuring that the evaluator's position bias is not
            impacting the scores. Default True.

    Returns:
        A MetricValue object
    """

    if language not in ["en", "ja", "de"]:
        raise ValueError(f"Unsupported language: {language}")

    metric_inputs = get_metric_inputs(
        generated_outputs=(generated_outputs_a, generated_outputs_b),
        prompts=prompts,
        sources=(sources_a, sources_b),
        reference_outputs=reference_outputs,
        required_params=[],
    )

    assert Path(
        template_path
    ).exists(), f"Prompt template file {template_path} does not exist."
    assert template_path.endswith(
        ".j2"
    ), 'The prompt template file must be a Jinja2 template file with the extension ".j2"'

    prompt_template_source = Path(template_path).read_text(encoding="utf-8")
    metric_inputs.validate_template(prompt_template_source)
    prompt_template = Template(prompt_template_source)

    if enforce_consistency:
        return compute_pairwise_comparison_metric_values_with_consistency(
            eval_client=eval_model,
            metric_inputs=metric_inputs,
            template=prompt_template,
            metric_name=metric_name,
            language=language,
            score_map=score_map,
        )
    else:
        return eval_model.compute_metric_values_from_template(
            metric_inputs=metric_inputs,
            template=prompt_template,
            metric_name=metric_name,
            language=language,
            score_map=score_map,
        )
