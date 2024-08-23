from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from langcheck.metrics._pairwise_text_quality_utils import (
    enforce_pairwise_comparison_consistency,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import MetricInputs
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

    Returns:
        A MetricValue object
    """
    if language not in ["en", "ja", "de"]:
        raise ValueError(f"Unsupported language: {language}")

    metric_inputs = MetricInputs(
        single_inputs={
            "generated_outputs": generated_outputs,
            "prompts": prompts,
            "sources": sources,
            "reference_outputs": reference_outputs,
        },
        optional_params=[
            "generated_outputs",
            "prompts",
            "sources",
            "reference_outputs",
        ],
        input_record_mapping={
            "generated_outputs": "gen_output",
            "prompts": "user_query",
            "sources": "src",
            "reference_outputs": "ref_output",
        },
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

    input_records = metric_inputs.get_input_records()
    populated_prompts = [
        prompt_template.render(input_record) for input_record in input_records
    ]

    scores, explanations = eval_model.get_score(
        metric_name=metric_name,
        language=language,
        prompts=populated_prompts,
        score_map=score_map,
    )

    return MetricValue(
        metric_name=metric_name,
        metric_inputs=metric_inputs,
        explanations=explanations,
        metric_values=scores,
        language=language,
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

    metric_inputs = MetricInputs(
        single_inputs={
            "prompts": prompts,
            "reference_outputs": reference_outputs,
        },
        pairwise_inputs={
            "generated_outputs": (generated_outputs_a, generated_outputs_b),
            "sources": (sources_a, sources_b),
        },
        optional_params=[
            "generated_outputs",
            "prompts",
            "sources",
            "reference_outputs",
        ],
        input_record_mapping={
            "generated_outputs": "gen_output",
            "prompts": "user_query",
            "sources": "src",
            "reference_outputs": "ref_output",
        },
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

    input_records = metric_inputs.get_input_records()
    populated_prompts = [
        prompt_template.render(input_record) for input_record in input_records
    ]

    scores, explanations = eval_model.get_score(
        metric_name=metric_name,
        language=language,
        prompts=populated_prompts,
        score_map=score_map,
    )

    if enforce_consistency:
        # Swap the generated outputs and sources and enforce consistency
        swapped_records = metric_inputs.get_input_records(swap_pairwise=True)
        swapped_prompts = [
            prompt_template.render(swapped_record)
            for swapped_record in swapped_records
        ]

        intermediate_tqdm = (
            "[Swapped model outputs order] Intermediate assessments (1/2)"
        )
        score_tqdm = "[Swapped model outputs order] Calculating scores (2/2)"
        swapped_scores, swapped_explanations = eval_model.get_score(
            metric_name=metric_name,
            language=language,
            prompts=swapped_prompts,
            score_map=score_map,
            intermediate_tqdm_description=intermediate_tqdm,
            score_tqdm_description=score_tqdm,
        )

        # NOTE: The enforce_pairwise_comparison_consistency function assumes
        # that the score_map is symmetric, in the sense that swapping Model A
        # and Model B should result in inverse scores. Most score maps should
        # naturally satisfy this property, but an example of a score map that
        # does *not* satisfy this property is:
        # {
        #   'Response A is much better': 0,
        #   'Response A is slightly better': 1,
        #   'Response B is slightly better': 2
        # }
        #
        # In this case, swapping Model A and Model B will not result in
        # inverse results. The score map should be modified to be symmetric by
        # adding the 'Response B is much better' option:
        # {
        #   'Response A is much better': 0,
        #   'Response A is slightly better': 1,
        #   'Response B is slightly better': 2,
        #   'Response B is much better': 3
        # }
        scores, explanations = enforce_pairwise_comparison_consistency(
            scores,
            explanations,
            swapped_scores,
            swapped_explanations,
            score_map,
        )

    return MetricValue(
        metric_name=metric_name,
        metric_inputs=metric_inputs,
        explanations=explanations,
        metric_values=scores,
        language=language,
    )
