from __future__ import annotations

from typing import List, Optional

from langcheck.metrics._pairwise_text_quality_utils import (
    enforce_pairwise_comparison_consistency,
    generate_pairwise_comparison_prompt_params)
from langcheck.metrics._validation import \
    validate_parameters_pairwise_comparison
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue

from ..prompts._utils import get_template


def pairwise_comparison(
        generated_outputs_a: List[str] | str,
        generated_outputs_b: List[str] | str,
        prompts: List[str] | str,
        sources_a: Optional[List[str] | str] = None,
        sources_b: Optional[List[str] | str] = None,
        reference_outputs: Optional[List[str] | str] = None,
        enforce_consistency: bool = True,
        eval_model: EvalClient | None = None) -> MetricValue[Optional[float]]:
    '''Calculates the pairwise comparison metric. This metric takes on float
    values of either 0.0 (Response A is better), 0.5 (Tie), or 1.0 (Response B
    is better). The score may also be `None` if it could not be computed.

    We currently only support the evaluation based on an EvalClient.

    Args:
        generated_outputs_a: Model A's generated output(s) to evaluate
        generated_outputs_b: Model B's generated output(s) to evaluate
        prompts: The prompts used to generate the output(s)
        sources_a: The source text(s) for Model A's generated output(s), default
            None
        sources_b: The source text(s) for Model B's generated output(s), default
            None
        reference_outputs: The reference output(s), default None
        enforce_consistency: When this is True, we will only return a score if
            the score is the same when Model A and Model B are swapped. This is
            useful for ensuring that the evaluator's position bias is not
            impacting the scores. Default True.
        eval_model: The EvalClient instance used for the evaluation. This is
            marked as Optional so that it can follow the above arguments that
            have default values (for consistency with the other metrics), but
            this is in fact a required argument.

    Returns:
        An MetricValue object
    '''
    generated_outputs_a, generated_outputs_b, prompts, sources_a, sources_b, reference_outputs = validate_parameters_pairwise_comparison(  # NOQA: E501
        generated_outputs_a, generated_outputs_b, prompts, sources_a, sources_b,
        reference_outputs)

    assert eval_model is not None, 'You must pass an EvalClient instance to the pairwise_comparison function.'  # NOQA: E501

    pairwise_comparison_assessment_to_score = {
        'Response B': 1.0,
        'Tie': 0.5,
        'Response A': 0.0
    }

    pairwise_comparison_template = get_template(
        'ja/metrics/pairwise_comparison.j2')

    prompt_params = generate_pairwise_comparison_prompt_params(
        generated_outputs_a, generated_outputs_b, prompts, sources_a, sources_b,
        reference_outputs)

    populated_prompts = [
        pairwise_comparison_template.render(prompt_param)
        for prompt_param in prompt_params
    ]

    scores, explanations = eval_model.get_score(
        metric_name='comparison of two responses',
        language='ja',
        prompts=populated_prompts,
        score_map=pairwise_comparison_assessment_to_score)

    if enforce_consistency:
        # Swap the generated outputs and enforce consistency
        swapped_prompt_params = generate_pairwise_comparison_prompt_params(
            generated_outputs_b, generated_outputs_a, prompts, sources_b,
            sources_a, reference_outputs)

        populated_swapped_prompts = [
            pairwise_comparison_template.render(prompt_param)
            for prompt_param in swapped_prompt_params
        ]

        intermediate_tqdm = '[Swapped model outputs order] Intermediate assessments (1/2)'  # NOQA: E501
        score_tqdm = '[Swapped model outputs order] Calculating scores (2/2)'
        swapped_scores, swapped_explanations = eval_model.get_score(
            metric_name='comparison of two responses',
            language='ja',
            prompts=populated_swapped_prompts,
            score_map=pairwise_comparison_assessment_to_score,
            intermediate_tqdm_description=intermediate_tqdm,
            score_tqdm_description=score_tqdm)

        scores, explanations = enforce_pairwise_comparison_consistency(
            scores, explanations, swapped_scores, swapped_explanations)

    return MetricValue(metric_name='pairwise_comparison',
                       prompts=prompts,
                       generated_outputs=(generated_outputs_a,
                                          generated_outputs_b),
                       reference_outputs=reference_outputs,
                       sources=(sources_a, sources_b),
                       explanations=explanations,
                       metric_values=scores,
                       language='ja')
