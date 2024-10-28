from __future__ import annotations

from langcheck.metrics._pairwise_text_quality_utils import (
    compute_pairwise_comparison_metric_values_with_consistency,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import get_metric_inputs
from langcheck.metrics.metric_value import MetricValue


def pairwise_comparison(
    generated_outputs_a: list[str] | str,
    generated_outputs_b: list[str] | str,
    prompts: list[str] | str,
    sources_a: list[str] | str | None = None,
    sources_b: list[str] | str | None = None,
    reference_outputs: list[str] | str | None = None,
    enforce_consistency: bool = True,
    eval_model: EvalClient | None = None,
) -> MetricValue[float | None]:
    """Calculates the pairwise comparison metric. This metric takes on float
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
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=(generated_outputs_a, generated_outputs_b),
        prompts=prompts,
        sources=(sources_a, sources_b),
        reference_outputs=reference_outputs,
        required_params=[],
    )

    assert (
        eval_model is not None
    ), "You must pass an EvalClient instance to the pairwise_comparison function."

    pairwise_comparison_assessment_to_score = {
        "Response B": 1.0,
        "Tie": 0.5,
        "Response A": 0.0,
    }
    metric_name = "pairwise_comparison"
    language = "ja"
    pairwise_comparison_template = eval_model.load_prompt_template(
        language=language, metric_name=metric_name
    )

    if enforce_consistency:
        return compute_pairwise_comparison_metric_values_with_consistency(
            eval_client=eval_model,
            metric_inputs=metric_inputs,
            template=pairwise_comparison_template,
            metric_name=metric_name,
            language=language,
            score_map=pairwise_comparison_assessment_to_score,
        )
    else:
        return eval_model.compute_metric_values_from_template(
            metric_inputs=metric_inputs,
            template=pairwise_comparison_template,
            metric_name=metric_name,
            language=language,
            score_map=pairwise_comparison_assessment_to_score,
        )
