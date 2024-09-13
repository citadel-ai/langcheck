from __future__ import annotations

from jinja2 import Template

from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import MetricInputs
from langcheck.metrics.metric_value import MetricValue


def enforce_pairwise_comparison_consistency(
    original_scores: list[float | None],
    original_explanations: list[str | None],
    swapped_scores: list[float | None],
    swapped_explanations: list[str | None],
    score_map: dict[str, float],
) -> tuple[list[float | None], list[str | None]]:
    """Enforce consistency in pairwise comparison scores.

    Args:
        original_scores: The scores for the original order of the models
        original_explanations: The explanations for the original order of the
            models
        swapped_scores: The scores for the swapped order of the models
        swapped_explanations: The explanations for the swapped order of the
            models
        score_map: The mapping from the assessment results to the scores (e.g.
            {'Response A': 0.0, 'Response B': 1.0, 'Tie': 0.5})
    """
    # Get the sorted list of available scores
    sorted_available_scores = list(score_map.values())
    sorted_available_scores.sort()

    # Iterate through the scores and explanations to check for consistency.
    # If a score is not consistent, set it to None, and merge the two
    # explanations to show the inconsistency.
    scores = original_scores.copy()
    explanations = original_explanations.copy()
    for i in range(len(scores)):
        if scores[i] is None or swapped_scores[i] is None:
            # If either score is None, we cannot determine consistency, so
            # we set the score and explanation to None
            scores[i] = None
            explanations[i] = None
            continue

        # A score is consistent if the score's index in the
        # sorted_available_scores list is the inverse of the swapped score's
        # index. For example, if the score_map is
        # {'Response A': 0.0, 'Response B': 1.0, 'Tie': 0.5}, and the score is
        # 0.0, the swapped score should be 1.0.
        if (
            sorted_available_scores.index(scores[i])  # type: ignore
            + sorted_available_scores.index(swapped_scores[i])  # type: ignore
            != len(sorted_available_scores) - 1
        ):
            scores[i] = None
            explanations[i] = (
                f"Original assessment: {explanations[i]}\nSwapped assessment: {swapped_explanations[i]}"
            )
    return scores, explanations


def compute_pairwise_comparison_metric_values_with_consistency(
    eval_client: EvalClient,
    metric_inputs: MetricInputs,
    template: Template,
    metric_name: str,
    language: str,
    score_map: dict[str, float],
) -> MetricValue[float | None]:
    """Utility function to compute the pairwise metric values from the given
    Jinja template with the metric inputs. This function always enforces the
    consistency between in the pairwise comparison scores.  This function
    assumes that the template parameters are already validated and the template is ready to be rendered.

    Args:
        eval_client: The EvalClient instance that is used to compute the scores.
        metric_inputs: The metric inputs that contain the prompts,
            generated outputs, reference outputs... etc.
        template: The Jinja template that is ready to be rendered.
        enforce_pairwise_consistency: Whether to enforce pairwise
            consistency when computing the metric values.
        metric_name: The name of the metric to be used. (e.g. "toxicity")
        language: The language of the prompts. (e.g. "en")
        score_map: The mapping from the short assessment results
            (e.g. "Good") to the scores.

    Returns:
        MetricValue: The metric values computed from the template.
    """
    prompt_template_inputs = metric_inputs.get_inputs_for_prompt_template()
    populated_prompts = [
        template.render(prompt_template_input)
        for prompt_template_input in prompt_template_inputs
    ]

    scores, explanations = eval_client.get_score(
        metric_name=metric_name,
        language=language,
        prompts=populated_prompts,
        score_map=score_map,
    )

    # Swap the generated outputs and sources to enforce consistency
    swapped_prompt_template_inputs = (
        metric_inputs.get_inputs_for_prompt_template(swap_pairwise=True)
    )
    swapped_prompts = [
        template.render(swapped_prompt_template_input)
        for swapped_prompt_template_input in swapped_prompt_template_inputs
    ]

    intermediate_tqdm = (
        "[Swapped model outputs order] Intermediate assessments (1/2)"
    )
    score_tqdm = "[Swapped model outputs order] Calculating scores (2/2)"
    swapped_scores, swapped_explanations = eval_client.get_score(
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
