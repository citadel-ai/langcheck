from __future__ import annotations

from jinja2 import Template

from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import MetricInputs
from langcheck.metrics.metric_value import MetricValue


def compute_metric_values_from_template(
    metric_inputs: MetricInputs,
    template: Template,
    metric_name: str,
    language: str,
    score_map: dict[str, float],
    eval_client: EvalClient,
    *,
    score_eval_client: EvalClient | None = None,
) -> MetricValue[float | None]:
    """Compute the metric values from the given Jinja template with the
    metric inputs. This function assumes that the template parameters are
    already validated and the template is ready to be rendered.

    Args:
        metric_inputs: The metric inputs that contain the prompts,
            generated outputs, reference outputs... etc.
        template: The Jinja template that is ready to be rendered.
        enforce_pairwise_consistency: Whether to enforce pairwise
            consistency when computing the metric values.
        metric_name: The name of the metric to be used. (e.g. "toxicity")
        language: The language of the prompts. (e.g. "en")
        score_map: The mapping from the short assessment results (e.g. "Good")
            to the float scores.
        eval_client: The eval client used to compute the metric values.
            It is used to get the unstructured assessment results, and also to
            get the scores in case `score_eval_client` is not provided.
        score_eval_client (Optional): The eval client used to compute the scores.
            If not provided, the scores will be computed using the `eval_client`.
    Returns:
        MetricValue: The metric values computed from the template.
    """
    prompt_template_inputs = metric_inputs.get_inputs_for_prompt_template()
    populated_prompts = [
        template.render(prompt_template_input)
        for prompt_template_input in prompt_template_inputs
    ]

    if score_eval_client:
        explanations = eval_client.get_text_responses(
            prompts=populated_prompts,
        )
        scores = score_eval_client.get_float_score(
            metric_name,
            language,
            explanations,
            score_map,
        )
    else:
        scores, explanations = eval_client.get_score(
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
