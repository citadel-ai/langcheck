from __future__ import annotations

from langcheck.metrics._validation import _validate_parameters
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.prompts._utils import get_custom_template


def custom_evaluator(generated_outputs: list[str] | str,
                     prompts: list[str] | str | None,
                     sources: list[str] | str | None,
                     reference_outputs: list[str] | str | None,
                     eval_model: EvalClient, metric_name: str,
                     score_map: dict[str, float], template_path: str,
                     language: str) -> MetricValue[float | None]:
    generated_outputs, prompts, reference_outputs, sources = _validate_parameters(  # NOQA: E501
        generated_outputs, prompts, reference_outputs, sources)

    prompt_template = get_custom_template(template_path)

    def _args_to_prompt_param(generated_outputs, prompts, sources,
                              reference_outputs, index):
        prompt_param = {
            'gen_output': generated_outputs[index],
        }
        if prompts is not None:
            prompt_param['user_query'] = prompts[index]
        if sources is not None:
            prompt_param['src'] = sources[index]
        if reference_outputs is not None:
            prompt_param['ref_output'] = reference_outputs[index]
        return prompt_param

    populated_prompts = []
    for i in range(len(generated_outputs)):
        prompt_param = _args_to_prompt_param(generated_outputs, prompts,
                                             sources, reference_outputs, i)
        populated_prompts.append(prompt_template.render(prompt_param))

    scores, explanations = eval_model.get_score(
        metric_name=metric_name,
        language=language,
        prompts=populated_prompts,
        score_map=score_map,
    )

    return MetricValue(metric_name=metric_name,
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=reference_outputs,
                       sources=sources,
                       explanations=explanations,
                       metric_values=scores,
                       language=language)
