from __future__ import annotations

from langcheck.metrics.eval_clients import (
    EvalClient,
)


def rephrase(
    instances: list[str] | str,
    *,
    num_perturbations: int = 1,
    eval_client: EvalClient,
    eval_prompt_version: str = "v2",
) -> list[str | None]:
    """Rephrases each string in instances (usually a list of prompts) without
    changing their meaning. We use a modified version of the prompt presented
    in `"Rethinking Benchmark and Contamination for Language Models with
    Rephrased Samples" <https://arxiv.org/abs/2311.04850>`__ to make an LLM
    rephrase the given text.

    Args:
        instances: A single string or a list of strings to be augmented.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances
        eval_model: The type of model to use.
        eval_prompt_version: The version of the eval prompt to use when the
            EvalClient is used. The default version is 'v2' (latest).

    Returns:
        A list of rephrased instances.
    """

    if eval_prompt_version not in ["v1", "v2"]:
        raise ValueError(
            f"Invalid eval_prompt_version: {eval_prompt_version}. The valid versions are 'v1' and 'v2'."
        )

    prompt_template = eval_client.load_prompt_template(
        language="en",
        metric_name="rephrase",
        eval_prompt_version=eval_prompt_version,
    )

    instances = [instances] if isinstance(instances, str) else instances
    prompt_template_inputs = [{"instance": instance} for instance in instances]

    return eval_client.repeat_requests_from_template(
        prompt_template_inputs=prompt_template_inputs,
        template=prompt_template,
        num_perturbations=num_perturbations,
    )
