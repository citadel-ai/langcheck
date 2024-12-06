from __future__ import annotations

from langcheck.metrics.eval_clients import (
    EvalClient,
)


def roleplay(
    instances: list[str] | str,
    system_role: str,
    *,
    num_perturbations: int = 1,
    eval_client: EvalClient,
) -> list[str | None]:
    """Rephrases each prompt in instances (usually a list of prompts) so that
    the prompt tells the system to act in the specified role.  For example, if
    the prompt is "フランスの首都はどこですか?", the role could be "先生". In that
    case, a possible augmented prompt would be "あなたは先生です。学生に地理を教えています。
    では以下のクエリに応えてください: フランスの首都はどこですか?".

    Args:
        instances: A single prompt or a list of prompts to be augmented.
        system_role: The role of the system in the augmented prompt.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances
        eval_client: The type of model to use.

    Returns:
        A list of rephrased instances.
    """

    prompt_template = eval_client.load_prompt_template(
        language="ja", metric_name="roleplay"
    )

    instances = [instances] if isinstance(instances, str) else instances
    prompt_template_inputs = [
        {"instance": instance, "system_role": system_role}
        for instance in instances
    ]

    return eval_client.repeat_requests_from_template(
        prompt_template_inputs=prompt_template_inputs,
        template=prompt_template,
        num_perturbations=num_perturbations,
    )
