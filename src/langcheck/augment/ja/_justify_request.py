from __future__ import annotations

from langcheck.metrics.eval_clients import (
    EvalClient,
)


def justify_request(
    instances: list[str] | str,
    user_role: str,
    *,
    num_perturbations: int = 1,
    eval_client: EvalClient,
) -> list[str | None]:
    """Rephrases each prompt in instances (usually a list of prompts) with the
    specified role and the justification for the prompt. For example, if the prompt
    is "フランスの首都はどこですか?", the role could be "学生". In that case,
    a possible augmented prompt would be "私は学生です。宿題をしています。
    フランスの首都はどこですか?".

    Args:
        instances: A single prompt or a list of prompts to be augmented.
        user_role: The role of the user in the prompt.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances
        eval_client: The type of model to use.

    Returns:
        A list of rephrased instances.
    """

    prompt_template = eval_client.load_prompt_template(
        language="ja", metric_name="justify_request"
    )

    instances = [instances] if isinstance(instances, str) else instances
    prompt_template_inputs = [
        {"instance": instance, "user_role": user_role} for instance in instances
    ]

    return eval_client.repeat_requests_from_template(
        prompt_template_inputs=prompt_template_inputs,
        template=prompt_template,
        num_perturbations=num_perturbations,
    )
