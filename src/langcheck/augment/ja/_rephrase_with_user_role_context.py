from __future__ import annotations

from langcheck.metrics.eval_clients import (
    EvalClient,
)


def rephrase_with_user_role_context(
    instances: list[str] | str,
    user_role: str,
    *,
    num_perturbations: int = 1,
    eval_client: EvalClient,
) -> list[str | None]:
    """Rephrases each prompt in instances (usually a list of prompts) by adding
    the specified user role as context to each prompt. This adds context about
    the role of the user that is making the request.

    For example, if the prompt is "フランスの首都はどこですか?" and the role is
    "学生", the augmented prompt might be "私は学生です。宿題をしています。フランスの首都
    はどこですか？".

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
        language="ja", metric_name="rephrase_with_user_role_context"
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
