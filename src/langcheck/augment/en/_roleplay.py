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
    the prompt is "What is the capital of France?", the role could be "teacher".
    In that, a possible augmented prompt would be "You are a teacher and you
    need to teach history to your students.  Now answer the query: What is the
    capital of France?".

    Args:
        instances: A single prompt or a list of prompts to be augmented.
        system_role: The of the system in the augmented prompt.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances
        eval_model: The type of model to use.

    Returns:
        A list of rephrased instances.
    """

    prompt_template = eval_client.load_prompt_template(
        language="en", metric_name="roleplay"
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
