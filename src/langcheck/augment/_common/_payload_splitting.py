from __future__ import annotations

import random

from langcheck.metrics.prompts._utils import get_template


def payload_splitting_common(
    instances: list[str] | str,
    language: str,
    *,
    num_perturbations: int = 1,
    seed: int | None = None,
) -> list[str]:
    """Applies payload splitting augmentation to each string in instances.

    Ref: https://arxiv.org/pdf/2302.05733

    Args:
        instances: A single string or a list of strings to be augmented.
        language: The language of the templates.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances. Should be equal to or less than the number
            of templates.
        seed: The seed for the random number generator. You can fix the seed to
            deterministically choose the indices to split the instances.

    Returns:
        A list of perturbed instances.
    """

    if seed is not None:
        random.seed(seed)

    instances = [instances] if isinstance(instances, str) else instances

    perturbed_instances = []
    for instance in instances:
        # smartgpt.j2 is the only template available for payload splitting
        template = get_template(f"{language}/payload_splitting/smartgpt.j2")
        for _ in range(num_perturbations):
            # Split the instance into three parts
            len_instance = len(instance)
            part_a_idx = random.randint(0, len_instance // 2)
            part_b_idx = random.randint(len_instance // 2, len_instance)

            part_a, part_b, part_c = (
                instance[:part_a_idx],
                instance[part_a_idx:part_b_idx],
                instance[part_b_idx:],
            )
            perturbed_instances.append(
                template.render(
                    {"part_a": part_a, "part_b": part_b, "part_c": part_c}
                ).strip()
            )
    return perturbed_instances
