from __future__ import annotations

import random


def change_case(
    instances: list[str] | str,
    *,
    to_case: str = 'uppercase',
    aug_char_p: float = 1.0,
    num_perturbations: int = 1,
) -> list[str]:
    '''Applies a text perturbation to each string in instances (usually a list
    of prompts) where some characters are changed to uppercase or lowercase.

    Args:
        instances: A single string or a list of strings to be augmented.
        to_case: Either 'uppercase' or 'lowercase'.
        aug_char_p: Percentage of all characters that will be augmented.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances.

    Returns:
        A list of perturbed instances.
    '''

    instances = [instances] if isinstance(instances, str) else instances
    perturbed_instances = []
    for instance in instances:
        for _ in range(num_perturbations):
            perturbed_instance = ''
            for char in instance:
                if random.random() > aug_char_p:
                    perturbed_instance += char  # No augmentation
                else:
                    if to_case == 'uppercase':
                        perturbed_instance += char.upper()
                    else:
                        perturbed_instance += char.lower()
            perturbed_instances.append(perturbed_instance)

    return perturbed_instances
