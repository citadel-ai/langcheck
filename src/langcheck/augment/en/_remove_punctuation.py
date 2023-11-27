from __future__ import annotations

import random
import string


def remove_punctuation(instances: list[str] | str,
                       *,
                       aug_char_p: float = 1.0,
                       num_perturbations: int = 1) -> list[str]:
    '''Applies a text perturbation to each string in instances (usually a list
    of prompts) where some punctuation is removed.

    Args:
        instances: A single string or a list of strings to be augmented.
        aug_char_p: Percentage of puncutation characters that will be removed.
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
                if char not in string.punctuation:
                    perturbed_instance += char  # No augmentation
                elif random.random() > aug_char_p:
                    perturbed_instance += char  # No augmentation
                else:
                    pass  # Remove character
            perturbed_instances.append(perturbed_instance)

    return perturbed_instances
