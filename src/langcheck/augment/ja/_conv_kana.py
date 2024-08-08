from __future__ import annotations

import random

import jaconv


def conv_hiragana(
    instances: list[str] | str,
    convert_to: str = "kata",
    *,
    aug_char_p: float = 1.0,
    num_perturbations: int = 1,
    seed: int | None = None,
) -> list[str]:
    """Convert hiragana in the text to katakana or vice versa.

    Args:
        instances: A single string or a list of strings to be augmented.
        convert_to: The target script to convert to. Available values are
            - 'kata' for katakana
            - 'hkata' for half-width katakana
            - 'alpha' for alphabets
        aug_char_p: Percentage of all characters that will be augmented.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances.
        seed: The seed for the random number generator. You can fix the seed to
            deterministically choose which characters to change.

    Returns:
        A list of perturbed instances.
    """

    # Validation on aug_char_p
    if aug_char_p < 0 or aug_char_p > 1:
        raise ValueError("aug_char_p must be between 0 and 1")

    if seed is not None:
        random.seed(seed)

    instances = [instances] if isinstance(instances, str) else instances
    perturbed_instances = []
    for instance in instances:
        for _ in range(num_perturbations):
            perturbed_instance = ""
            for char in instance:
                if random.random() > aug_char_p:
                    perturbed_instance += char  # No augmentation
                else:
                    if convert_to == "kata":
                        perturbed_instance += jaconv.hira2kata(char)
                    elif convert_to == "hkata":
                        perturbed_instance += jaconv.hira2hkata(char)
                    elif convert_to == "alpha":
                        perturbed_instance += jaconv.kana2alphabet(char)
                    else:
                        raise ValueError(
                            "convert_to must be one of 'kata', 'hkata', or 'alpha'"
                        )
            perturbed_instances.append(perturbed_instance)

    return perturbed_instances
