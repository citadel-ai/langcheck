from __future__ import annotations

from .._common._jailbreak_template import jailbreak_template_common

AVAILABLE_JAILBREAK_TEMPLATES = [
    "basic",
    "chatgpt_dan",
    "chatgpt_good_vs_evil",
    "john",
    "universal_adversarial_suffix",
]


def jailbreak_template(
    instances: list[str] | str,
    templates: list[str] | None = None,
    *,
    num_perturbations: int = 1,
    seed: int | None = None,
) -> list[str]:
    """Applies jailbreak templates to each string in instances.

    Args:
        instances: A single string or a list of strings to be augmented.
        templates: A list templates to apply. If None, some templates are
            randomly selected and used.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances. Should be equal to or less than the number
            of templates.
        seed: The seed for the random number generator. You can fix the seed to
            deterministically select the same templates.

    Returns:
        A list of perturbed instances.
    """
    return jailbreak_template_common(
        instances,
        templates,
        AVAILABLE_JAILBREAK_TEMPLATES,
        "en",
        num_perturbations=num_perturbations,
        seed=seed,
    )
