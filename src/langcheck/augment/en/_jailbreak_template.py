from __future__ import annotations

from .._common._jailbreak_template import _jailbreak_template

AVAILABLE_JAILBREAK_TEMPLATES = [
    "basic",
    "dan",
    "good_vs_evil",
    "john",
    "universal_adversarial_suffix",
]


def jailbreak_template(
    instances: list[str] | str,
    templates: list[str] | None = None,
    *,
    num_perturbations: int = 1,
) -> list[str]:
    """Applies jailbreak templates to each string in instances.

    Args:
        instances: A single string or a list of strings to be augmented.
        templates: A list of applied templates. If None, some templates are
            randomly selected and used.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances. Should be equal to or less than the number
            of templates.

    Returns:
        A list of perturbed instances.
    """
    return _jailbreak_template(
        instances,
        templates,
        AVAILABLE_JAILBREAK_TEMPLATES,
        "en",
        num_perturbations=num_perturbations,
    )
