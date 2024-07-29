from __future__ import annotations

from .._common._jailbreak_template import jailbreak_template_common

AVAILABLE_JAILBREAK_TEMPLATES = [
    # langcheck/metrics/prompts/ja/jailbreak_templates/basic.j2
    # Basic "Ignore the instruction" prompt
    "basic",
    # langcheck/metrics/prompts/ja/jailbreak_templates/chatgpt_good_vs_evil.j2
    # Prompt that asks ChatGPT to generate both good and evil outputs
    "chatgpt_good_vs_evil",
    # langcheck/metrics/prompts/ja/jailbreak_templates/john.j2
    # Prompt that asks the LLM to act as a virtual assistant "John"
    "john",
]


def jailbreak_template(
    instances: list[str] | str,
    templates: list[str] | None = None,
    *,
    num_perturbations: int = 1,
    randomize_order: bool = True,
    seed: int | None = None,
) -> list[str]:
    """Applies jailbreak templates to each string in instances.

    Args:
        instances: A single string or a list of strings to be augmented.
        templates: A list templates to apply. If None, some templates are
            randomly selected and used. Available templates are:
            - basic
            - chatgpt_good_vs_evil
            - john
        num_perturbations: The number of perturbed instances to generate for
            each string in instances. Should be equal to or less than the number
            of templates.
        randomize_order: If True, the order of the templates is randomized.
            When turned off, num_perturbations needs to be equal to the number
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
        "ja",
        num_perturbations=num_perturbations,
        randomize_order=randomize_order,
        seed=seed,
    )
