from __future__ import annotations

import random

from langcheck.metrics.prompts._utils import get_template


def jailbreak_template_common(
    instances: list[str] | str,
    templates: list[str] | None,
    available_templates: list[str],
    language: str,
    *,
    num_perturbations: int = 1,
) -> list[str]:
    """Applies jailbreak templates to each string in instances.

    Args:
        instances: A single string or a list of strings to be augmented.
        templates: A list templates to apply. If None, some templates are
            randomly selected and used.
        available_templates: A list of available templates.
        language: The language of the templates.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances. Should be equal to or less than the number
            of templates.

    Returns:
        A list of perturbed instances.
    """

    instances = [instances] if isinstance(instances, str) else instances

    if templates is None:
        templates = available_templates

    # Validation on the length of templates and num_perturbations
    if num_perturbations > len(templates):
        raise ValueError(
            "The number of perturbations should be equal to or less than the number of templates."
        )

    # Validate that only available templates are specified
    for template in templates:
        if template not in available_templates:
            raise ValueError(f"Invalid template: {template}")

    perturbed_instances = []
    for instance in instances:
        # Randomly select num_perturbations templates
        selected_templates = random.sample(templates, num_perturbations)
        for template_name in selected_templates:
            template = get_template(
                f"{language}/jailbreak_templates/{template_name}.j2"
            )
            perturbed_instances.append(
                template.render({"input_query": instance})
            )

    return perturbed_instances
