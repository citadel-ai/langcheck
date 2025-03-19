from __future__ import annotations

import random
from pathlib import Path

from jinja2 import Template

from langcheck.metrics.prompts._utils import get_template


def jailbreak_template_common(
    instances: list[str] | str,
    templates: list[str] | None,
    available_templates: list[str],
    language: str,
    *,
    num_perturbations: int = 1,
    randomize_order: bool = True,
    seed: int | None = None,
    custom_templates: list[tuple[str, str]] | None = None,
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
        randomize_order: If True, the order of the templates is randomized.
            When turned off, num_perturbations needs to be equal to the number
            of templates.
        seed: The seed for the random number generator. You can fix the seed to
            deterministically select the same templates.
        custom_templates: A list of tuples of names and paths to custom Jinja2
            templates. The template should contain an `{{input_query}}` placeholder,
            which will be replaced by the input query.

    Returns:
        A list of perturbed instances.
    """

    if seed is not None:
        random.seed(seed)

    instances = [instances] if isinstance(instances, str) else instances

    if templates is None:
        templates = available_templates

    if custom_templates is not None:
        for template_name, path_to_template in custom_templates:
            # validation
            if template_name in available_templates:
                raise ValueError(f"A template with the name {template_name} already exists!")
            template_file = Path(path_to_template)
            if not template_file.exists():
                raise ValueError(f"Custom template file {path_to_template} does not exist!")
            if not path_to_template.endswith(".j2"):
                raise ValueError("The custom template file must be a Jinja2 template file with the extension '.j2'!")
            available_templates = [*available_templates, template_name]

    # Validation on the length of templates and num_perturbations
    if num_perturbations > len(templates):
        raise ValueError(
            "The number of perturbations should be equal to or less than the number of templates."
        )

    if not randomize_order and num_perturbations < len(templates):
        raise ValueError(
            f"When randomize_order is False , the number of perturbations needs to be equal to the number of templates ({len(templates)})."
        )

    # Validate that only available templates are specified
    for template in templates:
        if template not in available_templates:
            raise ValueError(f"Invalid template: {template}")

    perturbed_instances = []
    for instance in instances:
        if randomize_order:
            # Randomly select num_perturbations templates
            selected_templates = random.sample(templates, num_perturbations)
        else:
            selected_templates = templates

        for template_name in selected_templates:
            # if the template is from custom_templates, fetch the path from there
            if custom_templates is not None and any(name == template_name for name, _ in custom_templates):
                template_path = next(path for name, path in custom_templates if name == template_name)
                template = Template(Path(template_path).read_text(encoding="utf-8"))
            else:
                template = get_template(
                    f"{language}/jailbreak_templates/{template_name}.j2"
                )
            perturbed_instances.append(
                template.render({"input_query": instance})
            )

    return perturbed_instances
