from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template


def load_prompt_template(
    language: str,
    metric_name: str,
    eval_prompt_version: str | None = None,
) -> Template:
    """
    Gets a Jinja template from the specified language, eval client, metric
    name, and (optionally) eval prompt version.

    Args:
        language (str): The language of the template.
        metric_name (str): The name of the metric.
        eval_prompt_version (str | None): The version of the eval prompt.
            If None, the default version is used.

    Returns:
        Template: The Jinja template.
    """
    if eval_prompt_version is None:
        return get_template(f"{language}/metrics/{metric_name}.j2")
    return get_template(
        f"{language}/metrics/{metric_name}_{eval_prompt_version}.j2"
    )


def get_template(relative_path: str) -> Template:
    """
    Gets a Jinja template from the specified prompt template file.

    Args:
        relative_path (str): The relative path of the template file.

    Returns:
        Template: The Jinja template.
    """
    cwd = Path(__file__).parent
    return Template((cwd / relative_path).read_text(encoding="utf-8"))


def load_few_shot_examples(relative_path: str) -> list[dict[str, str]]:
    """
    Loads few-shot examples from a JSONL file.

    Args:
        relative_path (str): The relative path of the JSONL file.

    Returns:
        list[str]: The few-shot examples.
    """
    cwd = Path(__file__).parent
    with open(cwd / relative_path) as f:
        return [json.loads(line) for line in f]
