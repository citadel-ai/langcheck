from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template


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
