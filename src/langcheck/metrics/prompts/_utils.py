from pathlib import Path

from jinja2 import Template


def get_template(relative_path: str) -> Template:
    '''
    Gets a Jinja template from the specified prompt template file.

    Args:
        relative_path (str): The relative path of the template file.

    Returns:
        Template: The Jinja template.
    '''
    cwd = Path(__file__).parent
    return Template((cwd / relative_path).read_text())
