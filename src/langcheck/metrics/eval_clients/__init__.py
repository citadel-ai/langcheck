from jinja2 import Template

from langcheck.metrics.eval_clients._base import EvalClient
from langcheck.metrics.eval_clients._openai import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
)
from langcheck.metrics.eval_clients._prometheus import PrometheusEvalClient

from ..prompts._utils import get_template

__all__ = [
    "AzureOpenAIEvalClient",
    "EvalClient",
    "OpenAIEvalClient",
    "PrometheusEvalClient",
]

try:
    from langcheck.metrics.eval_clients._anthropic import (
        AnthropicEvalClient,  # NOQA: F401
    )
except ModuleNotFoundError:
    pass
else:
    __all__.append("AnthropicEvalClient")

try:
    from langcheck.metrics.eval_clients._gemini import (
        GeminiEvalClient,  # NOQA: F401
    )
except ModuleNotFoundError:
    pass
else:
    __all__.append("GeminiEvalClient")


def load_prompt_template(
    language: str, eval_client: EvalClient, metric_name: str
) -> Template:
    """
    Gets a Jinja template from the specified language, eval client,
    and metric name.

    Args:
        language (str): The language of the template.
        eval_client (EvalClient): The evaluation client to use.
        metric_name (str): The name of the metric.

    Returns:
        Template: The Jinja template.
    """
    if type(eval_client) is PrometheusEvalClient:
        try:
            return get_template(f"{language}/metrics/prometheus/{metric_name}.j2")
        except FileNotFoundError:
            raise ValueError(
                f"The {metric_name} metric (language = {language}) is not yet supported by the Prometheus eval client."
            )
    return get_template(f"{language}/metrics/{metric_name}.j2")
