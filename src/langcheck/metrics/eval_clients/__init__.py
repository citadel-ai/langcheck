from langcheck.metrics.eval_clients._base import EvalClient
from langcheck.metrics.eval_clients._openai import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
)

__all__ = [
    "AzureOpenAIEvalClient",
    "EvalClient",
    "OpenAIEvalClient",
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

try:
    from langcheck.metrics.eval_clients._llama import (
        LlamaEvalClient,  # NOQA: F401
    )
    from langcheck.metrics.eval_clients._prometheus import (
        PrometheusEvalClient,  # NOQA: F401
    )
except ModuleNotFoundError:
    pass
else:
    __all__.append("PrometheusEvalClient")
    __all__.append("LlamaEvalClient")
