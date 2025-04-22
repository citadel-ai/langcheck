from langcheck.metrics.eval_clients._base import EvalClient
from langcheck.metrics.eval_clients._openai import (
    AzureOpenAIEvalClient,
    AzureOpenAIExtractor,
    OpenAIEvalClient,
    OpenAIExtractor,
)
from langcheck.metrics.eval_clients._openrouter import (
    OpenRouterEvalClient,
    OpenRouterExtractor,
)

__all__ = [
    "AzureOpenAIEvalClient",
    "AzureOpenAIExtractor",
    "EvalClient",
    "OpenAIEvalClient",
    "OpenAIExtractor",
    "OpenRouterEvalClient",
    "OpenRouterExtractor",
]

try:
    from langcheck.metrics.eval_clients._anthropic import (
        AnthropicEvalClient,  # NOQA: F401
        AnthropicExtractor,  # NOQA: F401
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(["AnthropicEvalClient", "AnthropicExtractor"])

try:
    from langcheck.metrics.eval_clients._gemini import (
        GeminiEvalClient,  # NOQA: F401
        GeminiExtractor,  # NOQA: F401
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(["GeminiEvalClient", "GeminiExtractor"])

try:
    from langcheck.metrics.eval_clients._llama import (
        LlamaEvalClient,  # NOQA: F401
        LlamaExtractor,  # NOQA: F401
    )
    from langcheck.metrics.eval_clients._prometheus import (
        PrometheusEvalClient,  # NOQA: F401
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(["PrometheusEvalClient"])
    __all__.extend(["LlamaEvalClient", "LlamaExtractor"])
