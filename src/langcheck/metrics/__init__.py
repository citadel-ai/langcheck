from langcheck.metrics import en, eval_clients
from langcheck.metrics.custom_text_quality import (
    custom_evaluator,
    custom_pairwise_evaluator,
)
from langcheck.metrics.en.pairwise_text_quality import pairwise_comparison
from langcheck.metrics.en.query_based_text_quality import (
    adult_content,
    answer_relevance,
    answer_safety,
    harmful_activity,
    hate_speech,
    personal_data_leakage,
)
from langcheck.metrics.en.reference_based_text_quality import (
    answer_correctness,
    rouge1,
    rouge2,
    rougeL,
    semantic_similarity,
)
from langcheck.metrics.en.reference_free_text_quality import (
    ai_disclaimer_similarity,
    flesch_kincaid_grade,
    flesch_reading_ease,
    fluency,
    jailbreak_prompt,
    prompt_leakage,
    sentiment,
    toxicity,
)
from langcheck.metrics.en.source_based_text_quality import (
    context_relevance,
    factual_consistency,
)
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.reference_based_text_quality import exact_match
from langcheck.metrics.text_structure import (
    contains_all_strings,
    contains_any_strings,
    contains_regex,
    is_float,
    is_int,
    is_json_array,
    is_json_object,
    matches_regex,
    validation_fn,
)

__all__ = [
    "adult_content",
    "ai_disclaimer_similarity",
    "answer_correctness",
    "answer_relevance",
    "answer_safety",
    "contains_all_strings",
    "contains_any_strings",
    "contains_regex",
    "context_relevance",
    "custom_evaluator",
    "custom_pairwise_evaluator",
    "MetricValue",
    "en",
    "eval_clients",
    "exact_match",
    "factual_consistency",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "fluency",
    "harmful_activity",
    "hate_speech",
    "is_float",
    "is_int",
    "is_json_array",
    "is_json_object",
    "jailbreak_prompt",
    "matches_regex",
    "pairwise_comparison",
    "personal_data_leakage",
    "prompt_leakage",
    "rouge1",
    "rouge2",
    "rougeL",
    "validation_fn",
    "semantic_similarity",
    "sentiment",
    "toxicity",
]

# Try to import language-specific packages. These packages will be hidden if
# the user didn't pip install the required language.
try:
    from langcheck.metrics import ja  # NOQA: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("ja")

try:
    from langcheck.metrics import de  # NOQA: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("de")

try:
    from langcheck.metrics import zh  # NOQA: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("zh")
