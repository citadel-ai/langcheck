from langcheck.metrics import en
from langcheck.metrics.en.pairwise_text_quality import pairwise_comparison
from langcheck.metrics.en.reference_based_text_quality import (
    rouge1, rouge2, rougeL, semantic_similarity)
from langcheck.metrics.en.reference_free_text_quality import (
    ai_disclaimer_similarity, answer_relevance, flesch_kincaid_grade,
    flesch_reading_ease, fluency, sentiment, toxicity)
from langcheck.metrics.en.source_based_text_quality import (context_relevance,
                                                            factual_consistency)
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.reference_based_text_quality import exact_match
from langcheck.metrics.text_structure import (contains_all_strings,
                                              contains_any_strings,
                                              contains_regex, is_float, is_int,
                                              is_json_array, is_json_object,
                                              matches_regex, validation_fn)

__all__ = [
    'ai_disclaimer_similarity',
    'answer_relevance',
    'contains_all_strings',
    'contains_any_strings',
    'contains_regex',
    'context_relevance',
    'MetricValue',
    'en',
    'exact_match',
    'factual_consistency',
    'flesch_kincaid_grade',
    'flesch_reading_ease',
    'fluency',
    'is_float',
    'is_int',
    'is_json_array',
    'is_json_object',
    'matches_regex',
    'pairwise_comparison',
    'rouge1',
    'rouge2',
    'rougeL',
    'validation_fn',
    'semantic_similarity',
    'sentiment',
    'toxicity',
]

# Try to import language-specific packages. These packages will be hidden if
# the user didn't pip install the required language.
try:
    from langcheck.metrics import ja
except ModuleNotFoundError:
    pass
else:
    __all__.append('ja')

try:
    from langcheck.metrics import de
except ModuleNotFoundError:
    pass
else:
    __all__.append('de')

try:
    from langcheck.metrics import zh
except ModuleNotFoundError:
    pass
else:
    __all__.append('zh')
