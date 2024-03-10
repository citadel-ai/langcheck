import importlib.util
import sys

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


def lazy_import(name):
    '''Lazily import the language-specific packages in langcheck.metrics.

    This prevents `import langcheck` from throwing ModuleNotFoundError if the
    user hasn't installed `langcheck[ja]`, while still allowing the package
    `langcheck.metrics.ja` to be visible even if the user didn't explicitly run
    `import langcheck.metrics.ja`.

    Copied from: https://docs.python.org/3/library/importlib.html#implementing-lazy-imports  # NOQA: E501
    '''
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None  # For type checking
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


# Use lazy import instead of directly importing language-specific packages
en = lazy_import('langcheck.metrics.en')
ja = lazy_import('langcheck.metrics.ja')
de = lazy_import('langcheck.metrics.de')
zh = lazy_import('langcheck.metrics.zh')

__all__ = [
    'en',
    'ja',
    'de',
    'zh',
    'ai_disclaimer_similarity',
    'answer_relevance',
    'contains_all_strings',
    'contains_any_strings',
    'contains_regex',
    'context_relevance',
    'MetricValue',
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
    'rouge1',
    'rouge2',
    'rougeL',
    'validation_fn',
    'semantic_similarity',
    'sentiment',
    'toxicity',
]
