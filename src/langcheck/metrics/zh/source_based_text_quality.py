from __future__ import annotations

from typing import List, Optional, cast

from transformers.pipelines import pipeline

from langcheck.metrics._validation import validate_parameters_source_based
from langcheck.metrics.en.source_based_text_quality import \
    factual_consistency as en_factual_consistency
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue


def factual_consistency(
        generated_outputs: List[str] | str,
        sources: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        eval_model: str | EvalClient = 'local') -> MetricValue[Optional[float]]:
    '''Calculates the factual consistency between the generated outputs and
    the sources. This metric takes on float values between [0, 1], where 0
    means that the output is not at all consistent with the source text, and 1
    means that the output is fully consistent with the source text. (NOTE: when
    using an EvalClient, the factuality scores are either 0.0, 0.5, or 1.0.
    The score may also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the 'unieval-fact' model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.
    This function wraps :func:`~langcheck.metrics.en.en_factual_consistency`
    using the translation model ``Helsinki-NLP/opus-mt-zh-en`` to translate the
    Chinese texts to English before computing the factual consistency
    scores. This is because the UniEval-fact model is trained on English
    text.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'

    Returns:
        An MetricValue object
    '''
    generated_outputs, sources, prompts = validate_parameters_source_based(
        generated_outputs, sources, prompts)

    if eval_model != 'local':  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), 'An EvalClient must be provided for non-local model types.'
        metric_value = en_factual_consistency(generated_outputs, sources,
                                              prompts, eval_model)
        metric_value.language = 'zh'
        return metric_value

    from langcheck.metrics.model_manager import manager
    tokenizer, model = manager.fetch_model(language='zh',
                                           metric='factual_consistency')
    _factual_consistency_translation_pipeline = pipeline(
        'translation',
        model=model,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
        truncation=True)

    # Translate the sources and generated outputs to English.
    # Currently, the type checks are not working for the pipeline, since
    # too diverse types can be returned.
    en_source = [
        cast(str,
             d['translation_text'])  # type: ignore[reportGeneralTypeIssues]
        for d in _factual_consistency_translation_pipeline(
            sources)  # type: ignore[reportOptionalIterable]
    ]
    en_generated_outputs = [
        cast(str,
             d['translation_text'])  # type: ignore[reportGeneralTypeIssues]
        for d in _factual_consistency_translation_pipeline(
            generated_outputs)  # type: ignore[reportOptionalIterable]
    ]
    # Compute the factual consistency scores in English.
    factual_consistency_scores = en_factual_consistency(
        generated_outputs=en_generated_outputs, sources=en_source).metric_values

    return MetricValue(metric_name='factual_consistency',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=sources,
                       explanations=None,
                       metric_values=factual_consistency_scores,
                       language='zh')
