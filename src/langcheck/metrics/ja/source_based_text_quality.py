from __future__ import annotations

from typing import Dict, List, Optional, cast

from openai import OpenAI
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline

from langcheck.metrics._validation import validate_parameters_source_based
from langcheck.metrics.en.source_based_text_quality import \
    factual_consistency as en_factual_consistency
from langcheck.metrics.metric_value import MetricValue

_factual_consistency_translation_model_path = 'Helsinki-NLP/opus-mt-ja-en'
_factual_consistency_translation_pipeline: Pipeline | None = None


def factual_consistency(
    generated_outputs: List[str] | str,
    sources: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    model_type: str = 'local',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    '''Calculates the factual consistency between the generated outputs and
    the sources. The factual consistency score for one generated output is
    computed as the average of the per-sentence consistencies of the generated
    output with the source text. This metric takes on float values between
    [0, 1], where 0 means that the output is not at all consistent with the
    source text, and 1 means that the output is fully consistent with the source
    text. (NOTE: when using the OpenAI model, the factuality score for each
    sentence is either 0.0, 0.5, or 1.0. The score may also be `None` if it
    could not be computed.)

    We currently support three model types:

    1. The 'local' type, where the 'unieval-fact' model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.
    This function wraps :func:`~langcheck.metrics.en.en_factual_consistency`
    using the translation model ``Helsinki-NLP/opus-mt-ja-en`` to translate the
    Japanese texts to English before computing the factual consistency
    scores. This is because the UniEval-fact model is trained on English
    text.

    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this page <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    3. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        model_type: The type of model to use ('local', 'openai', or
            'azure_openai'), default 'local'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None but ``model_type`` is 'openai' or 'azure_openai', we will
            attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.chat.completions.create`` function, default None

    Returns:
        An MetricValue object
    '''
    generated_outputs, sources, prompts = validate_parameters_source_based(
        generated_outputs, sources, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')

    # The English prompt works well enough for Japanese
    # TODO: Investigate the performance improvement with Japanese prompt
    if model_type == 'openai' or model_type == 'azure_openai':
        metric_value = en_factual_consistency(generated_outputs, sources,
                                              prompts, model_type,
                                              openai_client, openai_args)
        metric_value.language = 'ja'
        return metric_value

    global _factual_consistency_translation_pipeline
    if _factual_consistency_translation_pipeline is None:
        _factual_consistency_translation_pipeline = pipeline(
            'translation', model=_factual_consistency_translation_model_path)

    # Translate the sources and generated outputs to English.
    # Currently, the type checks are not working for the pipeline, since
    # too diverse types can be returned.
    en_source = [
        cast(str,
             d['translation_text'])  # type: ignore[reportGeneralTypeIssues]
        for d in _factual_consistency_translation_pipeline(sources)
    ]
    en_generated_outputs = [
        cast(str,
             d['translation_text'])  # type: ignore[reportGeneralTypeIssues]
        for d in _factual_consistency_translation_pipeline(generated_outputs)
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
                       language='ja')
