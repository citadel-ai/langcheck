from __future__ import annotations

from typing import Dict, List, Optional, cast

from openai import OpenAI
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline

from langcheck.metrics._validation import (
    validate_parameters_context_relevance, validate_parameters_source_based)
from langcheck.metrics.en._openai import OpenAIBasedEvaluator
from langcheck.metrics.en.source_based_text_quality import \
    factual_consistency as en_factual_consistency
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper

_factual_consistency_translation_model_path = 'Helsinki-NLP/opus-mt-ja-en'
_factual_consistency_translation_pipeline: Pipeline | None = None


def factual_consistency(
        generated_outputs: List[str] | str,
        sources: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        model_type: str = 'local',
        openai_client: Optional[OpenAI] = None,
        openai_args: Optional[Dict[str, str]] = None,
        *,
        use_async: bool = False) -> MetricValue[Optional[float]]:
    '''Calculates the factual consistency between the generated outputs and
    the sources. This metric takes on float values between [0, 1], where 0
    means that the output is not at all consistent with the source text, and 1
    means that the output is fully consistent with the source text. (NOTE: when
    using the OpenAI model, the factuality scores are either 0.0, 0.5, or 1.0.
    The score may also be `None` if it could not be computed.)

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
        use_async: Whether to use the asynchronous API of OpenAI, default False

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
        metric_value = en_factual_consistency(generated_outputs,
                                              sources,
                                              prompts,
                                              model_type,
                                              openai_client,
                                              openai_args,
                                              use_async=use_async)
        metric_value.language = 'ja'
        return metric_value

    global _factual_consistency_translation_pipeline
    if _factual_consistency_translation_pipeline is None:
        _factual_consistency_translation_pipeline = pipeline(
            'translation',
            model=_factual_consistency_translation_model_path,
            truncation=True)

    # Translate the sources and generated outputs to English.
    # Currently, the type checks are not working for the pipeline, since
    # too diverse types can be returned.
    batch_size = 8
    en_source = []
    for i in tqdm_wrapper(range(0, len(sources), batch_size),
                          desc='Translating sources',
                          total=(len(sources) + batch_size - 1) // batch_size):
        batch_sources = sources[i:i + batch_size]
        en_source.extend([
            cast(str,
                 d['translation_text'])  # type: ignore[reportGeneralTypeIssues]
            for d in _factual_consistency_translation_pipeline(
                batch_sources)  # type: ignore[reportGeneralTypeIssues]
        ])
    en_generated_outputs = []
    for i in tqdm_wrapper(range(0, len(generated_outputs), batch_size),
                          desc='Translating generated outputs',
                          total=(len(generated_outputs) + batch_size - 1) //
                          batch_size):
        batch_generated_outputs = generated_outputs[i:i + batch_size]
        en_generated_outputs.extend([
            cast(str,
                 d['translation_text'])  # type: ignore[reportGeneralTypeIssues]
            for d in _factual_consistency_translation_pipeline(
                batch_generated_outputs
            )  # type: ignore[reportGeneralTypeIssues]
        ])

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


def context_relevance(sources: List[str] | str,
                      prompts: List[str] | str,
                      model_type: str = 'openai',
                      openai_client: Optional[OpenAI] = None,
                      openai_args: Optional[Dict[str, str]] = None,
                      *,
                      use_async: bool = False) -> MetricValue[Optional[float]]:
    '''Calculates the relevance of the sources to the prompts. This metric takes
    on float values between [0, 1], where 0 means that the source text is not at
    all relevant to the prompt, and 1 means that the source text is fully
    relevant to the prompt.

    We currently support two model types:

    1. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this page <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    2. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Args:
        sources: The source text(s), one string per prompt
        prompts: The prompt(s)
        model_type: The type of model to use ('openai' or 'azure_openai'),
            default 'openai'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None, we will attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.chat.completions.create`` function, default None
        use_async: Whether to use the asynchronous API, default False
    '''
    prompts, sources = validate_parameters_context_relevance(prompts, sources)

    def _prompt(src: str, user_query: str) -> str:
        return f'''
        ユーザーの質問に対してソースの関連性を評価してください。データは以下の通りです:
        [BEGIN DATA]
        ************
        [ソース]: {src}
        ************
        [ユーザーの質問]: {user_query}
        ************
        [END DATA]

        ユーザーの質問に対応するために必要な、関連性のある情報がソースに含まれているかを判断
        してください。利用可能な評価は以下の通りです:
        `Fully Relevant` - ソーステキストには、ユーザーの質問に対応するために必要な情報が
        含まれています。
        `Partially Relevant` - ソーステキストはユーザーの質問に部分的に関連していますが、質問に
        対応するために必要なすべての情報を含んでいません。
        `Not Relevant` - ソーステキストはユーザーの質問に関連していません。

        深呼吸をして、この問題をステップバイステップで取り組んでください。
        '''

    def _function_call_prompt(long_assessment: str) -> str:
        return f'''
        以下はソースの関連性に関する評価です:
        ************
        [評価]: {long_assessment}
        ************

        結果として出た評価を保存してください。利用可能な評価は以下の通りです:
        `Fully Relevant`
        `Partially Relevant`
        `Not Relevant`
        '''

    context_relevance_assessment_to_score = {
        'Fully Relevant': 1.0,
        'Partially Relevant': 0.5,
        'Not Relevant': 0.0
    }
    oai_evaluator = OpenAIBasedEvaluator(
        assessment_to_score_mapping=context_relevance_assessment_to_score,
        function_name='save_context_relevance_assessment',
        function_description=("Saves a context relevance assessment."),
        argument_name='context_relevance',
        argument_description='The context relevance assessment',
        client_type=model_type,
        client=openai_client,
        openai_args=openai_args,
        use_async=use_async)

    scores, explanations = oai_evaluator.get_score(
        map(_prompt, sources, prompts), _function_call_prompt)

    return MetricValue(metric_name='context_relevance',
                       prompts=prompts,
                       generated_outputs=None,
                       reference_outputs=None,
                       sources=sources,
                       explanations=explanations,
                       metric_values=scores,
                       language='ja')
