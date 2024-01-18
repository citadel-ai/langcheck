from __future__ import annotations

from typing import Dict, List, Optional

from openai import OpenAI

from langcheck.metrics._validation import (
    validate_parameters_context_relevance, validate_parameters_source_based)
from langcheck.metrics.de._translation import Translate
from langcheck.metrics.en._openai import OpenAIBasedEvaluator
from langcheck.metrics.en.source_based_text_quality import \
    factual_consistency as en_factual_consistency
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper

_factual_consistency_translation_model_path = 'Helsinki-NLP/opus-mt-de-en'

LANG = 'de'


def factual_consistency(
    generated_outputs: List[str] | str,
    sources: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    model_type: str = 'local',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
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
    This function wraps :func:`~langcheck.metrics.en.factual_consistency`
    using the translation model ``Helsinki-NLP/opus-mt-de-en`` to translate the
    German texts to English before computing the factual consistency
    scores. This is because the UniEval-fact model is trained on English
    text.

    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this example <https://langcheck.readthedocs.io/en/latest/metrics.html
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

    # The English prompt works well enough for German, like with Japanese
    # TODO: Investigate performance improvement with German prompt / translation
    if model_type == 'openai' or model_type == 'azure_openai':
        metric_value = en_factual_consistency(generated_outputs, sources,
                                              prompts, model_type,
                                              openai_client, openai_args)
        metric_value.language = LANG
        return metric_value

    translation = Translate(_factual_consistency_translation_model_path)

    # Translate the sources and generated outputs to English.
    # Currently, the type checks are not working for the pipeline, since
    # too diverse types can be returned.
    en_source = [translation(source) for source in sources]
    en_generated_outputs = [
        translation(gen_out) for gen_out in generated_outputs
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
                       language=LANG)


def context_relevance(
    sources: List[str] | str,
    prompts: List[str] | str,
    model_type: str = 'openai',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
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
    '''
    prompts, sources = validate_parameters_context_relevance(prompts, sources)

    def _prompt(src: str, user_query: str) -> str:
        return f'''
        Sie bewerten die Relevanz der Quelle für eine Benutzeranfrage. Hier
        sind die Daten:
        [BEGINN DATEN]
        ************
        [Quelle]: {src}
        ************
        [Benutzeranfrage]: {user_query}
        ************
        [ENDE DATEN]

        Bestimmen Sie, ob die Quelle die relevanten und notwendigen
        Informationen enthält, um auf die Anfrage des Benutzers zu antworten.
        Die verfügbaren Bewertungen sind:
        `Vollständig relevant` - Der Quelltext enthält die Informationen, die
        notwendig sind, um auf die Anfrage des Benutzers zu antworten.
        `Teilweise relevant` - Der Quelltext ist teilweise relevant für die
        Anfrage des Benutzers, enthält aber nicht alle Informationen,
        die notwendig sind, um auf die Anfrage des Benutzers zu antworten.
        `Nicht relevant` - Der Quelltext ist nicht relevant für die Anfrage des
        Benutzers.

        Atmen Sie tief durch und arbeiten Sie Schritt für Schritt an diesem
        Problem.
        '''

    def _function_call_prompt(long_assessment: str) -> str:
        return f'''
        Folgendes ist eine Bewertung über die Relevanz einer Quelle:
        ************
        [Bewertung]: {long_assessment}
        ************

        Speichern Sie die resultierende Bewertung. Die verfügbaren Bewertungen
        sind:
        `Vollständig relevant`
        `Teilweise relevant`
        `Nicht relevant`
        '''

    context_relevance_assessment_to_score = {
        'Vollständig relevant': 1.0,
        'Teilweise relevant': 0.5,
        'Nicht relevant': 0.0
    }
    oai_evaluator = OpenAIBasedEvaluator(
        assessment_to_score_mapping=context_relevance_assessment_to_score,
        function_name='save_context_relevance_assessment',
        function_description=("Saves a context relevance assessment."),
        argument_name='context_relevance',
        argument_description='The context relevance assessment',
        client_type=model_type,
        client=openai_client,
        openai_args=openai_args)

    score_list = []
    explanation_list = []
    for src, user_query in tqdm_wrapper(zip(sources, prompts),
                                        desc='Calculating scores',
                                        total=len(prompts)):
        score, explanation = oai_evaluator.get_score(
            _prompt(src=src, user_query=user_query), _function_call_prompt)
        score_list.append(score)
        explanation_list.append(explanation)

    return MetricValue(metric_name='context_relevance',
                       prompts=prompts,
                       generated_outputs=None,
                       reference_outputs=None,
                       sources=sources,
                       explanations=explanation_list,
                       metric_values=score_list,
                       language=LANG)
