from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        use_async: Whether to use the asynchronous API for OpenAI, default False

    Returns:
        An MetricValue object
    '''
    generated_outputs, sources, prompts = validate_parameters_source_based(
        generated_outputs, sources, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')

    if model_type == 'openai' or model_type == 'azure_openai':
        scores, explanations = _factual_consistency_openai(generated_outputs,
                                                           sources,
                                                           model_type,
                                                           openai_client,
                                                           openai_args,
                                                           use_async=use_async)

        return MetricValue(metric_name='factual_consistency',
                           prompts=prompts,
                           generated_outputs=generated_outputs,
                           reference_outputs=None,
                           sources=sources,
                           explanations=explanations,
                           metric_values=scores,
                           language=LANG)

    # Translate the sources and generated outputs to English.
    # Currently, the type checks are not working for the pipeline, since
    # too diverse types can be returned.
    translation = Translate(_factual_consistency_translation_model_path)
    batch_size = 8
    en_source = []
    for i in tqdm_wrapper(range(0, len(sources), batch_size),
                          desc='Translating sources',
                          total=(len(sources) + batch_size - 1) // batch_size):
        batch_sources = sources[i:i + batch_size]
        en_source.extend([translation(src) for src in batch_sources])
    en_generated_outputs = []
    for i in tqdm_wrapper(range(0, len(generated_outputs), batch_size),
                          desc='Translating generated outputs',
                          total=(len(generated_outputs) + batch_size - 1) //
                          batch_size):
        batch_generated_outputs = generated_outputs[i:i + batch_size]
        en_generated_outputs.extend(
            [translation(gen_out) for gen_out in batch_generated_outputs])

    # Compute the factual consistency scores in English.
    metric_value = en_factual_consistency(
        generated_outputs=en_generated_outputs, sources=en_source)
    metric_value.language = LANG
    return metric_value


def _factual_consistency_openai(
    generated_outputs: List[str],
    sources: List[str],
    client_type: str,
    client: Optional[OpenAI],
    openai_args: Optional[Dict[str, str]],
    *,
    use_async: bool = False
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    '''Calculates the factual consistency and their associated explanations
    between each generated output and its corresponding source text. The
    consistency is computed by calling the OpenAI API, with a prompt similar to
    the one used in OpenAI Evals. We leverage the function calling API to make
    sure that the output is structured such that we can compute a score. If a
    score could not be computed, `None` is inserted to the score and explanation
    lists.

    Ref:
        https://github.com/openai/evals/blob/e49868e550babb7b1c5b4223c9b7a14511bf114d/evals/registry/modelgraded/fact.yaml
        https://platform.openai.com/docs/guides/gpt/function-calling

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output
        client_type: The type of OpenAI client ('openai' or 'azure_openai')
        client: (Optional) OpenAI or AzureOpenAI client. If this is None, we
            will attempt to create a default client depending on the
            ``client_type``.
        openai_args: (Optional) Dict of additional args to pass in to the
            ``client.chat.completions.create`` function
        use_async: Whether to use the asynchronous API for OpenAI

    Returns:
        score_list: a list of scores
        explanation_list: a list of explanations for the scores
    '''

    # TODO: The prompt formation, and the scoring system, can do with some
    # improvement. There are some cases where consistent outputs get incorrectly
    # assessed as "Partially Consistent", and there's no differentiation
    # between an output that is unrelated to the source and an output that is
    # straight up contradictory.
    def _prompt(src: str, gen_output: str) -> str:
        return f'''
        Sie bewerten die faktische Konsistenz einer eingereichten Behauptung.
        Hier sind die Daten:
        [BEGINN DER DATEN]
        ************
        [Quelle]: {src}
        ************
        [Benutzeranfrage]: {gen_output}
        ************
        [ENDE DER DATEN]

        Bestimmen Sie, ob die eingereichte Behauptung faktisch konsistent mit
        der Quelle ist. Die verfügbaren Bewertungen sind:
        `Vollständig Konsistent` - Die eingereichte Behauptung ist vollständig
        faktisch konsistent mit dem Quelltext.
        `Teilweise Konsistent` - Die eingereichte Behauptung ist teilweise
        faktisch konsistent mit dem Quelltext. Es gibt einige Aspekte der
        Behauptung, die faktisch konsistent sind, aber auch einige, die es
         nicht sind.
        `Nicht Konsistent` - Die eingereichte Behauptung ist nicht faktisch
        konsistent mit dem Quelltext.

        Atmen Sie tief durch und bearbeiten Sie dieses Problem Schritt für
        Schritt.
        '''

    def _function_call_prompt(long_assessment: str) -> str:
        return f'''
        Folgendes ist eine Bewertung zur faktischen Konsistenz einer Behauptung:
        ************
        [Bewertung]: {long_assessment}
        ************

        Speichern Sie die resultierende Bewertung. Die verfügbaren Bewertungen
         sind:
        `Vollständig Konsistent`
        `Teilweise Konsistent`
        `Nicht Konsistent`
        '''

    factuality_assessment_to_score = {
        'Vollständig Konsistent': 1.0,
        'Teilweise Konsistent': 0.5,
        'Nicht Konsistent': 0.0
    }
    oai_evaluator = OpenAIBasedEvaluator(
        assessment_to_score_mapping=factuality_assessment_to_score,
        function_name='save_factual_consistency_assessment',
        function_description=(
            "Saves a submitted claim's factual consistency assessment."),
        argument_name='factuality',
        argument_description='The factual consistency assessment of the claim',
        client_type=client_type,
        client=client,
        openai_args=openai_args,
        use_async=use_async)

    scores, explanations = oai_evaluator.get_score(
        map(_prompt, sources, generated_outputs), _function_call_prompt)

    return scores, explanations


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
        use_async: Whether to use the asynchronous API for OpenAI, default False
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
        openai_args=openai_args,
        use_async=use_async)

    scores, explanations = oai_evaluator.get_score(
        map(_prompt, sources, prompts), _function_call_prompt)

    return MetricValue(metric_name='context_relevance',
                       prompts=prompts,
                       generated_outputs=None,
                       reference_outputs=None,
                       sources=list(sources),
                       explanations=explanations,
                       metric_values=scores,
                       language=LANG)
