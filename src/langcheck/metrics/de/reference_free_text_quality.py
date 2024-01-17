from __future__ import annotations

from typing import Dict, List, Optional, cast

import torch
from openai import OpenAI
from transformers.models.auto.modeling_auto import \
    AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from langcheck._handle_logs import _handle_logging_level
from langcheck.metrics._detoxify import Detoxify
from langcheck.metrics._validation import validate_parameters_reference_free
from langcheck.metrics.de._translation import Translate
from langcheck.metrics.de.reference_based_text_quality import \
    semantic_similarity
from langcheck.metrics.en.reference_free_text_quality import _toxicity_openai
from langcheck.metrics.en.reference_free_text_quality import \
    flesch_kincaid_grade as en_flesch_kincaid_grade
from langcheck.metrics.en.reference_free_text_quality import \
    fluency as en_fluency
from langcheck.metrics.en.reference_free_text_quality import \
    sentiment as en_sentiment
from langcheck.metrics.metric_value import MetricValue
from langcheck.stats import compute_stats
from langcheck.utils.progess_bar import tqdm_wrapper

# this model seems to work much better than the cardiffnlp one; typo in the name
_sentiment_model_path = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"  # NOQA: E501
_sentiment_tokenizer = None
_sentiment_model = None

_translation_model_path = 'Helsinki-NLP/opus-mt-de-en'

_toxicity_model = None

LANG = 'de'


def sentiment(
    generated_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    model_type: str = 'local',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    '''Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using the OpenAI model, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive). The score may
    also be `None` if it could not be computed.)

    We currently support three model types:

    1. The 'local' type, where the twitter-xlm-roberta-base-sentiment-finetunned
    model is downloaded from HuggingFace and run locally. This is the default
    model type and there is no setup needed to run this.

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

    Ref:
        https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned

    Args:
        generated_outputs: The model generated output(s) to evaluate
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
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')

    # The English prompt works well enough for German
    # TODO: Investigate the performance improvement with German prompt
    if model_type == 'openai' or model_type == 'azure_openai':
        metric_value = en_sentiment(generated_outputs, prompts, model_type,
                                    openai_client, openai_args)
        metric_value.language = LANG
        return metric_value

    global _sentiment_tokenizer, _sentiment_model

    if _sentiment_tokenizer is None or _sentiment_model is None:
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(
            _sentiment_model_path)

        # There is a "Some weights are not used warning" but we ignore it
        # because that is intended.
        with _handle_logging_level():
            _sentiment_model = (AutoModelForSequenceClassification.
                                from_pretrained(_sentiment_model_path))

    input_tokens = _sentiment_tokenizer(generated_outputs,
                                        return_tensors='pt',
                                        padding=True)

    with torch.no_grad():
        # Probabilities of [negative, neutral, positive]
        probs = torch.nn.functional.softmax(
            _sentiment_model(**input_tokens).logits, dim=1)

    scores = (probs[:, 1] / 2 + probs[:, 2]).tolist()

    return MetricValue(metric_name='sentiment',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language=LANG)


def fluency(
    generated_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    model_type: str = 'local',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    ''' We first translate the generated outputs to English, and then use the
    Parrot fluency model to calculate the fluency scores, from the English
    counterpart.
    '''
    translation = Translate(_translation_model_path)

    if isinstance(generated_outputs, str):
        generated_outputs = [generated_outputs]

    # Translate to English
    generated_outputs_en = [translation(str) for str in generated_outputs]

    _metric_value = en_fluency(generated_outputs_en, prompts, model_type,
                               openai_client, openai_args)
    metric_value = MetricValue(
        metric_name=_metric_value.metric_name,
        prompts=_metric_value.prompts,
        generated_outputs=generated_outputs,
        reference_outputs=_metric_value.reference_outputs,
        sources=_metric_value.sources,
        explanations=_metric_value.explanations,
        metric_values=_metric_value.metric_values,
        language=LANG)
    return metric_value


def toxicity(
    generated_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    model_type: str = 'local',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    '''Calculates the toxicity scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low toxicity and 1 is high toxicity.
    (NOTE: when using the OpenAI model, the toxicity scores are in steps of
    0.25. The score may also be `None` if it could not be computed.)

    We currently support three model types:

    1. The 'local' type, where the multilingual Detoxify model is downloaded
    from GitHub and run locally. This is the default model type and there is
    no setup needed to run this.

    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default, in the same way as english counterpart. While the model you use
    is configurable, please make sure to use one that supports function calling
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
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')

    if model_type == 'local':
        scores = _toxicity_local(generated_outputs)
        explanations = None
    else:  # openai or azure_openai
        scores, explanations = _toxicity_openai(generated_outputs, model_type,
                                                openai_client, openai_args)

    return MetricValue(metric_name='toxicity',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanations,
                       metric_values=scores,
                       language=LANG)


def _toxicity_local(generated_outputs: List[str]) -> List[float]:
    '''Calculates the toxicity scores of generated outputs using the Detoxify
    model. This metric takes on float values between [0, 1], where 0 is low
    toxicity and 1 is high toxicity.

    Ref:
        https://github.com/unitaryai/detoxify

    Args:
        generated_outputs: A list of model generated outputs to evaluate

    Returns:
        A list of scores
    '''
    global _toxicity_model
    if _toxicity_model is None:
        _toxicity_model = Detoxify(lang=LANG)

    scores = []
    batch_size = 8
    for i in tqdm_wrapper(range(0, len(generated_outputs), batch_size),
                          total=(len(generated_outputs) + batch_size - 1) //
                          batch_size):
        scores.extend(
            _toxicity_model.predict(generated_outputs[i:i +
                                                      batch_size])['toxicity'])

    return scores


def flesch_kincaid_grade(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None) -> MetricValue[float]:
    '''Calculates the readability of generated outputs using the Flesch-Kincaid.
    It is the same as in English (but higher):
    ref:
    https://de.wikipedia.org/wiki/Lesbarkeitsindex#Flesch-Kincaid-Grade-Level
    '''
    metric_value = en_flesch_kincaid_grade(generated_outputs, prompts)
    metric_value.language = LANG
    return metric_value


def flesch_reading_ease(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None) -> MetricValue[float]:
    '''Calculates the readability of generated outputs using the Flesch Reading
    Ease Score. This metric takes on float values between (-∞, 121.22], but
    typically ranges between 0 and 100, where higher scores mean the text is
    easier to read.

    The score is based on the number of sentences, words, and syllables in the
    text. See "How to Write Plain English" by Rudolf Franz Flesch for more
    details.
    For the German Formula, see
    https://de.wikipedia.org/wiki/Lesbarkeitsindex#Flesch-Reading-Ease
    FRE(Deutsch) = 180 - ASL - 58.5 * ASW

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)

    output_stats = [
        compute_stats(output)
        for output in tqdm_wrapper(generated_outputs, desc='Computing stats')
    ]
    scores = [
        180 - (stat.num_words / stat.num_sentences) - 58.5 *
        (stat.num_syllables / stat.num_words) for stat in output_stats
    ]
    return MetricValue(metric_name='flesch_reading_ease',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language=LANG)


def ai_disclaimer_similarity(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        ai_disclaimer_phrase: str = (
            "Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein."
        ),
        openai_client: Optional[OpenAI] = None,
        model_type: str = 'local',
        openai_args: Optional[Dict[str, str]] = None) -> MetricValue[float]:
    '''Calculates the degree to which the LLM's output contains a disclaimer
    that it is an AI. This is calculated by computing the semantic similarity
    between the generated outputs and a reference AI disclaimer phrase; by
    default, this phrase is "Ich habe keine persönlichen Meinungen, Emotionen
    oder Bewusstsein." (the most common reply from chatGPT in German),
    but you can also pass in a custom phrase. Please refer to
    :func:`~langcheck.eval.de.reference_based_text_quality.semantic_similarity`
    for details on the typical output ranges and the supported embedding model
    types.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: An optional list of prompts used to generate the outputs.
            Prompts are not evaluated and only used as metadata.
        ai_disclaimer_phrase: Reference AI disclaimer phrase, default "I don't
            have personal opinions, emotions, or consciousness."
        model_type: The type of embedding model to use ('local', 'openai', or
            'azure_openai'), default 'local'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None but ``model_type`` is 'openai' or 'azure_openai', we will
            attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.embeddings.create`` function, default None

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)

    ai_disclaimer_phrase_list = [ai_disclaimer_phrase] * len(generated_outputs)
    semantic_similarity_values = semantic_similarity(generated_outputs,
                                                     ai_disclaimer_phrase_list,
                                                     prompts, model_type,
                                                     openai_client, openai_args)
    return MetricValue(metric_name='ai_disclaimer_similarity',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=semantic_similarity_values.metric_values,
                       language=LANG)
