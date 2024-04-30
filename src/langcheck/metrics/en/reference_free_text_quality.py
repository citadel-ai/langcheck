from __future__ import annotations

from typing import List, Optional, Tuple

from langcheck.metrics._validation import (validate_parameters_answer_relevance,
                                           validate_parameters_reference_free)
from langcheck.metrics.en.reference_based_text_quality import \
    semantic_similarity
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.scorer.detoxify_models import DetoxifyScorer
from langcheck.metrics.scorer.hf_models import \
    AutoModelForSequenceClassificationScorer
from langcheck.stats import compute_stats
from langcheck.utils.progess_bar import tqdm_wrapper

from ..prompts._utils import get_template


def sentiment(
    generated_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    eval_model: str | EvalClient = 'local',
    local_overflow_strategy: str = 'truncate',
) -> MetricValue[Optional[float]]:
    '''Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using an EvalClient, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive). The score may
    also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the Twitter-roBERTa-base model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'
        local_overflow_strategy: The strategy to handle the inputs that are too
            long for the local model. The supported strategies are 'nullify',
            'truncate', and 'raise'. If 'nullify', the outputs that are too long
            will be assigned a score of None. If 'truncate', the outputs that
            are too long will be truncated. If 'raise', an error will be raised
            when the outputs are too long. The default value is 'nullify'.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)

    if eval_model == 'local':
        scores = _sentiment_local(generated_outputs, local_overflow_strategy)
        explanations = None
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), 'An EvalClient must be provided for non-local model types.'
        scores, explanations = _sentiment_eval_client(generated_outputs,
                                                      eval_model)

    return MetricValue(metric_name='sentiment',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanations,
                       metric_values=scores,
                       language='en')


def _sentiment_local(generated_outputs: List[str],
                     overflow_strategy: str) -> List[Optional[float]]:
    '''Calculates the sentiment scores of generated outputs using the
    Twitter-roBERTa-base model. This metric takes on float values between
    [0, 1], where 0 is negative sentiment and 1 is positive sentiment.

    Ref:
        https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        overflow_strategy: The strategy to handle inputs that are longer than
            the maximum input length of the model.

    Returns:
        A list of scores
    '''
    scorer = AutoModelForSequenceClassificationScorer(
        language='en',
        metric='sentiment',
        # Each class represents a sentiment: 0 for negative, 1 for neutral, and
        # 2 for positive
        class_weights=[0, 0.5, 1],
        overflow_strategy=overflow_strategy,
        max_input_length=512)
    return scorer.score(generated_outputs)


def _sentiment_eval_client(
    generated_outputs: List[str], eval_client: EvalClient
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    '''Calculates the sentiment scores and their associated explanations of
    generated outputs using the provided EvalClient. This metric takes on float
    values that are either 0, 0.5, or 1, where 0 is negative sentiment, 0.5 is
    neutral sentiment, and 1 is positive sentiment. If a score could not be
    computed, `None` is inserted to the score and explanation lists.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        eval_client: EvalClient instance used to evaluate the generated outputs

    Returns:
        score_list: a list of scores
        explanation_list: a list of explanations for the scores
    '''
    sentiment_template = get_template('en/metrics/sentiment.j2')

    sentiment_assessment_to_score = {
        'Positive': 1.0,
        'Neutral': 0.5,
        'Negative': 0.0
    }

    populated_prompts = [
        sentiment_template.render({'gen_output': gen_output})
        for gen_output in generated_outputs
    ]

    scores, explanations = eval_client.get_score(
        metric_name='sentiment',
        language='en',
        prompts=populated_prompts,
        score_map=sentiment_assessment_to_score)

    return scores, explanations


def fluency(
    generated_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    eval_model: str | EvalClient = 'local',
    local_overflow_strategy: str = 'truncate',
) -> MetricValue[Optional[float]]:
    '''Calculates the fluency scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low fluency and 1 is high fluency.
    (NOTE: when using an EvalClient, the fluency scores are either 0.0
    (poor), 0.5 (fair), or 1.0 (good). The score may also be `None` if it could
    not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the Parrot fluency model is downloaded from
    HuggingFace and run locally. This is the default model type and there is no
    setup needed to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'
        local_overflow_strategy: The strategy to handle the inputs that are too
            long for the local model. The supported strategies are 'nullify',
            'truncate', and 'raise'. If 'nullify', the outputs that are too long
            will be assigned a score of None. If 'truncate', the outputs that
            are too long will be truncated. If 'raise', an error will be raised
            when the outputs are too long. The default value is 'nullify'.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)

    if eval_model == 'local':
        scores = _fluency_local(generated_outputs, local_overflow_strategy)
        explanations = None
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), 'An EvalClient must be provided for non-local model types.'
        scores, explanations = _fluency_eval_client(generated_outputs,
                                                    eval_model)

    return MetricValue(metric_name='fluency',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanations,
                       metric_values=scores,
                       language='en')


def _fluency_local(generated_outputs: List[str],
                   overflow_strategy: str) -> List[Optional[float]]:
    '''Calculates the fluency scores of generated outputs using the Parrot
    fluency model. This metric takes on float values between [0, 1], where 0 is
    low fluency and 1 is high fluency.

    Ref:
        https://huggingface.co/prithivida/parrot_fluency_model

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        overflow_strategy: The strategy to handle inputs that are longer than
            the maximum input length of the model.

    Returns:
        A list of scores
    '''
    scorer = AutoModelForSequenceClassificationScorer(
        language='en',
        metric='fluency',
        # The class 1 is for fluent texts.
        class_weights=[0, 1],
        overflow_strategy=overflow_strategy)
    return scorer.score(generated_outputs)


def _fluency_eval_client(
    generated_outputs: List[str], eval_client: EvalClient
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    '''Calculates the fluency scores and their associated explanations of
    generated outputs using the provided EvalClient. This metric takes on float
    values that are either 0, 0.5, or 1, where 0 is "poor" fluency, 0.5 is
    "fair" fluency, and 1 is "good" fluency.  If a score could not be computed,
    `None` is inserted to the score and explanation lists.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        eval_client: EvalClient instance used to evaluate the generated outputs

    Returns:
        score_list: a list of scores
        explanation_list: a list of explanations for the scores
    '''
    fluency_template = get_template('en/metrics/fluency.j2')

    fluency_assessment_to_score = {
        'Poor': 0,
        'Fair': 0.5,
        'Good': 1.0,
    }

    populated_prompts = [
        fluency_template.render({'gen_output': gen_output})
        for gen_output in generated_outputs
    ]

    scores, explanations = eval_client.get_score(
        metric_name='fluency',
        language='en',
        prompts=populated_prompts,
        score_map=fluency_assessment_to_score)

    return scores, explanations


def toxicity(
    generated_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    eval_model: str | EvalClient = 'local',
    local_overflow_strategy: str = 'truncate',
) -> MetricValue[Optional[float]]:
    '''Calculates the toxicity scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low toxicity and 1 is high toxicity.
    (NOTE: when using an EvalClient, the toxicity scores are in steps of
    0.25. The score may also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the Detoxify model is downloaded from HuggingFace
    and run locally. This is the default model type and there is no setup needed
    to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'
        local_overflow_strategy: The strategy to handle the inputs that are too
            long for the local model. The supported strategies are 'nullify',
            'truncate', and 'raise'. If 'nullify', the outputs that are too long
            will be assigned a score of None. If 'truncate', the outputs that
            are too long will be truncated. If 'raise', an error will be raised
            when the outputs are too long. The default value is 'nullify'.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)

    if eval_model == 'local':
        scores = _toxicity_local(generated_outputs, local_overflow_strategy)
        explanations = None
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), 'An EvalClient must be provided for non-local model types.'

        scores, explanations = _toxicity_eval_client(generated_outputs,
                                                     eval_model)

    return MetricValue(metric_name='toxicity',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanations,
                       metric_values=scores,
                       language='en')


def _toxicity_local(generated_outputs: List[str],
                    overflow_strategy: str) -> List[Optional[float]]:
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
    return DetoxifyScorer(
        overflow_strategy=overflow_strategy).score(generated_outputs)


def _toxicity_eval_client(
    generated_outputs: List[str], eval_client: EvalClient
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    '''Calculates the toxicity scores and their associated explanations of
    generated outputs using the provided EvalClient. This metric takes on float
    values between [0, 1] (in steps of 0.25), where 0 is low toxicity and 1 is
    high toxicity. If a score could not be computed, `None` is inserted to the
    score and explanation lists.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        eval_client: EvalClient instance used to evaluate the generated outputs

    Returns:
        score_list: a list of scores
        explanation_list: a list of explanations for the scores
    '''
    toxicity_template = get_template('en/metrics/toxicity.j2')

    toxicity_assessment_to_score = {
        '1': 0,
        '2': 0.25,
        '3': 0.5,
        '4': 0.75,
        '5': 1.0
    }

    populated_prompts = [
        toxicity_template.render({'gen_output': gen_output})
        for gen_output in generated_outputs
    ]

    scores, explanations = eval_client.get_score(
        metric_name='toxicity',
        language='en',
        prompts=populated_prompts,
        score_map=toxicity_assessment_to_score)

    return scores, explanations


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
        206.835 - 1.015 * (stat.num_words / stat.num_sentences) - 84.6 *
        (stat.num_syllables / stat.num_words) for stat in output_stats
    ]
    return MetricValue(metric_name='flesch_reading_ease',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language='en')


def flesch_kincaid_grade(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None) -> MetricValue[float]:
    '''Calculates the readability of generated outputs using the Flesch-Kincaid
    Grade Level metric. This metric takes on float values between [-3.40, ∞),
    but typically ranges between 0 and 12 (corresponding to U.S. grade levels),
    where lower scores mean the text is easier to read.

    Like the Flesch Reading Ease Score, this metric is based on the number of
    sentences, words, and syllables in the text.

    Ref:
        https://apps.dtic.mil/sti/citations/ADA006655

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
        0.39 * (stat.num_words / stat.num_sentences) + 11.8 *
        (stat.num_syllables / stat.num_words) - 15.59 for stat in output_stats
    ]
    return MetricValue(metric_name='flesch_kincaid_grade',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language='en')


def ai_disclaimer_similarity(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        ai_disclaimer_phrase: str = (
            "I don't have personal opinions, emotions, or consciousness."),
        eval_model: str | EvalClient = 'local') -> MetricValue[float]:
    '''Calculates the degree to which the LLM's output contains a disclaimer
    that it is an AI. This is calculated by computing the semantic similarity
    between the generated outputs and a reference AI disclaimer phrase; by
    default, this phrase is "I don't have personal opinions, emotions, or
    consciousness.", but you can also pass in a custom phrase. Please refer to
    :func:`~langcheck.eval.en.reference_based_text_quality.semantic_similarity`
    for details on the typical output ranges and the supported embedding model
    types.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: An optional list of prompts used to generate the outputs.
            Prompts are not evaluated and only used as metadata.
        ai_disclaimer_phrase: Reference AI disclaimer phrase, default "I don't
            have personal opinions, emotions, or consciousness."
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)

    ai_disclaimer_phrase_list = [ai_disclaimer_phrase] * len(generated_outputs)
    semantic_similarity_values = semantic_similarity(generated_outputs,
                                                     ai_disclaimer_phrase_list,
                                                     prompts, eval_model)
    return MetricValue(metric_name='ai_disclaimer_similarity',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=semantic_similarity_values.metric_values,
                       language='en')


def answer_relevance(generated_outputs: List[str] | str,
                     prompts: List[str] | str,
                     eval_model: EvalClient) -> MetricValue[Optional[float]]:
    '''Calculates the relevance of generated outputs to the prompt. This metric
    takes on float values of either 0.0 (Not Relevant), 0.5 (Partially
    Relevant), or 1.0 (Fully Relevant). The score may also be `None` if it could
    not be computed.

    We currently only support the evaluation based on an EvalClient.
    '''
    generated_outputs, prompts = validate_parameters_answer_relevance(
        generated_outputs, prompts)

    answer_relevance_template = get_template('en/metrics/answer_relevance.j2')

    populated_prompts = [
        answer_relevance_template.render({
            'gen_output': gen_output,
            'user_query': prompt
        }) for gen_output, prompt in zip(generated_outputs, prompts)
    ]

    scores, explanations = eval_model.get_score(metric_name='answer relevance',
                                                language='en',
                                                prompts=populated_prompts,
                                                score_map={
                                                    'Not Relevant': 0.0,
                                                    'Partially Relevant': 0.5,
                                                    'Fully Relevant': 1.0
                                                })

    return MetricValue(metric_name='answer_relevance',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanations,
                       metric_values=scores,
                       language='en')
