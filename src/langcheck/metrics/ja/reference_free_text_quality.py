from __future__ import annotations

from typing import List, Optional, Tuple

import regex as re

from langcheck.metrics._validation import (validate_parameters_answer_relevance,
                                           validate_parameters_reference_free)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.scorer.hf_models import \
    AutoModelForSequenceClassificationScorer
from langcheck.utils.progess_bar import tqdm_wrapper

from ..prompts._utils import get_template


def sentiment(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        eval_model: str | EvalClient = 'local',
        local_overflow_strategy: str = 'truncate'
) -> MetricValue[Optional[float]]:
    '''Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using an EvalClient, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive). The score may
    also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the Twitter-roBERTa-base-sentiment-multilingual
    model is downloaded from HuggingFace and run locally. This is the default
    model type and there is no setup needed to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Ref:
        https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

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
                       language='ja')


def _sentiment_local(generated_outputs: List[str],
                     overflow_strategy: str) -> List[Optional[float]]:
    '''Calculates the sentiment scores of generated outputs using the
    Twitter-roBERTa-base-sentiment-multilingual model. This metric takes on
    float values between [0, 1], where 0 is negative sentiment and 1 is positive
    sentiment.

    Ref:
        https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        overflow_strategy: The strategy to handle inputs that are longer than
            the maximum input length of the model.

    Returns:
        A list of scores
    '''
    scorer = AutoModelForSequenceClassificationScorer(
        language='ja',
        metric='sentiment',
        # Each class represents a sentiment: 0 is negative, 1 is neutral, and 2
        # is positive
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
    sentiment_template = get_template('ja/metrics/sentiment.j2')

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
        language='ja',
        prompts=populated_prompts,
        score_map=sentiment_assessment_to_score)

    return scores, explanations


def toxicity(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        eval_model: str | EvalClient = 'local',
        local_overflow_strategy: str = 'truncate'
) -> MetricValue[Optional[float]]:
    '''Calculates the toxicity scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low toxicity and 1 is high toxicity.
    (NOTE: when using an EvalClient, the toxicity scores are in steps of
    0.25. The score may also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where a model file is downloaded from HuggingFace and
    run locally. This is the default model type and there is no setup needed to
    run this.
    The model (Alnusjaponica/toxicity-score-multi-classification) is a
    fine-tuned model based on line-corporation/line-distilbert-base-japanese
    model.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/Alnusjaponica/toxicity-score-multi-classification

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
                       language='ja')


def _toxicity_local(generated_outputs: List[str],
                    overflow_strategy: str) -> List[Optional[float]]:
    '''Calculates the toxicity scores of generated outputs using a fine-tuned
    model from `line-corporation/line-distilbert-base-japanese`. This metric
    takes on float values between [0, 1], where 0 is low toxicity and 1 is high
    toxicity.

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/Alnusjaponica/toxicity-score-multi-classification

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        overflow_strategy: The strategy to handle inputs that are longer than
            the maximum input length of the model.

    Returns:
        A list of scores
    '''
    scorer = AutoModelForSequenceClassificationScorer(
        language='ja',
        metric='toxicity',
        # The class 0 is for toxic texts.
        class_weights=[1, 0],
        overflow_strategy=overflow_strategy)
    return scorer.score(generated_outputs)


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
    toxicity_template = get_template('ja/metrics/toxicity.j2')

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
        language='ja',
        prompts=populated_prompts,
        score_map=toxicity_assessment_to_score)

    return scores, explanations


def fluency(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        eval_model: str | EvalClient = 'local',
        local_overflow_strategy: str = 'truncate'
) -> MetricValue[Optional[float]]:
    '''Calculates the fluency scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low fluency and 1 is high fluency.
    (NOTE: when using an EvalClient, the fluency scores are either 0.0
    (poor), 0.5 (fair), or 1.0 (good). The score may also be `None` if it could
    not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where a model file is downloaded from HuggingFace and
    run locally. This is the default model type and there is no setup needed to
    run this.
    The model (liwii/fluency-score-classification-ja) is a fine-tuned model
    based on line-corporation/line-distilbert-base-japanese model.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/liwii/fluency-score-classification-ja

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
                       language='ja')


def _fluency_local(generated_outputs: List[str],
                   overflow_strategy: str) -> List[Optional[float]]:
    '''Calculates the fluency scores of generated outputs using a fine-tuned
    model from `line-corporation/line-distilbert-base-japanese`. This metric
    takes on float values between [0, 1], where 0 is low fluency and 1 is high
    fluency.

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/liwii/fluency-score-classification-ja

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        overflow_strategy: The strategy to handle inputs that are longer than
            the maximum input length of the model.

    Returns:
        A list of scores
    '''
    scorer = AutoModelForSequenceClassificationScorer(
        language='ja',
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
    fluency_template = get_template('ja/metrics/fluency.j2')

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
        language='ja',
        prompts=populated_prompts,
        score_map=fluency_assessment_to_score)

    return scores, explanations


def tateishi_ono_yamada_reading_ease(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None) -> MetricValue[float]:
    '''Calculates the readability of generated Japanese outputs using the
    reading ease score introduced in "日本文の読みやすさの評価式 (A Computer
    Readability Formula of Japanese Texts for Machine Scoring)". This metric
    takes on float values between (-∞, ∞), but in the paper it is reported that
    the average & the standard deviation of the scores obtained for 77 texts
    used for the experiment are 50 and 10 respectively.  Higher scores mean the
    text is easier to read.

    The score is based on the number of "run"s, which are sequences of
    characters with the same type (hiragana, katakana, kanji... etc). See the
    original paper for details.

    Ref:
        https://www.jstage.jst.go.jp/article/nihongokyoiku/158/0/158_49/_pdf/-char/ja (Japanese)
        https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=37773&item_no=1&page_id=13&block_id=8 (Japanese)
        https://aclanthology.org/C88-2135/ (English)

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''  # NOQA: E501
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)

    # Regular expressions used to compute the reading ease score
    blank_re = r'[ |　|\n]'
    hiragana_run_re = r'[\u3041-\u309F]+'
    katakana_run_re = r'[\u30A1-\u30FE]+'
    alphanumeric_run_re = r'[a-zA-Zａ-ｚＡ-Ｚ0-9０-９]+'
    kanji_run_re = r'[\u4E00-\u9FFF]+'
    delimiters_re = r'[、|。|!|？|!|?|「|」|,|，|.|．|…|『|』]'

    # Aux function to compute the average length of strings in the list
    def _mean_str_length(ls: List[str]) -> float:
        if len(ls) == 0:
            return 0
        lens = [len(el) for el in ls]
        return sum(lens) / len(lens)

    def _get_reading_ease(text: str) -> float:
        '''Computes reading ease for each example
        '''
        # Preprocess the text: Delete all blanks
        text = re.sub(blank_re, '', text)

        # Get each term
        hiragana_runs = re.findall(hiragana_run_re, text)
        katakana_runs = re.findall(katakana_run_re, text)
        alphanumeric_runs = re.findall(alphanumeric_run_re, text)
        kanji_runs = re.findall(kanji_run_re, text)
        sentences = re.split(delimiters_re, text)
        period_count = text.count('。')
        if period_count == 0:
            # Just ignore the term
            comma_period_ratio = 0
        else:
            comma_period_ratio = text.count('、') / period_count

        return -0.12 * _mean_str_length(sentences)\
            - 1.37 * _mean_str_length(alphanumeric_runs)\
            + 7.4 * _mean_str_length(hiragana_runs)\
            - 23.18 * _mean_str_length(kanji_runs)\
            - 5.3 * _mean_str_length(katakana_runs)\
            - 4.6 * comma_period_ratio + 115.79

    scores = [
        _get_reading_ease(text) for text in tqdm_wrapper(generated_outputs)
    ]
    return MetricValue(metric_name='tateishi_ono_yamada_reading_ease',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language='ja')


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

    answer_relevance_template = get_template('ja/metrics/answer_relevance.j2')

    populated_prompts = [
        answer_relevance_template.render({
            'gen_output': gen_output,
            'user_query': prompt
        }) for gen_output, prompt in zip(generated_outputs, prompts)
    ]

    scores, explanations = eval_model.get_score(metric_name='answer relevance',
                                                language='ja',
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
                       language='ja')
