from __future__ import annotations

from typing import Dict, List, Optional

import regex as re
import torch
from openai import OpenAI
from transformers.models.auto.modeling_auto import \
    AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from langcheck._handle_logs import _handle_logging_level
from langcheck.metrics._validation import (validate_parameters_answer_relevance,
                                           validate_parameters_reference_free)
from langcheck.metrics.en._openai import OpenAIBasedEvaluator
from langcheck.metrics.en.reference_free_text_quality import (_fluency_openai,
                                                              _toxicity_openai)
from langcheck.metrics.en.reference_free_text_quality import \
    sentiment as en_sentiment
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper

_sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"  # NOQA: E501
_sentiment_tokenizer = None
_sentiment_model = None

_toxicity_model_path = "Alnusjaponica/toxicity-score-multi-classification"
_toxicity_tokenizer_path = "line-corporation/line-distilbert-base-japanese"
_toxicity_tokenizer = None
_toxicity_model = None

_fluency_model_path = "liwii/fluency-score-classification-ja"
_fluency_tokenizer_path = "line-corporation/line-distilbert-base-japanese"
_fluency_tokenizer = None
_fluency_model = None


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

    1. The 'local' type, where the Twitter-roBERTa-base-sentiment-multilingual
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
        https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

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

    # The English prompt works well enough for Japanese
    # TODO: Investigate the performance improvement with Japanese prompt
    if model_type == 'openai' or model_type == 'azure_openai':
        metric_value = en_sentiment(generated_outputs, prompts, model_type,
                                    openai_client, openai_args)
        metric_value.language = 'ja'
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
                       language='ja')


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

    1. The 'local' type, where a model file is downloaded from HuggingFace and
    run locally. This is the default model type and there is no setup needed to
    run this.
    The model (Alnusjaponica/toxicity-score-multi-classification) is a
    fine-tuned model based on line-corporation/line-distilbert-base-japanese
    model.

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

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/Alnusjaponica/toxicity-score-multi-classification

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
                       language='ja')


def _toxicity_local(generated_outputs: List[str]) -> List[float]:
    '''Calculates the toxicity scores of generated outputs using a fine-tuned
    model from `line-corporation/line-distilbert-base-japanese`. This metric
    takes on float values between [0, 1], where 0 is low toxicity and 1 is high
    toxicity.

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/Alnusjaponica/toxicity-score-multi-classification

    Args:
        generated_outputs: A list of model generated outputs to evaluate

    Returns:
        A list of scores
    '''
    global _toxicity_model, _toxicity_tokenizer
    _toxicity_model = (
        _toxicity_model or
        AutoModelForSequenceClassification.from_pretrained(_toxicity_model_path)
    )
    _toxicity_tokenizer = _toxicity_tokenizer or AutoTokenizer.from_pretrained(
        _toxicity_tokenizer_path, trust_remote_code=True)

    input_tokens = _toxicity_tokenizer(generated_outputs,
                                       return_tensors='pt',
                                       padding=True)
    batchsize = 8
    toxicity_scores = []
    for i in tqdm_wrapper(range(0, len(generated_outputs), batchsize),
                          total=(len(generated_outputs) + batchsize - 1) //
                          batchsize):
        with torch.no_grad():
            batch_input_tokens = {
                k: v[i:i + batchsize] for k, v in input_tokens.items()
            }
            batch_output = _toxicity_model(**batch_input_tokens)
            toxicity_scores.extend(
                torch.sigmoid(batch_output.logits[:, 0]).tolist())

    return toxicity_scores


def fluency(
    generated_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    model_type: str = 'local',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    '''Calculates the fluency scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low fluency and 1 is high fluency.
    (NOTE: when using the OpenAI model, the fluency scores are either 0.0
    (poor), 0.5 (fair), or 1.0 (good). The score may also be `None` if it could
    not be computed.)

    We currently support three model types:

    1. The 'local' type, where a model file is downloaded from HuggingFace and
    run locally. This is the default model type and there is no setup needed to
    run this.
    The model (liwii/fluency-score-classification-ja) is a fine-tuned model
    based on line-corporation/line-distilbert-base-japanese model.

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

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/liwii/fluency-score-classification-ja

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
        scores = _fluency_local(generated_outputs)
        explanations = None
    else:  # openai or azure_openai
        scores, explanations = _fluency_openai(generated_outputs, model_type,
                                               openai_client, openai_args)

    return MetricValue(metric_name='fluency',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanations,
                       metric_values=scores,
                       language='ja')


def _fluency_local(generated_outputs: List[str]) -> List[float]:
    '''Calculates the fluency scores of generated outputs using a fine-tuned
    model from `line-corporation/line-distilbert-base-japanese`. This metric
    takes on float values between [0, 1], where 0 is low fluency and 1 is high
    fluency.

    Ref:
        https://huggingface.co/line-corporation/line-distilbert-base-japanese
        https://huggingface.co/liwii/fluency-score-classification-ja

    Args:
        generated_outputs: A list of model generated outputs to evaluate

    Returns:
        A list of scores
    '''
    global _fluency_model, _fluency_tokenizer
    _fluency_model = (
        _fluency_model or
        AutoModelForSequenceClassification.from_pretrained(_fluency_model_path))

    # Suppress "tokenizer class you load ... is not the same type ..." error
    # because AutoTokenzier is suggested by the readme of the original model
    # and "DistilBertJapaneseTokenizer", which is suggested by the warning,
    # is not exposed to transformers.
    with _handle_logging_level():
        _fluency_tokenizer = (_fluency_tokenizer or
                              AutoTokenizer.from_pretrained(
                                  _fluency_tokenizer_path,
                                  trust_remote_code=True,
                                  revision='main'))

    input_tokens = _fluency_tokenizer(generated_outputs,
                                      return_tensors='pt',
                                      padding=True)
    batchsize = 8
    fluency_scores = []
    with torch.no_grad():
        for i in tqdm_wrapper(range(0, len(generated_outputs), batchsize)):
            batch_input_tokens = {
                k: v[i:i + batchsize] for k, v in input_tokens.items()
            }
            batch_probs = torch.nn.functional.softmax(
                _fluency_model(**batch_input_tokens).logits, dim=1)
            fluency_scores.extend(batch_probs[:, 1].tolist())

    return fluency_scores


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


def answer_relevance(
    generated_outputs: List[str] | str,
    prompts: List[str] | str,
    model_type: str = 'openai',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    '''Calculates the relevance of generated outputs to the prompt. This metric
    takes on float values of either 0.0 (Not Relevant), 0.5 (Partially
    Relevant), or 1.0 (Fully Relevant). The score may also be `None` if it could
    not be computed.

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
    '''
    generated_outputs, prompts = validate_parameters_answer_relevance(
        generated_outputs, prompts)
    assert model_type in [
        'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["openai", "azure_openai"]')

    def _prompt(gen_output: str, user_query: str) -> str:
        return f'''
        ユーザーの質問に対する回答の関連性を評価してください。データは以下の通りです:
        [BEGIN DATA]
        ************
        [ユーザーの質問]: {user_query}
        ************
        [回答]: {gen_output}
        ************
        [END DATA]

        ユーザーの質問に対して回答が関連性のあるものかどうかを判断してください。利用可能な評価
        は以下の通りです:
        `完全に関連` - 回答はユーザーの質問に完全に関連し、十分に答えています。
        `部分的に関連` - 回答はユーザーの質問に部分的に関連していますが、質問に完全に答えて
        いないか、関連しない情報が含まれています。
        `関連なし` - 回答はユーザーの質問に関連していない、または質問に適切に対応していません。

        深呼吸をして、この問題をステップバイステップで取り組んでください。
        '''

    def _function_call_prompt(long_assessment: str) -> str:
        return f'''
        以下はユーザーの質問に対する回答の関連性に関する評価です:
        ************
        [評価]: {long_assessment}
        ************

        結果として出た評価を保存してください。利用可能な評価は以下の通りです:
        `完全に関連`
        `部分的に関連`
        `関連なし`
        '''

    answer_relevance_assessment_to_score = {
        '完全に関連': 1.0,
        '部分的に関連': 0.5,
        '関連なし': 0.0
    }
    oai_evaluator = OpenAIBasedEvaluator(
        assessment_to_score_mapping=answer_relevance_assessment_to_score,
        function_name='save_answer_relevance_assessment',
        function_description=("Saves an answer relevance assessment."),
        argument_name='answer_relevance',
        argument_description='The answer relevance assessment',
        client_type=model_type,
        client=openai_client,
        openai_args=openai_args)

    score_list = []
    explanation_list = []
    for gen, user_query in tqdm_wrapper(zip(generated_outputs, prompts),
                                        desc='Calculating scores',
                                        total=len(prompts)):
        score, explanation = oai_evaluator.get_score(_prompt(gen, user_query),
                                                     _function_call_prompt)
        score_list.append(score)
        explanation_list.append(explanation)

    return MetricValue(metric_name='answer_relevance',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanation_list,
                       metric_values=score_list,
                       language='ja')
