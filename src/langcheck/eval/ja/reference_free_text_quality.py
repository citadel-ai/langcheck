from typing import Dict, List, Optional

import regex as re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langcheck._handle_logs import _handle_logging_level
from langcheck.eval.en.reference_free_text_quality import _toxicity_openai
from langcheck.eval.en.reference_free_text_quality import \
    sentiment as en_sentiment
from langcheck.eval.eval_value import EvalValue

_sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"  # NOQA E501
_sentiment_tokenizer = None
_sentiment_model = None

_toxicity_model_path = "Alnusjaponica/toxicity-score-multi-classification"
_toxicity_tokenizer_path = "line-corporation/line-distilbert-base-japanese"
_toxicity_tokenizer = None
_toxicity_model = None


def sentiment(generated_outputs: List[str],
              prompts: Optional[List[str]] = None,
              model_type: str = 'local',
              openai_args: Optional[Dict[str, str]] = None) -> EvalValue[float]:
    '''Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using the OpenAI model, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive).)

    We currently support two model types:
    1. The 'local' type, where the Twitter-roBERTa-base-sentiment-multilingual
    model is downloaded from HuggingFace and run locally. This is the default
    model type and there is no setup needed to run this.
    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    https://github.com/citadel-ai/langcheck#evaluate-text for examples on
    setting up the OpenAI API key.

    Ref:
        https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: An optional list of prompts used to generate the outputs.
            Prompts are not evaluated and only used as metadata.
        model_type: The type of model to use ('local' or 'openai'),
            default 'local'
        openai_args: Dict of additional args to pass in to the
            `openai.ChatCompletion.create` function, default None

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''

    assert model_type in ['local', 'openai'
                         ], ('Unsupported model type. '
                             'The supported ones are ["local", "openai"]')

    # The English prompt works well enough for Japanese
    # TODO: Investigate the performance improvement with Japanese prompt
    if model_type == 'openai':
        eval_value = en_sentiment(generated_outputs, prompts, model_type,
                                  openai_args)
        eval_value.language = 'ja'
        return eval_value

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

    return EvalValue(metric_name='sentiment',
                     prompts=prompts,
                     generated_outputs=generated_outputs,
                     reference_outputs=None,
                     sources=None,
                     metric_values=scores,
                     language='ja')


def toxicity(generated_outputs: List[str],
             prompts: Optional[List[str]] = None,
             model_type: str = 'local',
             openai_args: Optional[Dict[str, str]] = None) -> EvalValue[float]:
    '''Calculates the toxicity scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low toxicity and 1 is high toxicity.

    We currently support two model types:
    1. The 'local' type, where the our fine-tuned model is downloaded from
    HuggingFace and run locally. This is the default model type and there is
    no setup needed to run this.
    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default, in the same way as english counterpart. While the model you use
    is configurable, please make sure to use one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    https://github.com/citadel-ai/langcheck#evaluate-text for examples on
    setting up the OpenAI API key.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: An optional list of prompts used to generate the outputs.
            Prompts are not evaluated and only used as metadata.
        model_type: The type of model to use ('local' or 'openai'),
            default 'local'
        openai_args: Dict of additional args to pass in to the
            `openai.ChatCompletion.create` function, default None

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''

    assert model_type in ['local', 'openai'
                         ], ('Unsupported model type. '
                             'The supported ones are ["local", "openai"]')

    if model_type == 'local':
        scores = _toxicity_local(generated_outputs)
    else:  # openai
        scores = _toxicity_openai(generated_outputs, openai_args)

    return EvalValue(metric_name='toxicity',
                     prompts=prompts,
                     generated_outputs=generated_outputs,
                     reference_outputs=None,
                     sources=None,
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
    output = _toxicity_model(**input_tokens)
    toxicity_scores = torch.sigmoid(output.logits[:, 0]).tolist()

    return toxicity_scores


def tateishi_ono_yamada_reading_ease(
        generated_outputs: List[str],
        prompts: Optional[List[str]] = None) -> EvalValue[float]:
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
        https://www.jstage.jst.go.jp/article/nihongokyoiku/158/0/158_49/_pdf/-char/ja (Japanese) # NOQA E501
        https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=37773&item_no=1&page_id=13&block_id=8 (Japanese) # NOQA E501
        https://aclanthology.org/C88-2135/ (English)

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: An optional list of prompts used to generate the outputs.
            Prompts are not evaluated and only used as metadata.

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''
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

    scores = [_get_reading_ease(text) for text in generated_outputs]
    return EvalValue(metric_name='tateishi_ono_yamada_reading_ease',
                     prompts=prompts,
                     generated_outputs=generated_outputs,
                     reference_outputs=None,
                     sources=None,
                     metric_values=scores,
                     language='ja')
