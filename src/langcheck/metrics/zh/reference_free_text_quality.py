from __future__ import annotations

from typing import Dict, List, Optional

import regex as re
from sklearn.conftest import dataset_fetchers
import torch
from transformers import pipeline

from langcheck._handle_logs import _handle_logging_level
from langcheck.metrics._validation import validate_parameters_reference_free
from langcheck.metrics.en.reference_free_text_quality import (_fluency_openai,
                                                              _toxicity_openai)
from langcheck.metrics.en.reference_free_text_quality import \
    sentiment as en_sentiment
from langcheck.metrics.metric_value import MetricValue

_sentiment_model_path = 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'  # NOQA: E501
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
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    '''Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using the OpenAI model, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive). The score may
    also be `None` if it could not be computed.)

    We currently support two model types:
    1. The 'local' type, where the IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
    model is downloaded from HuggingFace and run locally. This is the default
    model type and there is no setup needed to run this.
    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this example <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    Ref:
        https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        model_type: The type of model to use ('local' or 'openai'),
            default 'local'
        openai_args: Dict of additional args to pass in to the
            `openai.ChatCompletion.create` function, default None

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)
    assert model_type in ['local', 'openai'
                         ], ('Unsupported model type. '
                             'The supported ones are ["local", "openai"]')

    if model_type == 'openai':
        metric_value = en_sentiment(generated_outputs, prompts, model_type,
                                    openai_args)
        metric_value.language = 'ja'
        return metric_value

    global _sentiment_model_path, _sentiment_tokenizer, _sentiment_model

    _sentiment_pipeline = pipeline('sentiment-analysis', model=_sentiment_model_path)  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
    # {0:"Negative", 1:'Positive'}
    _model_id2label = _sentiment_pipeline.model.config.id2label
    _predict_result = _sentiment_pipeline(generated_outputs)  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501

    scores = [
        1 - x['score'] if x['label'] == _model_id2label[0]  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
        else
        x['score'] for x in _predict_result  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
              ]

    return MetricValue(metric_name='sentiment',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=None,
                       metric_values=scores,  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
                       language='zh')
