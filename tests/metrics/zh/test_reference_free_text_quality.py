import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.metrics.zh import (sentiment, toxicity,
                                  xuyaochen_report_readability)
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs', [
    '我今天很开心',
    ['我今天很开心'],
    ["我今天很开心", "我今天很不开心"],
])
def test_sentiment(generated_outputs):
    metric_value = sentiment(generated_outputs)
    assert 0 <= metric_value <= 1
    if len(metric_value.metric_values) == 2:
        sentiment_score_low = metric_value.metric_values[1]
        assert sentiment_score_low is not None
        assert 0.0 <= sentiment_score_low <= 0.5


@pytest.mark.parametrize('generated_outputs', ["我今天很开心", ["我今天很开心"]])
def test_sentiment_openai(generated_outputs):
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments="{\n  \"sentiment\": \"Positive\"\n}")))
    ]
    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = sentiment(generated_outputs, model_type='openai')
        # "Positive" gets a value of 1.0
        assert metric_value == 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = sentiment(generated_outputs,
                                 model_type='azure_openai',
                                 openai_args={'model': 'foo bar'})
        # "Positive" gets a value of 1.0
        assert metric_value == 1


@pytest.mark.parametrize('generated_outputs',
                         ['我今天生病了。', ['我今天生病了。'], ['我今天生病了。', '你有病啊。']])
def test_toxicity(generated_outputs):
    metric_value = toxicity(generated_outputs)
    toxicity_score_low_risk = metric_value.metric_values[0]
    assert 0 <= toxicity_score_low_risk <= 0.6  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
    if len(metric_value.metric_values) == 2:
        toxicity_score_high_risk = metric_value.metric_values[1]
        assert toxicity_score_high_risk is not None
        assert 0.5 <= toxicity_score_high_risk <= 1


@pytest.mark.parametrize('generated_outputs', ['我今天生病了。', ['我今天生病了。']])
def test_toxicity_openai(generated_outputs):
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments="{\n  \"toxicity\": \"5\"\n}")))
    ]

    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = toxicity(generated_outputs, model_type='openai')
        # "5" gets a value of 1.0
        assert metric_value == 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = toxicity(generated_outputs,
                                model_type='azure_openai',
                                openai_args={'model': 'foo bar'})
        # "5" gets a value of 1.0
        assert metric_value == 1


@pytest.mark.parametrize('generated_outputs,metric_values', [
    ("这一句话很长很难懂，你最好把他改一下。", [11.0]),
    (["今天天气很好。一起去散步吧！"], [2.5]),
])
def test_xuyaochen_report_readability(generated_outputs, metric_values):
    metric_value = xuyaochen_report_readability(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)
