from unittest.mock import Mock, patch

import pytest

from langcheck.metrics.zh import sentiment, toxicity
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
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"sentiment\": \"Positive\"\n}"
                },
                'content': 'foo bar'
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        metric_value = sentiment(generated_outputs, model_type='openai')
        # "Positive" gets a value of 1.0
        assert metric_value == 1


@pytest.mark.parametrize('generated_outputs',
                         ['我今天生病了。', ['我今天生病了。'], ['我今天生病了。', '你有病啊。']])
def test_toxicity(generated_outputs):
    metric_value = toxicity(generated_outputs)
    assert 0 <= metric_value <= 1
    if len(metric_value.metric_values) == 2:
        toxicity_score_high_risk = metric_value.metric_values[1]
        assert toxicity_score_high_risk is not None
        assert 0.5 <= toxicity_score_high_risk <= 1


@pytest.mark.parametrize('generated_outputs',
                         ['我今天生病了。', ['我今天生病了。'], ['我今天生病了。', '你有病啊。']])
def test_toxicity_openai(generated_outputs):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"toxicity\": \"5\"\n}"
                },
                'content': 'foo bar'
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        metric_value = toxicity(generated_outputs, model_type='openai')
        # "5" gets a value of 1.0
        assert metric_value == 1
