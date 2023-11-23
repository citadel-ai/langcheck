from unittest.mock import Mock, patch

import pytest

from langcheck.metrics.zh import (sentiment)
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