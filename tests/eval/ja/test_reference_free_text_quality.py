from unittest.mock import Mock, patch

import pytest

from langcheck.eval.ja import sentiment, toxicity

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs', [["私は嬉しい", "私は悲しい"], ['こんにちは']])
def test_sentiment(generated_outputs):
    eval_value = sentiment(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize('generated_outputs', ["私は嬉しい"])
def test_sentiment_openai(generated_outputs):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"sentiment\": \"Positive\"\n}"
                }
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        eval_value = sentiment(generated_outputs, model_type='openai')
        # "Positive" gets a value of 1.0
        assert eval_value.metric_values[0] == 1


@pytest.mark.parametrize('generated_outputs', [['馬鹿', '今日はりんごを食べました。'], ['猫']])
def test_toxicity(generated_outputs):
    eval_value = toxicity(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize('generated_outputs', ['アホ'])
def test_toxicity_openai(generated_outputs):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"toxicity\": \"5\"\n}"
                }
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        eval_value = toxicity(generated_outputs, model_type='openai')
        # "5" gets a value of 1.0
        assert eval_value.metric_values[0] == 1