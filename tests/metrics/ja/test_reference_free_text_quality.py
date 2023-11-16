from unittest.mock import Mock, patch

import pytest

from langcheck.metrics.ja import (fluency, sentiment,
                                  tateishi_ono_yamada_reading_ease, toxicity)
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs', [
    'こんにちは',
    ['こんにちは'],
    ["私は嬉しい", "私は悲しい"],
])
def test_sentiment(generated_outputs):
    metric_value = sentiment(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs', ["私は嬉しい", ["私は嬉しい"]])
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
                         [['馬鹿', '今日はりんごを食べました。'], ['猫'], '猫'])
def test_toxicity(generated_outputs):
    metric_value = toxicity(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs', ['アホ', ['アホ']])
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


@pytest.mark.parametrize(
    'generated_outputs',
    [['ご機嫌いかがですか？私はとても元気です。', '機嫌いかが？私はとても元気人です。'], ['猫'], '猫'])
def test_fluency(generated_outputs):
    metric_value = fluency(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs',
                         ['ご機嫌いかがですか？私はとても元気です。', ['ご機嫌いかがですか？私はとても元気です。']])
def test_fluency_openai(generated_outputs):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"fluency\": \"Good\"\n}"
                },
                'content': 'foo bar'
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        metric_value = fluency(generated_outputs, model_type='openai')
        # "Good" gets a value of 1.0
        assert metric_value == 1


@pytest.mark.parametrize('generated_outputs,metric_values', [
    ('吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。',
     [73.499359]),
    (['吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。'
     ], [73.499359]),
    (['日本語自然言語処理には、日本語独特の技法が多数必要で、欧米系言語と比較して難易度が高い。'], [24.7875]),
])
def test_tateishi_ono_yamada_reading_ease(generated_outputs, metric_values):
    metric_value = tateishi_ono_yamada_reading_ease(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)
