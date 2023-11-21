from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.metrics.ja import factual_consistency

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [('東京は日本の首都です。', '東京は日本の首都です。'),
     (['東京は日本の首都です。', '地球は平面です。'], ['東京は日本の首都です。', '地球は球体です。'])])
def test_factual_consistency(generated_outputs, sources):
    metric_value = factual_consistency(generated_outputs, sources)
    factual_consistency_high = metric_value.metric_values[0]
    assert factual_consistency_high is not None
    assert 0.9 <= factual_consistency_high <= 1
    if len(metric_value.metric_values) == 2:
        factual_consistency_low = metric_value.metric_values[1]
        assert factual_consistency_low is not None
        assert 0.0 <= factual_consistency_low <= 0.1


@pytest.mark.parametrize('generated_outputs,sources',
                         [('東京は日本の首都です。', "東京は日本の首都です。"),
                          (['東京は日本の首都です。'], ["東京は日本の首都です。"])])
def test_factual_consistency_openai(generated_outputs, sources):
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments="{\n  \"factuality\": \"Fully Consistent\"\n}")))
    ]

    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):
        metric_value = factual_consistency(generated_outputs,
                                           sources,
                                           model_type='openai')
        # "Fully Consistent" gets a value of 1.0
        assert metric_value == 1
