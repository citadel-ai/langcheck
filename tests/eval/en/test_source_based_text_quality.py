from unittest.mock import Mock, patch

import pytest

from langcheck.eval.en import factual_consistency

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [(['Tokyo is the capital of Japan.', 'The Earth is flat.'
      ], ["Tokyo is Japan's capital city.", 'The Earth is round.'])])
def test_factual_consistency(generated_outputs, sources):
    eval_value = factual_consistency(generated_outputs,
                                     sources,
                                     model_type='local')
    factual_consistency_high = eval_value.metric_values[0]
    assert 0.9 <= factual_consistency_high <= 1
    factual_consistency_low = eval_value.metric_values[1]
    assert 0.0 <= factual_consistency_low <= 0.1


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [(['Tokyo is the capital of Japan.'], ["Tokyo is Japan's capital city."])])
def test_factual_consistency_openai(generated_outputs, sources):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': {
                        'factuality': 1.0
                    }
                }
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        eval_value = factual_consistency(generated_outputs,
                                         sources,
                                         model_type='openai')
        factual_consistency_high = eval_value.metric_values[0]
        assert factual_consistency_high == 1
