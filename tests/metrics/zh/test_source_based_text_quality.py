from unittest.mock import Mock, patch
import pytest

from langcheck.metrics.zh import factual_consistency

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [('北京是中国的首都。', '中国的首都是北京'),
     pytest.param('地球围绕着太阳转动。', '太阳是太阳系的中心。', marks=pytest.mark.xfail),
     pytest.param(['飞机是一种水中交通工具。', '太阳围绕着地球转动。'],
                  ['飞机在可以在天上飞。', '太阳是太阳系的中心。']),])
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
                         [('北京是中国的首都。', '中国的首都是北京'),
                          (['北京是中国的首都。'], ['中国的首都是北京'])])
def test_factual_consistency_openai(generated_outputs, sources):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"factuality\": \"Fully Consistent\"\n}"
                },
                'content': 'foo bar'
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        metric_value = factual_consistency(generated_outputs,
                                           sources,
                                           model_type='openai')
        # "Fully Consistent" gets a value of 1.0
        assert metric_value == 1
