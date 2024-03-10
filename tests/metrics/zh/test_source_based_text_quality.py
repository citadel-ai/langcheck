import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.metrics.zh import factual_consistency

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs,sources', [
    ('北京是中国的首都。', '中国的首都是北京'),
    pytest.param('地球围绕着太阳转动。', '太阳是太阳系的中心。', marks=pytest.mark.xfail),
    pytest.param(['飞机在是一种空中交通工具。', '太阳围绕着地球转动。'], ['飞机在可以在天上飞。', '太阳是太阳系的中心。']),
])
def test_factual_consistency(generated_outputs, sources):
    metric_value = factual_consistency(generated_outputs, sources)
    factual_consistency_high = metric_value.metric_values[0]
    assert factual_consistency_high is not None
    assert 0.8 <= factual_consistency_high <= 1
    if len(metric_value.metric_values) == 2:
        factual_consistency_low = metric_value.metric_values[1]
        assert factual_consistency_low is not None
        assert 0.0 <= factual_consistency_low <= 0.1


@pytest.mark.parametrize('generated_outputs,sources',
                         [('北京是中国的首都。', '中国的首都是北京'),
                          (['北京是中国的首都。'], ['中国的首都是北京'])])
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
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = factual_consistency(generated_outputs,
                                           sources,
                                           model_type='openai')
        # "Fully Consistent" gets a value of 1.0
        assert metric_value == 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = factual_consistency(generated_outputs,
                                           sources,
                                           model_type='azure_openai',
                                           openai_args={'model': 'foo bar'})
        # "Fully Consistent" gets a value of 1.0
        assert metric_value == 1
