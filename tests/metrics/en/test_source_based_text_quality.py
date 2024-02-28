import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.metrics.en import context_relevance, factual_consistency

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [('Tokyo is the capital of Japan.', "Tokyo is Japan's capital city."),
     (['Tokyo is the capital of Japan.', 'The Earth is flat.'
      ], ["Tokyo is Japan's capital city.", 'The Earth is round.'])])
def test_factual_consistency(generated_outputs, sources):
    metric_value = factual_consistency(generated_outputs,
                                       sources,
                                       model_type='local')
    factual_consistency_high = metric_value.metric_values[0]
    assert factual_consistency_high is not None
    assert 0.9 <= factual_consistency_high <= 1
    if len(generated_outputs) == 2:
        factual_consistency_low = metric_value.metric_values[1]
        assert factual_consistency_low is not None
        assert 0.0 <= factual_consistency_low <= 0.1


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [('Tokyo is the capital of Japan.', "Tokyo is Japan's capital city."),
     (['Tokyo is the capital of Japan.'], ["Tokyo is Japan's capital city."])])
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


@pytest.mark.parametrize(
    'prompts,sources',
    [('What is the capital of Japan?', "Tokyo is Japan's capital city."),
     (['What is the capital of Japan?'], ["Tokyo is Japan's capital city."])])
def test_context_relevance_openai(prompts, sources):
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments="{\n  \"context_relevance\": \"Fully Relevant\"\n}")))
    ]

    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = context_relevance(prompts, sources, model_type='openai')
        # "Fully Relevant" gets a value of 1.0
        assert metric_value == 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = context_relevance(prompts,
                                         sources,
                                         model_type='azure_openai',
                                         openai_args={'model': 'foo bar'})
        # "Fully Relevant" gets a value of 1.0
        assert metric_value == 1
