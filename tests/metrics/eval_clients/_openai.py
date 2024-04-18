from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion

from langcheck.metrics.eval_clients import (AzureOpenAIEvalClient,
                                            OpenAIEvalClient)


def test_get_text_response_openai():
    prompts = ['What is the capital of France?'] * 2
    answer = 'Paris is the capital of France.'
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [Mock(message=Mock(content=answer))]
    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):

        # Set the necessary env vars for the OpenAIEValClient
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        client = OpenAIEvalClient()
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


def test_get_float_score_openai():
    unstructured_assessment_result: list[str | None] = [
        'Paris is the capital of France.'
    ] * 2
    short_assessment_result = 'Paris'
    score_map = {short_assessment_result: 1.0}

    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments=json.dumps({'assessment': short_assessment_result}))))
    ]
    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):

        # Set the necessary env vars for the OpenAIEValClient
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        client = OpenAIEvalClient()

        scores = client.get_float_score('dummy_metric', 'en',
                                        unstructured_assessment_result,
                                        score_map)
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0


def test_get_text_response_azure_openai():
    prompts = ['What is the capital of France?'] * 2
    answer = 'Paris is the capital of France.'
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [Mock(message=Mock(content=answer))]
    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"

        client = AzureOpenAIEvalClient(text_model_name='foo bar')
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


def test_get_float_score_azure_openai():
    unstructured_assessment_result: list[str | None] = [
        'Paris is the capital of France.'
    ] * 2
    short_assessment_result = 'Paris'
    score_map = {short_assessment_result: 1.0}

    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments=json.dumps({'assessment': short_assessment_result}))))
    ]
    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        client = AzureOpenAIEvalClient(text_model_name='foo bar')

        scores = client.get_float_score('dummy_metric', 'en',
                                        unstructured_assessment_result,
                                        score_map)
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0
