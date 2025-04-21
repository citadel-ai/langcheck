from __future__ import annotations

import os
from typing import Literal
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    AzureOpenAIExtractor,
    OpenAIEvalClient,
    OpenAIExtractor,
)


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_openai(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [Mock(message=Mock(content=answer))]
    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch(
        "openai.resources.chat.Completions.create",
        return_value=mock_chat_completion,
    ):
        # Set the necessary env vars for the OpenAIEValClient
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        client = OpenAIEvalClient(system_prompt=system_prompt)
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("language", ["en", "de", "ja"])
def test_get_float_score_openai(language):
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    class Response(BaseModel):
        score: Literal[tuple(score_map.keys())]  # type: ignore

    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(parsed=Response(score=short_assessment_result)))
    ]
    # Calling the openai.resources.beta.chat.Completions.parse method requires
    # an OpenAI API key, so we mock the return value instead
    with patch(
        "openai.resources.beta.chat.Completions.parse",
        return_value=mock_chat_completion,
    ):
        # Set the necessary env vars for the OpenAIEValClient
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        extractor = OpenAIExtractor()

        scores = extractor.get_float_score(
            "dummy_metric", language, unstructured_assessment_result, score_map
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_azure_openai(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [Mock(message=Mock(content=answer))]
    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch(
        "openai.resources.chat.Completions.create",
        return_value=mock_chat_completion,
    ):
        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_API_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"

        client = AzureOpenAIEvalClient(
            text_model_name="foo bar", system_prompt=system_prompt
        )
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


def test_get_float_score_azure_openai():
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    class Response(BaseModel):
        score: Literal[tuple(score_map.keys())]  # type: ignore

    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(parsed=Response(score=short_assessment_result)))
    ]
    # Calling the openai.resources.beta.chat.Completions.parse method requires
    # an OpenAI API key, so we mock the return value instead
    with patch(
        "openai.resources.beta.chat.Completions.parse",
        return_value=mock_chat_completion,
    ):
        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_API_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        extractor = AzureOpenAIExtractor(text_model_name="foo bar")

        scores = extractor.get_float_score(
            "dummy_metric", "en", unstructured_assessment_result, score_map
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0
