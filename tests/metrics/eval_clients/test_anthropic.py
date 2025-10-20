from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
from anthropic.types.message import Message

from langcheck.metrics.eval_clients import (
    AnthropicEvalClient,
    AnthropicExtractor,
)


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_anthropic(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_chat_completion = Mock(spec=Message)
    mock_chat_completion.content = [Mock(text=answer)]
    # Calling the anthropic.resources.Messages.create method requires an
    # Anthropic API key, so we mock the return value instead
    with patch(
        "anthropic.resources.Messages.create", return_value=mock_chat_completion
    ):
        # Set the necessary env vars for the AnthropicEvalClient
        os.environ["ANTHROPIC_API_KEY"] = "dummy_key"
        client = AnthropicEvalClient(system_prompt=system_prompt)
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_anthropic_vertex_ai(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_chat_completion = Mock(spec=Message)
    mock_chat_completion.content = [Mock(text=answer)]
    # Calling the anthropic.resources.Messages.create method requires a Google
    # Cloud credentials, so we mock the return value instead
    with patch(
        "anthropic.resources.Messages.create", return_value=mock_chat_completion
    ):
        # Set the necessary env vars for the Vertex AI AnthropicEvalClient
        os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] = "dummy_project"
        os.environ["CLOUD_ML_REGION"] = "dummy_location"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dummy_credentials_path"
        client = AnthropicEvalClient(vertexai=True, system_prompt=system_prompt)
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("language", ["en", "de", "ja"])
def test_get_float_score_anthropic(language):
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    mock_chat_completion = Mock(spec=Message)
    mock_chat_completion.content = [Mock(text=short_assessment_result)]

    # Calling the anthropic.resources.Messages.create method requires an
    # Anthropic API key, so we mock the return value instead
    with patch(
        "anthropic.resources.Messages.create", return_value=mock_chat_completion
    ):
        # Set the necessary env vars for the AnthropicEvalClient
        os.environ["ANTHROPIC_API_KEY"] = "dummy_key"
        extractor = AnthropicExtractor()
        scores = extractor.get_float_score(
            "dummy_metric", language, unstructured_assessment_result, score_map
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0


@pytest.mark.parametrize("language", ["en", "de", "ja"])
def test_get_float_score_anthropic_vertex_ai(language):
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    mock_chat_completion = Mock(spec=Message)
    mock_chat_completion.content = [Mock(text=short_assessment_result)]

    # Calling the anthropic.resources.Messages.create method requires a Google
    # Cloud credentials, so we mock the return value instead
    with patch(
        "anthropic.resources.Messages.create", return_value=mock_chat_completion
    ):
        # Set the necessary env vars for the Vertex AI AnthropicEvalClient
        os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] = "dummy_project"
        os.environ["CLOUD_ML_REGION"] = "dummy_location"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dummy_credentials_path"
        extractor = AnthropicExtractor(vertexai=True)

        scores = extractor.get_float_score(
            "dummy_metric", language, unstructured_assessment_result, score_map
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0
