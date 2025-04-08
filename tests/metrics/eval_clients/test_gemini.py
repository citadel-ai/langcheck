from __future__ import annotations

import os
from typing import Literal
from unittest.mock import Mock, patch

import pytest
from google.genai import types
from google.oauth2.credentials import Credentials
from pydantic import BaseModel

from langcheck.metrics.eval_clients import GeminiEvalClient


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_gemini(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_response = Mock(spec=types.GenerateContentResponse)
    mock_response.text = answer
    mock_response.candidates = [Mock(finish_reason=1)]
    # Calling the google.genai.models.Models.generate_content method requires a
    # Google API key, so we mock the return value instead
    with patch(
        "google.genai.models.Models.generate_content",
        return_value=mock_response,
    ):
        # Set the necessary env vars for the GeminiEvalClient
        os.environ["GOOGLE_API_KEY"] = "dummy_key"
        client = GeminiEvalClient(system_prompt=system_prompt)
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_gemini_vertex_ai(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_response = Mock(spec=types.GenerateContentResponse)
    mock_response.text = answer
    mock_response.candidates = [Mock(finish_reason=1)]
    # Calling the google.genai.models.Models.generate_content method requires a
    # Google Cloud credentials, so we mock the return value instead
    with patch(
        "google.genai.models.Models.generate_content",
        return_value=mock_response,
    ):
        client = GeminiEvalClient(
            google_cloud_project="dummy_project",
            google_cloud_location="dummy_location",
            google_cloud_credentials=Mock(spec=Credentials),
            system_prompt=system_prompt,
        )
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
@pytest.mark.parametrize("language", ["en", "de", "ja"])
def test_get_float_score_gemini(system_prompt, language):
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    class Response(BaseModel):
        score: Literal[tuple(score_map.keys())]  # type: ignore

    mock_response = Mock(spec=types.GenerateContentResponse)
    mock_response.parsed = Response(score=short_assessment_result)
    mock_response.candidates = [Mock(finish_reason=1)]

    # Calling the google.genai.models.Models.generate_content method requires a
    # Google API key, so we mock the return value instead
    with patch(
        "google.genai.models.Models.generate_content",
        return_value=mock_response,
    ):
        # Set the necessary env vars for the GeminiEvalClient
        os.environ["GOOGLE_API_KEY"] = "dummy_key"
        client = GeminiEvalClient(system_prompt=system_prompt)

        scores = client.get_float_score(
            "dummy_metric", language, unstructured_assessment_result, score_map
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
@pytest.mark.parametrize("language", ["en", "de", "ja"])
def test_get_float_score_gemini_vertex_ai(system_prompt, language):
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    class Response(BaseModel):
        score: Literal[tuple(score_map.keys())]  # type: ignore

    mock_response = Mock(spec=types.GenerateContentResponse)
    mock_response.parsed = Response(score=short_assessment_result)
    mock_response.candidates = [Mock(finish_reason=1)]

    # Calling the google.genai.models.Models.generate_content method requires a
    # Google Cloud credentials, so we mock the return value instead
    with patch(
        "google.genai.models.Models.generate_content",
        return_value=mock_response,
    ):
        client = GeminiEvalClient(
            google_cloud_project="dummy_project",
            google_cloud_location="dummy_location",
            google_cloud_credentials=Mock(spec=Credentials),
            system_prompt=system_prompt,
        )

        scores = client.get_float_score(
            "dummy_metric", language, unstructured_assessment_result, score_map
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0


def test_similarity_scorer_gemini():
    mock_embedding_response = [0.1, 0.2, 0.3]

    # Calling the google.genai.models.Models.embed_content method requires a
    # Google API key, so we mock the return value instead
    with patch(
        "google.genai.models.Models.embed_content",
        Mock(
            return_value=types.EmbedContentResponse(
                embeddings=[
                    types.ContentEmbedding(values=mock_embedding_response)
                ]
            )
        ),
    ):
        # Set the necessary env vars for the GeminiEvalClient
        os.environ["GOOGLE_API_KEY"] = "dummy_key"
        gemini_client = GeminiEvalClient()
        scorer = gemini_client.similarity_scorer()
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the similarity score should be 1.
        score = scorer.score(
            ["The cat sat on the mat."], ["The cat sat on the mat."]
        )
        assert 0.99 <= score[0] <= 1


def test_similarity_scorer_gemini_vertex_ai():
    mock_embedding_response = [0.1, 0.2, 0.3]

    # Calling the google.genai.models.Models.embed_content method requires a
    # Google Cloud credentials, so we mock the return value instead
    with patch(
        "google.genai.models.Models.embed_content",
        Mock(
            return_value=types.EmbedContentResponse(
                embeddings=[
                    types.ContentEmbedding(values=mock_embedding_response)
                ]
            )
        ),
    ):
        gemini_client = GeminiEvalClient(
            google_cloud_project="dummy_project",
            google_cloud_location="dummy_location",
            google_cloud_credentials=Mock(spec=Credentials),
        )
        scorer = gemini_client.similarity_scorer()
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the similarity score should be 1.
        score = scorer.score(
            ["The cat sat on the mat."], ["The cat sat on the mat."]
        )
        assert 0.99 <= score[0] <= 1
