from __future__ import annotations

from typing import Literal
from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    Message,
    ModelResponse,
)
from pydantic import BaseModel

from langcheck.metrics.eval_clients import (
    LiteLLMEvalClient,
    LiteLLMExtractor,
)


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_response = Mock(
        spec=ModelResponse,
        choices=[
            Mock(
                spec=Choices,
                message=Mock(spec=Message, content=answer),
            )
        ],
    )
    mock_response.choices[0].message.content = answer
    # Calling litellm.completion requires a credentials, so we mock the return
    # value instead
    with patch(
        "langcheck.metrics.eval_clients._litellm.completion",
        return_value=mock_response,
    ):
        client = LiteLLMEvalClient(
            model="dummy_model",
            system_prompt=system_prompt,
            api_key="dummy_key",
        )
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("language", ["en", "de", "ja"])
def test_get_float_score(language):
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    class Response(BaseModel):
        score: Literal[tuple(score_map.keys())]  # type: ignore

    mock_response = Mock(spec=Response, score=short_assessment_result)
    mock_response.score = short_assessment_result

    # Calling litellm.completion requires a credentials, so we mock the return
    # value instead
    with patch(
        "instructor.Instructor.create",
        return_value=mock_response,
    ):
        # Set the necessary env vars for the GeminiEvalClient
        extractor = LiteLLMExtractor(
            model="openai/dummy_model", api_key="dummy_key"
        )

        scores = extractor.get_float_score(
            "dummy_metric",
            language,
            unstructured_assessment_result,
            score_map,
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0


def test_similarity_scorer():
    mock_embedding_response = [0.1, 0.2, 0.3]

    mock_response = Mock(
        spec=EmbeddingResponse,
        data=[
            {
                "embedding": mock_embedding_response,
            }
        ],
    )

    # Calling the litellm.embedding method requires credentials, so we mock the
    # return value instead
    with patch(
        "langcheck.metrics.eval_clients._litellm.embedding",
        return_value=mock_response,
    ):
        # Set the necessary env vars for the GeminiEvalClient
        client = LiteLLMEvalClient(
            model="dummy_model",
            embedding_model="dummy-embedding-model",
            api_key="dummy_key",
        )
        scorer = client.similarity_scorer()
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the similarity score should be 1.
        score = scorer.score(
            ["The cat sat on the mat."], ["The cat sat on the mat."]
        )
        assert 0.99 <= score[0] <= 1
