from __future__ import annotations

from typing import Literal
from unittest.mock import Mock, patch

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    Message,
    ModelResponse,
)
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import Summary
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
    with patch("litellm.completion", return_value=mock_response):
        client = LiteLLMEvalClient(
            model="dummy_model",
            system_prompt=system_prompt,
            api_key="dummy_key",
        )
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_with_reasoning_summary(system_prompt):
    prompts = ["Prompt A", "Prompt B"]
    answer = "Here is the direct answer."
    reasoning_summary1 = "This is the reasoning summary 1."
    reasoning_summary2 = "This is the reasoning summary 2."
    expected = f"{answer}\n\n**Reasoning Summary:**\n\n{reasoning_summary1}\n\n{reasoning_summary2}"

    # Build a fake response from litellm.responses
    mock_response = Mock(
        spec=ResponsesAPIResponse,
        output=[
            Mock(
                spec=ResponseReasoningItem,
                summary=[
                    Mock(spec=Summary, text=reasoning_summary1),
                    Mock(spec=Summary, text=reasoning_summary2),
                ],
            ),
            Mock(
                spec=ResponseOutputMessage,
                content=[Mock(spec=ResponseOutputText, text=answer)],
            ),
        ],
    )

    # Calling litellm.responses requires a credentials, so we mock the return
    # value instead
    with patch("litellm.responses", return_value=mock_response):
        client = LiteLLMEvalClient(
            model="dummy_model",
            use_reasoning_summary=True,
            system_prompt=system_prompt,
            api_key="dummy_key",
        )
        responses = client.get_text_responses(prompts)
        # We expect each to include the answer, then a blank line, then the **Reasoning Summary:** block
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == expected


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
    with patch("instructor.Instructor.create", return_value=mock_response):
        # Set the necessary env vars for the GeminiEvalClient
        extractor = LiteLLMExtractor(model="dummy_model", api_key="dummy_key")

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
    with patch("litellm.embedding", return_value=mock_response):
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
