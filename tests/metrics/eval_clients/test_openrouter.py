from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from langcheck.metrics.eval_clients import OpenRouterEvalClient


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
def test_get_text_response_openrouter(system_prompt):
    prompts = ["Assess the factual consistency of the generated output..."] * 2
    answer = "The output is fully factually consistent."
    mock_response = [{"choices": [{"message": {"content": answer}}]}] * 2
    # Calling the _call_api method requires an
    # OpenRouter API key, so we mock the return value instead
    with patch(
        "langcheck.metrics.eval_clients.OpenRouterEvalClient._call_api",
        return_value=mock_response,
    ):
        # Set the necessary env vars for the OpenRouterEvalClient
        os.environ["OPENROUTER_API_KEY"] = "dummy_key"
        client = OpenRouterEvalClient(system_prompt=system_prompt)
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize("system_prompt", [None, "Answer in English."])
@pytest.mark.parametrize("language", ["en", "ja"])
def test_get_float_score_openrouter(system_prompt, language):
    unstructured_assessment_result: list[str | None] = [
        "The output is fully factually consistent."
    ] * 2
    short_assessment_result = "Fully Consistent"
    score_map = {short_assessment_result: 1.0}

    mock_response = [
        {"choices": [{"message": {"content": short_assessment_result}}]}
    ] * 2

    with patch(
        "langcheck.metrics.eval_clients.OpenRouterEvalClient._call_api",
        return_value=mock_response,
    ):
        # Set the necessary env vars for the OpenRouterEvalClient
        os.environ["OPENROUTER_API_KEY"] = "dummy_key"
        client = OpenRouterEvalClient(system_prompt=system_prompt)

        scores = client.get_float_score(
            "dummy_metric", language, unstructured_assessment_result, score_map
        )
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0
