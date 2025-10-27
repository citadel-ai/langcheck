from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from langcheck.augment.en import rephrase
from langcheck.metrics.eval_clients import (
    LiteLLMEvalClient,
)


@pytest.mark.parametrize(
    "instances, num_perturbations, expected",
    [
        (
            "List three representative testing methods for LLMs.",
            1,
            ["Illuminate three representative methods for testing LLMs."],
        ),
        (
            ["List three representative testing methods for LLMs."],
            2,
            [
                "Illuminate three representative methods for testing LLMs.",
                "Identify three typical techniques for testing LLMs.",
            ],
        ),
    ],
)
def test_rephrase(
    instances: list[str] | str, num_perturbations: int, expected: list[str]
):
    mock_response_1 = Mock(
        spec=ModelResponse,
        choices=[
            Mock(
                spec=Choices,
                message=Mock(
                    spec=Message,
                    content="Illuminate three representative methods for testing LLMs.",
                ),
            )
        ],
    )
    mock_response_1.choices[
        0
    ].message.content = (
        "Illuminate three representative methods for testing LLMs."
    )
    mock_response_1.usage = Mock(
        spec=Usage, prompt_tokens=10, completion_tokens=15
    )

    mock_response_2 = Mock(
        spec=ModelResponse,
        choices=[
            Mock(
                spec=Choices,
                message=Mock(
                    spec=Message,
                    content="Identify three typical techniques for testing LLMs.",
                ),
            )
        ],
    )
    mock_response_2.choices[
        0
    ].message.content = "Identify three typical techniques for testing LLMs."
    mock_response_2.usage = Mock(
        spec=Usage, prompt_tokens=10, completion_tokens=15
    )

    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch(
        "litellm.completion",
        side_effect=[mock_response_1, mock_response_2],
    ):
        client = LiteLLMEvalClient(
            model="openai/gpt-4o-mini",
            api_key="dummy_key",
        )
        actual = rephrase(
            instances,
            num_perturbations=num_perturbations,
            eval_client=client,
        )
        assert actual == expected
        assert actual.token_usage is not None
