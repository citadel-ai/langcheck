from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from langcheck.augment.en import rephrase_with_system_role_context
from langcheck.metrics.eval_clients import (
    LiteLLMEvalClient,
)

RESPONSE_1 = "You're a teacher, and it's your duty to provide accurate information and assist learners in expanding their general knowledge.\nNow answer the query: What is the capital of France?"
RESPONSE_2 = "You're a teacher, and you need to provide accurate information to educate your students effectively.\nNow answer the query: What is the capital of France?"


@pytest.mark.parametrize(
    "instances, system_role, num_perturbations, expected",
    [
        (
            "What is the capital of France?",
            "teacher",
            1,
            [RESPONSE_1],
        ),
        (
            ["What is the capital of France?"],
            "teacher",
            2,
            [RESPONSE_1, RESPONSE_2],
        ),
    ],
)
def test_rephrase_with_system_role_context(
    instances: list[str] | str,
    system_role: str,
    num_perturbations: int,
    expected: list[str],
):
    mock_response_1 = Mock(
        spec=ModelResponse,
        choices=[
            Mock(
                spec=Choices,
                message=Mock(spec=Message, content=RESPONSE_1),
            )
        ],
    )
    mock_response_1.choices[0].message.content = RESPONSE_1
    mock_response_1.usage = Mock(
        spec=Usage, prompt_tokens=10, completion_tokens=15
    )

    mock_response_2 = Mock(
        spec=ModelResponse,
        choices=[
            Mock(
                spec=Choices,
                message=Mock(spec=Message, content=RESPONSE_2),
            )
        ],
    )
    mock_response_2.choices[0].message.content = RESPONSE_2
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
        actual = rephrase_with_system_role_context(
            instances,
            system_role,
            num_perturbations=num_perturbations,
            eval_client=client,
        )
        assert actual == expected
        assert actual.token_usage is not None
