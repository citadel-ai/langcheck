from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from langcheck.augment.ja import rephrase_with_user_role_context
from langcheck.metrics.eval_clients import (
    LiteLLMEvalClient,
)

RESPONSE_1 = "私は学生で、勉強の一環として地理を学んでいます。フランスの首都はどこですか？"
RESPONSE_2 = "私は学生で、フランスの地理について学んでいます。フランスの首都はどこですか？"


@pytest.mark.parametrize(
    "instances, user_role, num_perturbations, expected",
    [
        (
            "フランスの首都はどこですか?",
            "学生",
            1,
            [RESPONSE_1],
        ),
        (
            ["フランスの首都はどこですか?"],
            "学生",
            2,
            [RESPONSE_1, RESPONSE_2],
        ),
    ],
)
def test_rephrase_with_user_role_context(
    instances: list[str] | str,
    user_role: str,
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

    # Calling the litellm.completion method requires a credentials, so we mock the return value instead
    with patch(
        "litellm.completion",
        side_effect=[mock_response_1, mock_response_2],
    ):
        # Set the necessary env vars for the 'openai' model type
        client = LiteLLMEvalClient(
            model="openai/gpt-4o-mini",
            api_key="dummy_key",
        )
        actual = rephrase_with_user_role_context(
            instances,
            user_role,
            num_perturbations=num_perturbations,
            eval_client=client,
        )
        assert actual == expected
        assert actual.token_usage is not None
