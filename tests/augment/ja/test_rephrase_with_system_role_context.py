from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import ModelResponse, Usage

from langcheck.augment.ja import rephrase_with_system_role_context
from langcheck.metrics.eval_clients import LiteLLMEvalClient

RESPONSE_1 = "あなたは先生で、学生が地理の勉強をする上で必要な知識を学べるようサポートしなければなりません。\nでは以下のクエリに応えてください: フランスの首都はどこですか？"
RESPONSE_2 = "あなたは先生で、学生に地理の知識を教える責任があります。\nでは以下のクエリに応えてください: フランスの首都はどこですか？"


@pytest.mark.parametrize(
    "instances, system_role, num_perturbations, expected",
    [
        (
            "フランスの首都はどこですか?",
            "先生",
            1,
            [RESPONSE_1],
        ),
        (
            ["フランスの首都はどこですか?"],
            "先生",
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
    mock_response_1 = Mock(spec=ModelResponse)
    mock_response_1.choices = [Mock(message=Mock(content=RESPONSE_1))]
    mock_response_1.usage = Mock(
        spec=Usage, prompt_tokens=10, completion_tokens=15
    )
    mock_response_2 = Mock(spec=ModelResponse)
    mock_response_2.choices = [Mock(message=Mock(content=RESPONSE_2))]
    mock_response_2.usage = Mock(
        spec=Usage, prompt_tokens=10, completion_tokens=15
    )
    # Calling the litellm.completion method requires a credentials, so we mock the return value instead
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
