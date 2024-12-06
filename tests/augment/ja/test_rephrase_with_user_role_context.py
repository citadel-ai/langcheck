from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.augment.ja import rephrase_with_user_role_context
from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
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
    mock_chat_completion1 = Mock(spec=ChatCompletion)
    mock_chat_completion1.choices = [Mock(message=Mock(content=RESPONSE_1))]
    mock_chat_completion2 = Mock(spec=ChatCompletion)
    mock_chat_completion2.choices = [Mock(message=Mock(content=RESPONSE_2))]

    side_effect = [mock_chat_completion1, mock_chat_completion2]

    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch(
        "openai.resources.chat.Completions.create",
        side_effect=side_effect,
    ):
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        openai_client = OpenAIEvalClient()
        actual = rephrase_with_user_role_context(
            instances,
            user_role,
            num_perturbations=num_perturbations,
            eval_client=openai_client,
        )
        assert actual == expected

    with patch(
        "openai.resources.chat.Completions.create",
        side_effect=side_effect,
    ):
        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        azure_openai_client = AzureOpenAIEvalClient(
            embedding_model_name="foo bar"
        )
        actual = rephrase_with_user_role_context(
            instances,
            user_role,
            num_perturbations=num_perturbations,
            eval_client=azure_openai_client,
        )
        assert actual == expected
