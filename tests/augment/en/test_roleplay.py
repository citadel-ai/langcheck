from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.augment.en import roleplay
from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
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
def test_roleplay(
    instances: list[str] | str,
    system_role: str,
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
        actual = roleplay(
            instances,
            system_role,
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
        actual = roleplay(
            instances,
            system_role,
            num_perturbations=num_perturbations,
            eval_client=azure_openai_client,
        )
        assert actual == expected
