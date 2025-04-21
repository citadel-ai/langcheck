from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.augment.en import rephrase
from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
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
    mock_chat_completion1 = Mock(spec=ChatCompletion)
    mock_chat_completion1.choices = [
        Mock(
            message=Mock(
                content="Illuminate three representative methods for testing LLMs."
            )
        )
    ]
    mock_chat_completion2 = Mock(spec=ChatCompletion)
    mock_chat_completion2.choices = [
        Mock(
            message=Mock(
                content="Identify three typical techniques for testing LLMs."
            )
        )
    ]

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
        actual = rephrase(
            instances,
            num_perturbations=num_perturbations,
            eval_client=openai_client,
        )
        assert actual == expected

    with patch(
        "openai.resources.chat.Completions.create",
        side_effect=side_effect,
    ):
        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_API_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        azure_openai_client = AzureOpenAIEvalClient(
            embedding_model_name="foo bar"
        )
        actual = rephrase(
            instances,
            num_perturbations=num_perturbations,
            eval_client=azure_openai_client,
        )
        assert actual == expected
