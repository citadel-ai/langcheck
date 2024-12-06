from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from langcheck.augment.ja import rephrase_with_system_role_context
from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
)

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
        actual = rephrase_with_system_role_context(
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
        actual = rephrase_with_system_role_context(
            instances,
            system_role,
            num_perturbations=num_perturbations,
            eval_client=azure_openai_client,
        )
        assert actual == expected
