from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from langcheck.augment.en import rephrase


@pytest.mark.parametrize(
    "instances, expected",
    [
        ("List three representative testing methods for LLMs.",
         ["Identify three typical methods used for evaluating LLMs."]),
        (["List three representative testing methods for LLMs."
         ], ["Identify three typical methods used for evaluating LLMs."]),
    ],
)
def test_rephrase(instances: list[str] | str, expected: list[str]):
    mock_chat_response = {
        'choices': [{
            'message': {
                'content':
                    'Identify three typical methods used for evaluating LLMs.'
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        actual = rephrase(instances)
        assert actual == expected
