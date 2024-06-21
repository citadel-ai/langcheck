from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
from anthropic.types.message import Message
from langcheck.metrics.eval_clients import AnthropicEvalClient


def test_get_text_response_anthropic():
    prompts = ['Assess the factual consistency of the generated output...'] * 2
    answer = 'The output is fully factually consistent.'
    mock_chat_completion = Mock(spec=Message)
    mock_chat_completion.content = [Mock(text=answer)]
    # Calling the anthropic.resources.Messages.create method requires an
    # Anthropic API key, so we mock the return value instead
    with patch('anthropic.resources.Messages.create',
               return_value=mock_chat_completion):

        # Set the necessary env vars for the AnthropicEvalClient
        os.environ["ANTHROPIC_API_KEY"] = "dummy_key"
        client = AnthropicEvalClient()
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize('language', ['en', 'de', 'ja'])
def test_get_float_score_anthropic(language):
    unstructured_assessment_result: list[str | None] = [
        'The output is fully factually consistent.'
    ] * 2
    short_assessment_result = 'Fully Consistent'
    score_map = {short_assessment_result: 1.0}

    mock_chat_completion = Mock(spec=Message)
    mock_chat_completion.content = [Mock(text=short_assessment_result)]

    # Calling the anthropic.resources.Messages.create method requires an
    # Anthropic API key, so we mock the return value instead
    with patch('anthropic.resources.Messages.create',
               return_value=mock_chat_completion):

        # Set the necessary env vars for the AnthropicEvalClient
        os.environ["ANTHROPIC_API_KEY"] = "dummy_key"
        client = AnthropicEvalClient()

        scores = client.get_float_score('dummy_metric', language,
                                        unstructured_assessment_result,
                                        score_map)
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0
