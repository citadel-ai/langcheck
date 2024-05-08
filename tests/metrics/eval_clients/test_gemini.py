from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
from google.generativeai.types import generation_types

from langcheck.metrics.eval_clients import GeminiEvalClient


def test_get_text_response_gemini():
    prompts = ['Assess the factual consistency of the generated output...'] * 2
    answer = 'The output is fully factually consistent.'
    mock_response = Mock(spec=generation_types.GenerateContentResponse)
    mock_response.text = answer
    mock_response.candidates = [Mock(finish_reason=1)]
    # Calling the google.generativeai.GenerativeModel.generate_content method
    # requires a Google API key, so we mock the return value instead
    with patch('google.generativeai.GenerativeModel.generate_content',
               return_value=mock_response):

        # Set the necessary env vars for the GeminiEvalClient
        os.environ["GOOGLE_API_KEY"] = "dummy_key"
        client = GeminiEvalClient()
        responses = client.get_text_responses(prompts)
        assert len(responses) == len(prompts)
        for response in responses:
            assert response == answer


@pytest.mark.parametrize('language', ['en', 'de', 'ja'])
def test_get_float_score_gemini(language):
    unstructured_assessment_result: list[str | None] = [
        'The output is fully factually consistent.'
    ] * 2
    short_assessment_result = 'Fully Consistent'
    score_map = {short_assessment_result: 1.0}

    mock_response = Mock(spec=generation_types.GenerateContentResponse)
    mock_response.text = short_assessment_result

    class FunctionCallMock(Mock):

        @classmethod
        def to_dict(cls, instance):
            return {'args': {'assessment': short_assessment_result}}

    mock_response.candidates = [
        Mock(finish_reason=1,
             content=Mock(parts=[Mock(function_call=FunctionCallMock())]))
    ]

    # Calling the google.generativeai.GenerativeModel.generate_content method
    # requires a Google API key, so we mock the return value instead
    with patch('google.generativeai.GenerativeModel.generate_content',
               return_value=mock_response):

        # Set the necessary env vars for the GeminiEvalClient
        os.environ["GOOGLE_API_KEY"] = "dummy_key"
        client = GeminiEvalClient()

        scores = client.get_float_score('dummy_metric', language,
                                        unstructured_assessment_result,
                                        score_map)
        assert len(scores) == len(unstructured_assessment_result)
        for score in scores:
            assert score == 1.0


def test_similarity_scorer_gemini():
    mock_embedding_response = {'embedding': [[0.1, 0.2, 0.3]]}

    # Calling the google.generativeai.embed_content method requires a Google
    # API key, so we mock the return value instead
    with patch('google.generativeai.embed_content',
               Mock(return_value=mock_embedding_response)):
        # Set the necessary env vars for the GeminiEvalClient
        os.environ["GOOGLE_API_KEY"] = "dummy_key"
        gemini_client = GeminiEvalClient()
        scorer = gemini_client.similarity_scorer()
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the similarity score should be 1.
        score = scorer.score(['The cat sat on the mat.'],
                             ['The cat sat on the mat.'])
        assert 0.99 <= score[0] <= 1
