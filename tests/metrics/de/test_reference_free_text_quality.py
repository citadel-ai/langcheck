import os
from unittest.mock import Mock, patch

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from langcheck.metrics.de import (ai_disclaimer_similarity,
                                  flesch_kincaid_grade, flesch_reading_ease,
                                  fluency, sentiment, toxicity)
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs', [
    'Hallo', ['Hallo'],
    ["Mir geht es gut!", "Mir geht es heute ziemlich schlecht."]
])
def test_sentiment(generated_outputs):
    metric_value = sentiment(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs',
                         ["Mir geht es gut!", ["Mir geht es gut!"]])
def test_sentiment_openai(generated_outputs):
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments="{\n  \"sentiment\": \"Positive\"\n}")))
    ]

    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = sentiment(generated_outputs, model_type='openai')
        # "Positive" gets a value of 1.0
        assert metric_value == 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = sentiment(generated_outputs,
                                 model_type='azure_openai',
                                 openai_args={'model': 'foo bar'})
        # "Positive" gets a value of 1.0
        assert metric_value == 1


@pytest.mark.parametrize('generated_outputs', [
    'Katze', ['Katze'],
    [
        "Ich würde deine Hilfe schätzen.",
        'Heute essen ich sehr viel gute Äpfel.'
    ]
])
def test_fluency(generated_outputs):
    metric_value = fluency(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize(
    'generated_outputs',
    ["Ich würde deine Hilfe schätzen.", ["Ich würde deine Hilfe schätzen."]])
def test_fluency_openai(generated_outputs):
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments="{\n  \"fluency\": \"Good\"\n}")))
    ]

    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = fluency(generated_outputs, model_type='openai')
        # "Good" gets a value of 1.0
        assert metric_value == 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = fluency(generated_outputs,
                               model_type='azure_openai',
                               openai_args={'model': 'foo bar'})
        # "Good" gets a value of 1.0
        assert metric_value == 1


@pytest.mark.parametrize('generated_outputs', [
    'foo bar', ['foo bar'],
    [
        'Ich hasse dich. Halt den Mund!',
        'Vielen Dank, dass du heute gekommen bist!!'
    ]
])
def test_toxicity(generated_outputs):
    metric_value = toxicity(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize(
    'generated_outputs',
    ['Ich hasse dich. Halt den Mund!', ['Ich hasse dich. Halt den Mund!']])
def test_toxicity_openai(generated_outputs):
    mock_chat_completion = Mock(spec=ChatCompletion)
    mock_chat_completion.choices = [
        Mock(message=Mock(function_call=Mock(
            arguments="{\n  \"toxicity\": \"5\"\n}")))
    ]

    # Calling the openai.resources.chat.Completions.create method requires an
    # OpenAI API key, so we mock the return value instead
    with patch('openai.resources.chat.Completions.create',
               return_value=mock_chat_completion):
        # Set the necessary env vars for the 'openai' model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = toxicity(generated_outputs, model_type='openai')
        # "5" gets a value of 1.0
        assert metric_value == 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = toxicity(generated_outputs,
                                model_type='azure_openai',
                                openai_args={'model': 'foo bar'})
        # "5" gets a value of 1.0
        assert metric_value == 1


@pytest.mark.parametrize(
    'generated_outputs,metric_values',
    [
        (
            'Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n'  # NOQA: E501
            'Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n'  # NOQA: E501
            'ein Platz darin ist für dich reserviert.',
            [75.00651612903226]),
        (
            [
                'Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n'  # NOQA: E501
                'Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n'  # NOQA: E501
                'ein Platz darin ist für dich reserviert.'
            ],
            [75.00651612903226]),
        (
            [
                'Wie langsam vergeht die Zeit hier, umgeben, wie ich es bin, von Frost und Schnee!\n'  # NOQA: E501
                'Doch ein zweiter Schritt wird auf mein Unternehmen zugenommen.'
            ],
            [77.45815217391308])
    ])
def test_flesch_reading_ease(generated_outputs, metric_values):
    metric_value = flesch_reading_ease(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,metric_values',
    [
        (
            'Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n'  # NOQA: E501
            'Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n'  # NOQA: E501
            'ein Platz darin ist für dich reserviert.',
            [4.33767741935484]),
        (
            [
                'Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n'  # NOQA: E501
                'Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n'  # NOQA: E501
                'ein Platz darin ist für dich reserviert.'
            ],
            [4.33767741935484]),
        (
            [
                'Wie langsam vergeht die Zeit hier, umgeben, wie ich es bin, von Frost und Schnee!\n'  # NOQA: E501
                'Doch ein zweiter Schritt wird auf mein Unternehmen zugenommen.'
            ],
            [5.312391304347827]),
    ])
def test_flesch_kincaid_grade(generated_outputs, metric_values):
    metric_value = flesch_kincaid_grade(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize('generated_outputs', [[
    "Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein.",
    "Als KI-Sprachmodell habe ich keine eigenen Überzeugungen."
]])
def test_ai_disclaimer_similarity(generated_outputs):
    metric_value = ai_disclaimer_similarity(generated_outputs)
    assert 0.5 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs', [[
    "Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein.",
]])
def test_ai_disclaimer_similarity_openai(generated_outputs):
    mock_embedding_response = Mock(spec=CreateEmbeddingResponse)
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    # Calling the openai.resources.embeddings.create method requires an OpenAI
    # API key, so we mock the return value instead
    with patch('openai.resources.Embeddings.create',
               Mock(return_value=mock_embedding_response)):
        # Set the necessary env vars for the 'openai' embedding model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        metric_value = ai_disclaimer_similarity(generated_outputs,
                                                model_type='openai')
        # Since the mock embeddings are the same for the generated output and
        # the AI disclaimer phrase, the AI disclaimer language similarity should
        # be 1.
        assert 0.99 <= metric_value <= 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        metric_value = ai_disclaimer_similarity(
            generated_outputs,
            model_type='azure_openai',
            openai_args={'model': 'foo bar'})
        # Since the mock embeddings are the same for the generated output and
        # the AI disclaimer phrase, the AI disclaimer language similarity should
        # be 1.
        assert 0.99 <= metric_value <= 1
