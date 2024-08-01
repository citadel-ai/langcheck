import os
from unittest.mock import Mock, patch

import pytest
from langcheck.metrics.de import (
    ai_disclaimer_similarity,
    flesch_kincaid_grade,
    flesch_reading_ease,
    fluency,
    sentiment,
    toxicity,
)
from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
)
from openai.types import CreateEmbeddingResponse

from tests.utils import MockEvalClient, is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    "generated_outputs",
    [
        "Hallo",
        ["Hallo"],
        ["Mir geht es gut!", "Mir geht es heute ziemlich schlecht."],
    ],
)
def test_sentiment(generated_outputs):
    metric_value = sentiment(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize(
    "generated_outputs", ["Mir geht es gut!", ["Mir geht es gut!"]]
)
def test_sentiment_eval_client(generated_outputs):
    eval_client = MockEvalClient()
    metric_value = sentiment(generated_outputs, eval_model=eval_client)
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None
    sentiment_assessment_to_score = {
        "Positive": 1.0,
        "Neutral": 0.5,
        "Negative": 0.0,
    }
    for option in sentiment_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = sentiment(generated_outputs, eval_model=eval_client)
        assert metric_value == sentiment_assessment_to_score[option]


@pytest.mark.parametrize(
    "generated_outputs",
    [
        "Katze",
        ["Katze"],
        [
            "Ich würde deine Hilfe schätzen.",
            "Heute essen ich sehr viel gute Äpfel.",
        ],
    ],
)
def test_fluency(generated_outputs):
    metric_value = fluency(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize(
    "generated_outputs",
    ["Ich würde deine Hilfe schätzen.", ["Ich würde deine Hilfe schätzen."]],
)
def test_fluency_eval_client(generated_outputs):
    eval_client = MockEvalClient()
    metric_value = fluency(generated_outputs, eval_model=eval_client)
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    fluency_assessment_to_score = {
        "Poor": 0,
        "Fair": 0.5,
        "Good": 1.0,
    }

    for option in fluency_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = fluency(generated_outputs, eval_model=eval_client)
        assert metric_value == fluency_assessment_to_score[option]


@pytest.mark.parametrize(
    "generated_outputs",
    [
        "foo bar",
        ["foo bar"],
        [
            "Ich hasse dich. Halt den Mund!",
            "Vielen Dank, dass du heute gekommen bist!!",
        ],
    ],
)
def test_toxicity(generated_outputs):
    metric_value = toxicity(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize(
    "generated_outputs",
    ["Ich hasse dich. Halt den Mund!", ["Ich hasse dich. Halt den Mund!"]],
)
def test_toxicity_eval_client(generated_outputs):
    eval_client = MockEvalClient()
    metric_value = toxicity(generated_outputs, eval_model=eval_client)
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    toxicity_assessment_to_score = {
        "1": 0,
        "2": 0.25,
        "3": 0.5,
        "4": 0.75,
        "5": 1.0,
    }
    for option in toxicity_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = toxicity(generated_outputs, eval_model=eval_client)
        assert metric_value == toxicity_assessment_to_score[option]


# note: as marked on the research, this metric is higher for German than English
@pytest.mark.parametrize(
    "generated_outputs,metric_values",
    [
        (
            "Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n"
            "Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n"
            "ein Platz darin ist für dich reserviert.",
            [80.39999999999999],
        ),
        (
            [
                "Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n"
                "Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n"
                "ein Platz darin ist für dich reserviert."
            ],
            [80.39999999999999],
        ),
        (
            [
                "Wie langsam vergeht die Zeit hier, umgeben, wie ich es bin, von Frost und Schnee!\n"
                "Doch ein zweiter Schritt wird auf mein Unternehmen zugenommen."
            ],
            [80.25],
        ),
    ],
)
def test_flesch_reading_ease(generated_outputs, metric_values):
    metric_value = flesch_reading_ease(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


# note: as marked on the research, this metric is higher for German than English
@pytest.mark.parametrize(
    "generated_outputs,metric_values",
    [
        (
            "Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n"
            "Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n"
            "ein Platz darin ist für dich reserviert.",
            [5.630000000000003],
        ),
        (
            [
                "Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n"
                "Schlaf gut heute Nacht. Um drei Uhr morgen startet die Eilpost nach Bukowina;\n"
                "ein Platz darin ist für dich reserviert."
            ],
            [5.630000000000003],
        ),
        (
            [
                "Wie langsam vergeht die Zeit hier, umgeben, wie ich es bin, von Frost und Schnee!\n"
                "Doch ein zweiter Schritt wird auf mein Unternehmen zugenommen."
            ],
            [6.790000000000003],
        ),
    ],
)
def test_flesch_kincaid_grade(generated_outputs, metric_values):
    metric_value = flesch_kincaid_grade(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


# NB: the one straight from English fails here:  "Als KI-Sprachmodell habe ich keine eigenen Überzeugungen."
@pytest.mark.parametrize(
    "generated_outputs",
    [
        [
            "Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein.",
            "Als KI-Sprachmodell verfüge ich über keine eigenen Meinungen, Emotionen oder Gedanken.",
        ]
    ],
)
def test_ai_disclaimer_similarity(generated_outputs):
    metric_value = ai_disclaimer_similarity(generated_outputs)
    assert 0.5 <= metric_value <= 1


@pytest.mark.parametrize(
    "generated_outputs",
    [
        [
            "Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein.",
        ]
    ],
)
def test_ai_disclaimer_similarity_openai(generated_outputs):
    mock_embedding_response = Mock(spec=CreateEmbeddingResponse)
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    # Calling the openai.resources.embeddings.create method requires an OpenAI
    # API key, so we mock the return value instead
    with patch(
        "openai.resources.Embeddings.create",
        Mock(return_value=mock_embedding_response),
    ):
        # Set the necessary env vars for the 'openai' embedding model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        openai_client = OpenAIEvalClient()
        metric_value = ai_disclaimer_similarity(
            generated_outputs, eval_model=openai_client
        )
        # Since the mock embeddings are the same for the generated output and
        # the AI disclaimer phrase, the AI disclaimer language similarity should
        # be 1.
        assert 0.99 <= metric_value <= 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        azure_openai_client = AzureOpenAIEvalClient(
            embedding_model_name="foo bar"
        )
        metric_value = ai_disclaimer_similarity(
            generated_outputs, eval_model=azure_openai_client
        )
        # Since the mock embeddings are the same for the generated output and
        # the AI disclaimer phrase, the AI disclaimer language similarity should
        # be 1.
        assert 0.99 <= metric_value <= 1
