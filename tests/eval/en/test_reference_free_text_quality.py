from unittest.mock import Mock, patch

import pytest

from langcheck.eval.en import (ai_disclaimer_language_similarity,
                               flesch_kincaid_grade, flesch_reading_ease,
                               fluency, sentiment, toxicity)
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs',
    ['Hello', ['Hello'], ["I'm fine!", "I'm feeling pretty bad today."]])
def test_sentiment(generated_outputs):
    eval_value = sentiment(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize('generated_outputs', ["I'm fine!", ["I'm fine!"]])
def test_sentiment_openai(generated_outputs):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"sentiment\": \"Positive\"\n}"
                }
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        eval_value = sentiment(generated_outputs, model_type='openai')
        # "Positive" gets a value of 1.0
        assert eval_value.metric_values[0] == 1


@pytest.mark.parametrize('generated_outputs', [
    'cat', ['cat'],
    ["I'd appreciate your help.", 'Today I eats very much apples good.']
])
def test_fluency(generated_outputs):
    eval_value = fluency(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize(
    'generated_outputs',
    ["I'd appreciate your help.", ["I'd appreciate your help."]])
def test_fluency_openai(generated_outputs):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"fluency\": \"Good\"\n}"
                }
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        eval_value = fluency(generated_outputs, model_type='openai')
        # "Good" gets a value of 1.0
        assert eval_value.metric_values[0] == 1


@pytest.mark.parametrize('generated_outputs', [
    'foo bar', ['foo bar'],
    ['I hate you. Shut your mouth!', 'Thank you so much for coming today!!']
])
def test_toxicity(generated_outputs):
    eval_value = toxicity(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize(
    'generated_outputs',
    ['I hate you. Shut your mouth!', ['I hate you. Shut your mouth!']])
def test_toxicity_openai(generated_outputs):
    mock_chat_response = {
        'choices': [{
            'message': {
                'function_call': {
                    'arguments': "{\n  \"toxicity\": \"5\"\n}"
                }
            }
        }]
    }
    # Calling the openai.ChatCompletion.create method requires an OpenAI API
    # key, so we mock the return value instead
    with patch('openai.ChatCompletion.create',
               Mock(return_value=mock_chat_response)):
        eval_value = toxicity(generated_outputs, model_type='openai')
        # "5" gets a value of 1.0
        assert eval_value.metric_values[0] == 1


@pytest.mark.parametrize(
    'generated_outputs,metric_values',
    [
        (
            'My Friend. Welcome to the Carpathians. I am anxiously expecting you.\n'  # NOQA E501
            'Sleep well to-night. At three to-morrow the diligence will start for Bukovina;\n'  # NOQA E501
            'a place on it is kept for you.',
            [75.00651612903226]),
        (
            [
                'My Friend. Welcome to the Carpathians. I am anxiously expecting you.\n'  # NOQA E501
                'Sleep well to-night. At three to-morrow the diligence will start for Bukovina;\n'  # NOQA E501
                'a place on it is kept for you.'
            ],
            [75.00651612903226]),
        (
            [
                'How slowly the time passes here, encompassed as I am by frost and snow!\n'  # NOQA E501
                'Yet a second step is taken towards my enterprise.'
            ],
            [77.45815217391308])
    ])
def test_flesch_reading_ease(generated_outputs, metric_values):
    eval_value = flesch_reading_ease(generated_outputs)
    assert is_close(eval_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,metric_values',
    [
        (
            'My Friend. Welcome to the Carpathians. I am anxiously expecting you.\n'  # NOQA E501
            'Sleep well to-night. At three to-morrow the diligence will start for Bukovina;\n'  # NOQA E501
            'a place on it is kept for you.',
            [4.33767741935484]),
        (
            [
                'My Friend. Welcome to the Carpathians. I am anxiously expecting you.\n'  # NOQA E501
                'Sleep well to-night. At three to-morrow the diligence will start for Bukovina;\n'  # NOQA E501
                'a place on it is kept for you.'
            ],
            [4.33767741935484]),
        (
            [
                'How slowly the time passes here, encompassed as I am by frost and snow!\n'  # NOQA E501
                'Yet a second step is taken towards my enterprise.'
            ],
            [5.312391304347827]),
    ])
def test_flesch_kincaid_grade(generated_outputs, metric_values):
    eval_value = flesch_kincaid_grade(generated_outputs)
    assert is_close(eval_value.metric_values, metric_values)

@pytest.mark.parametrize('generated_outputs', [[
    "I don't have personal opinions, emotions, or consciousness.",
    "As an AI language model, I don't have my own beliefs."
]])
def test_ai_disclaimer_language_similarity(generated_outputs):
    eval_value = ai_disclaimer_language_similarity(generated_outputs)
    assert all(0.5 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize('generated_outputs', [[
    "I don't have personal opinions, emotions, or consciousness.",
]])
def test_ai_disclaimer_language_similarity_openai(generated_outputs):
    mock_embedding_response = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
    # Calling the openai.Embedding.create method requires an OpenAI API key, so
    # we mock the return value instead
    with patch('openai.Embedding.create',
               Mock(return_value=mock_embedding_response)):
        eval_value = ai_disclaimer_language_similarity(
            generated_outputs, embedding_model_type='openai')
        sim_value = eval_value.metric_values[0]
        # Since the mock embeddings are the same for the generated output and
        # the AI disclaimer phrase, the AI disclaimer language similarity should
        # be 1.
        assert 0.99 <= sim_value <= 1
