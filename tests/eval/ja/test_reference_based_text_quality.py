from typing import Callable, Optional
from unittest.mock import Mock, patch

import pytest

from langcheck.eval.eval_value import EvalValue
from langcheck.eval.ja import (JanomeTokenizer, MeCabTokenizer, rouge1, rouge2,
                               rougeL, semantic_sim)
from langcheck.eval.ja._tokenizers import _JapaneseTokenizer
from tests.utils import is_close

################################################################################
# Tests
################################################################################

parametrize_rouge_function = pytest.mark.parametrize("rouge_function",
                                                     [rouge1, rouge2, rougeL])
parametrize_tokenizer = pytest.mark.parametrize('tokenizer', [
    None, JanomeTokenizer,
    pytest.param(MeCabTokenizer, marks=pytest.mark.optional)
])


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫がマットの上に座った。"], ["猫がマットの上に座った。"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_identical(generated_outputs: str, reference_outputs: str,
                         rouge_function: Callable[[str, str], EvalValue[float]],
                         tokenizer: Optional[_JapaneseTokenizer]) -> None:
    # All ROUGE scores are 1 if the generated and reference outputs are
    # identical
    actual_eval_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer() if tokenizer else None)
    assert actual_eval_value.metric_values == [1.]
    assert actual_eval_value.language == 'ja'


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が、マットの上に座った。"], ["私は、アイスクリームを食べます。"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_no_overlap(generated_outputs: str, reference_outputs: str,
                          rouge_function: Callable[[str, str],
                                                   EvalValue[float]],
                          tokenizer: Optional[_JapaneseTokenizer]) -> None:
    # All ROUGE scores are 0 if the generated and reference outputs have no
    # overlapping words
    actual_eval_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer() if tokenizer else None)
    assert actual_eval_value.metric_values == [0.]
    assert actual_eval_value.language == 'ja'


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫がマットの上に座った。"], ["猫がマットの上に座っている。"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_some_overlap(generated_outputs: str, reference_outputs: str,
                            rouge_function: Callable[[str, str],
                                                     EvalValue[float]],
                            tokenizer: Optional[_JapaneseTokenizer]) -> None:
    expected_value = {
        rouge1: [0.823529411764706],
        rouge2: [0.7999999999999999],
        rougeL: [0.823529411764706]
    }
    # The ROUGE-2 score is lower than the ROUGE-1 and ROUGE-L scores
    actual_eval_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer() if tokenizer else None)
    is_close(actual_eval_value.metric_values, expected_value[rouge_function])
    assert actual_eval_value.language == 'ja'


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が座っています。"], ["猫が座っています。"])])
def test_semantic_sim_identical(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.99 <= semantic_sim_value <= 1


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が座っています。"], ["ネコがすわっています。"])])
def test_semantic_sim_character_sensitivity(generated_outputs,
                                            reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.75 <= semantic_sim_value <= 1


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が座っています。"], ["僕はアイスクリームを食べるのが好きです。"])])
def test_semantic_sim_not_similar(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.0 <= semantic_sim_value <= 0.25


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が座っています。"], ["猫が座っています。"])])
def test_semantic_sim_openai(generated_outputs, reference_outputs):
    mock_embedding_response = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
    # Calling the openai.Embedding.create method requires an OpenAI API key, so
    # we mock the return value instead
    with patch('openai.Embedding.create',
               Mock(return_value=mock_embedding_response)):
        eval_value = semantic_sim(generated_outputs,
                                  reference_outputs,
                                  embedding_model_type='openai')
        semantic_sim_value = eval_value.metric_values[0]
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= semantic_sim_value <= 1
