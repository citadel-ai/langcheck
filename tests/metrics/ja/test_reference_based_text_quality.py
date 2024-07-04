import os
from typing import Callable, Optional
from unittest.mock import Mock, patch

import pytest
from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
)
from langcheck.metrics.ja import (
    JanomeTokenizer,
    MeCabTokenizer,
    rouge1,
    rouge2,
    rougeL,
    semantic_similarity,
)
from langcheck.metrics.ja._tokenizers import _JapaneseTokenizer
from langcheck.metrics.metric_value import MetricValue
from openai.types import CreateEmbeddingResponse

from tests.utils import is_close

################################################################################
# Tests
################################################################################

parametrize_rouge_function = pytest.mark.parametrize("rouge_function",
                                                     [rouge1, rouge2, rougeL])
parametrize_tokenizer = pytest.mark.parametrize("tokenizer", [
    None, JanomeTokenizer,
    pytest.param(MeCabTokenizer, marks=pytest.mark.optional)
])


@pytest.mark.parametrize("generated_outputs,reference_outputs",
                         [("猫がマットの上に座った。", "猫がマットの上に座った。"),
                          (["猫がマットの上に座った。"], ["猫がマットの上に座った。"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_identical(generated_outputs: str, reference_outputs: str,
                         rouge_function: Callable[
                             [str, str, Optional[_JapaneseTokenizer]],
                             MetricValue[float]],
                         tokenizer: Optional[_JapaneseTokenizer]) -> None:
    # All ROUGE scores are 1 if the generated and reference outputs are
    # identical
    actual_metric_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer()  # type: ignore[reportGeneralTypeIssues]
        if tokenizer else None)
    assert actual_metric_value.metric_values == [1.]
    assert actual_metric_value.language == "ja"


@pytest.mark.parametrize("generated_outputs,reference_outputs",
                         [("猫が、マットの上に座った。", "私は、アイスクリームを食べます。"),
                          (["猫が、マットの上に座った。"], ["私は、アイスクリームを食べます。"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_no_overlap(generated_outputs: str, reference_outputs: str,
                          rouge_function: Callable[[str, str],
                                                   MetricValue[float]],
                          tokenizer: Optional[_JapaneseTokenizer]) -> None:
    # All ROUGE scores are 0 if the generated and reference outputs have no
    # overlapping words
    actual_metric_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer()  # type: ignore[reportGeneralTypeIssues]
        if tokenizer else None)
    assert actual_metric_value.metric_values == [0.]
    assert actual_metric_value.language == "ja"


@pytest.mark.parametrize("generated_outputs,reference_outputs",
                         [("猫がマットの上に座った。", "猫がマットの上に座っている。"),
                          (["猫がマットの上に座った。"], ["猫がマットの上に座っている。"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_some_overlap(generated_outputs: str, reference_outputs: str,
                            rouge_function: Callable[[str, str],
                                                     MetricValue[float]],
                            tokenizer: Optional[_JapaneseTokenizer]) -> None:
    expected_value = {
        "rouge1": [0.823529411764706],
        "rouge2": [0.7999999999999999],
        "rougeL": [0.823529411764706]
    }
    # The ROUGE-2 score is lower than the ROUGE-1 and ROUGE-L scores
    actual_metric_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer()  # type: ignore[reportGeneralTypeIssues]
        if tokenizer else None)
    is_close(actual_metric_value.metric_values,
             expected_value[rouge_function.__name__])
    assert actual_metric_value.language == "ja"


@pytest.mark.parametrize("generated_outputs,reference_outputs",
                         [("猫が座っています。", "猫が座っています。"),
                          (["猫が座っています。"], ["猫が座っています。"])])
def test_semantic_similarity_identical(generated_outputs, reference_outputs):
    metric_value = semantic_similarity(generated_outputs, reference_outputs)
    assert 0.99 <= metric_value <= 1


@pytest.mark.parametrize("generated_outputs,reference_outputs",
                         [("猫が座っています。", "ネコがすわっています。"),
                          (["猫が座っています。"], ["ネコがすわっています。"])])
def test_semantic_similarity_character_sensitivity(generated_outputs,
                                                   reference_outputs):
    metric_value = semantic_similarity(generated_outputs, reference_outputs)
    assert 0.75 <= metric_value <= 1


@pytest.mark.parametrize("generated_outputs,reference_outputs",
                         [("猫が座っています。", "僕はアイスクリームを食べるのが好きです。"),
                          (["猫が座っています。"], ["僕はアイスクリームを食べるのが好きです。"])])
def test_semantic_similarity_not_similar(generated_outputs, reference_outputs):
    metric_value = semantic_similarity(generated_outputs, reference_outputs)
    assert 0.0 <= metric_value <= 0.25


@pytest.mark.parametrize("generated_outputs,reference_outputs",
                         [("猫が座っています。", "猫が座っています。"),
                          (["猫が座っています。"], ["猫が座っています。"])])
def test_semantic_similarity_openai(generated_outputs, reference_outputs):
    mock_embedding_response = Mock(spec=CreateEmbeddingResponse)
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    # Calling the openai.resources.Embeddings.create method requires an OpenAI
    # API key, so we mock the return value instead
    with patch("openai.resources.Embeddings.create",
               Mock(return_value=mock_embedding_response)):
        # Set the necessary env vars for the 'openai' embedding model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        openai_client = OpenAIEvalClient()
        metric_value = semantic_similarity(generated_outputs,
                                           reference_outputs,
                                           eval_model=openai_client)
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= metric_value <= 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        azure_openai_client = AzureOpenAIEvalClient(
            embedding_model_name="foo bar")
        metric_value = semantic_similarity(generated_outputs,
                                           reference_outputs,
                                           eval_model=azure_openai_client)
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= metric_value <= 1
