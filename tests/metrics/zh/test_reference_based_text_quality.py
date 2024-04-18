import os
from typing import Callable, Optional
from unittest.mock import Mock, patch

import pytest
from openai.types import CreateEmbeddingResponse

from langcheck.metrics.eval_clients import (AzureOpenAIEvalClient,
                                            OpenAIEvalClient)
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.zh import (HanLPTokenizer, rouge1, rouge2, rougeL,
                                  semantic_similarity)
from langcheck.metrics.zh._tokenizers import _ChineseTokenizer
from tests.utils import is_close

################################################################################
# Tests
################################################################################
parametrize_rouge_function = pytest.mark.parametrize("rouge_function",
                                                     [rouge1, rouge2, rougeL])
parametrize_tokenizer = pytest.mark.parametrize('tokenizer',
                                                [None, HanLPTokenizer])


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("宇宙的终极答案是什么？", "宇宙的终极答案是什么。"),
                          (["宇宙的终极答案是什么。"], ["宇宙的终极答案是什么？"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_identical(generated_outputs: str, reference_outputs: str,
                         rouge_function: Callable[
                             [str, str, Optional[_ChineseTokenizer]],
                             MetricValue[float]],
                         tokenizer: Optional[_ChineseTokenizer]) -> None:
    # All ROUGE scores are 1 if the generated and reference outputs are
    # identical
    actual_metric_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer()  # type: ignore[reportGeneralTypeIssues]
        if tokenizer else None)
    assert actual_metric_value.metric_values == [1.]
    assert actual_metric_value.language == 'zh'


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("这样的姑娘是受不了的。", "您到底有什么事？"),
                          (["这样的姑娘是受不了的。"], ["您到底有什么事？"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_no_overlap(generated_outputs: str, reference_outputs: str,
                          rouge_function: Callable[[str, str],
                                                   MetricValue[float]],
                          tokenizer: Optional[_ChineseTokenizer]) -> None:
    # All ROUGE scores are 0 if the generated and reference outputs have no
    # overlapping words
    actual_metric_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer()  # type: ignore[reportGeneralTypeIssues]
        if tokenizer else None)
    assert actual_metric_value.metric_values == [0.]
    assert actual_metric_value.language == 'zh'


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("床前明月光，下一句是什么？", "床前明月光的下一句是什么？"),
                          (["床前明月光，下一句是什么？"], ["床前明月光的下一句是什么？"])])
@parametrize_rouge_function
@parametrize_tokenizer
def test_rouge_some_overlap(generated_outputs: str, reference_outputs: str,
                            rouge_function: Callable[[str, str],
                                                     MetricValue[float]],
                            tokenizer: Optional[_ChineseTokenizer]) -> None:
    expected_value = {
        'rouge1': [0.941176],
        'rouge2': [0.8],
        'rougeL': [0.941176]
    }
    # The ROUGE-2 score is lower than the ROUGE-1 and ROUGE-L scores
    actual_metric_value = rouge_function(
        generated_outputs,
        reference_outputs,
        tokenizer=tokenizer()  # type: ignore[reportGeneralTypeIssues]
        if tokenizer else None)
    is_close(actual_metric_value.metric_values,
             expected_value[rouge_function.__name__])
    assert actual_metric_value.language == 'zh'


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("那里有一本三体小说。", "那里有一本三体小说。"),
                          (["那里有一本三体小说。"], ["那里有一本三体小说。"])])
def test_semantic_similarity_identical(generated_outputs, reference_outputs):
    metric_value = semantic_similarity(generated_outputs, reference_outputs)
    assert 0.99 <= metric_value <= 1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("php是世界上最好的语言，学计算机要从娃娃抓起。", "在石家庄，有一支摇滚乐队，他们创作了很多音乐。"),
     (["php是世界上最好的语言，学计算机要从娃娃抓起。"], ["在石家庄，有一支摇滚乐队，他们创作了很多音乐。"])])
def test_semantic_similarity_not_similar(generated_outputs, reference_outputs):
    metric_value = semantic_similarity(generated_outputs, reference_outputs)
    assert 0.0 <= metric_value <= 0.5


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("学习中文很快乐。", "学习中文很快乐。"),
                          (["学习中文很快乐。"], ["学习中文很快乐。"])])
def test_semantic_similarity_openai(generated_outputs, reference_outputs):
    mock_embedding_response = Mock(spec=CreateEmbeddingResponse)
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    # Calling the openai.Embedding.create method requires an OpenAI API key, so
    # we mock the return value instead
    with patch('openai.resources.Embeddings.create',
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
            embedding_model_name='foo bar')
        metric_value = semantic_similarity(generated_outputs,
                                           reference_outputs,
                                           eval_model=azure_openai_client)
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= metric_value <= 1
