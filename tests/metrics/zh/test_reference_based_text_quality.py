from typing import Callable, Optional
from unittest.mock import Mock, patch

import pytest

from langcheck.metrics.zh import (HanLPTokenizer, semantic_similarity)
from langcheck.metrics.zh._tokenizers import _ChineseTokenizer
from langcheck.metrics.metric_value import MetricValue
from tests.utils import is_close

################################################################################
# Tests
################################################################################

parametrize_tokenizer = pytest.mark.parametrize('tokenizer',
                                                [None, HanLPTokenizer])


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("那里有一本三体小说。", "那里有一本三体小说。"),
                          (["那里有一本三体小说。"], ["那里有一本三体小说。"])])
def test_semantic_similarity_identical(generated_outputs, reference_outputs):
    metric_value = semantic_similarity(generated_outputs, reference_outputs)
    assert 0.99 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("php是世界上最好的语言，学计算机要从娃娃抓起。",
                          "在石家庄，有一支摇滚乐队，他们创作了很多音乐。"),
                          (["php是世界上最好的语言，学计算机要从娃娃抓起。"],
                          ["在石家庄，有一支摇滚乐队，他们创作了很多音乐。"])
                          ])
def test_semantic_similarity_not_similar(generated_outputs, reference_outputs):
    # hard to find two sentence similiar < 0.25 using current model
    # if '。' in one of the sentence pair be deleted, it would < 0.25 much easier
    metric_value = semantic_similarity(generated_outputs, reference_outputs)
    assert 0.0 <= metric_value <= 0.36


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [("学习中文很快乐。", "学习中文很快乐。"),
                          (["学习中文很快乐。"], ["学习中文很快乐。"])])
def test_semantic_similarity_openai(generated_outputs, reference_outputs):
    mock_embedding_response = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
    # Calling the openai.Embedding.create method requires an OpenAI API key, so
    # we mock the return value instead
    with patch('openai.Embedding.create',
               Mock(return_value=mock_embedding_response)):
        metric_value = semantic_similarity(generated_outputs,
                                           reference_outputs,
                                           embedding_model_type='openai')
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= metric_value <= 1
