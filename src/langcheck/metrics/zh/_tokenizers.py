from __future__ import annotations

import abc
from collections.abc import Iterator

import hanlp
from rouge_score.tokenizers import Tokenizer as BaseTokenizer

# size 43M+, fine grained tokenizer
DEFAULT_TOKENIZER_WEIGHT = hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH  # type: ignore[reportGeneralTypeIssues] # noqa: E501
# Chinese puncuations list
# https://github.com/yikeke/zh-style-guide/blob/master/source/%E6%A0%87%E7%82%B9%E7%AC%A6%E5%8F%B7/%E5%B8%B8%E7%94%A8%E4%B8%AD%E6%96%87%E6%A0%87%E7%82%B9%E7%AC%A6%E5%8F%B7.md
_PUNCTUATIONS = [
    '、', '，', '。', '：', '；', '?', '!', "？", "！", '～', '-', '—', '——', '……',
    '⋯⋯', '/'
]


class _ChineseTokenizer(BaseTokenizer):

    @abc.abstractmethod
    def _tokenize(self, text: str) -> Iterator[str]:
        raise NotImplementedError(
            "Tokenizer for Chinese must override `_tokenize()` method")

    def tokenize(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        return [
            token for token in tokens if (token and token not in _PUNCTUATIONS)
        ]


class HanLPTokenizer(_ChineseTokenizer):
    '''HanLP based Tokenizer for Chinese.

    The default tokenizer to calculate rouge score based on HanLP.

    .. note::
        `HanLP <https://github.com/hankcs/HanLP/tree/doc-zh>`_ is an actively
        maintained NLP library that was initially developed for Chinese
        language processing.
        We run HanLP's single-task models using HanLP's pipeline mode, because:
        1. HanLP has both multi-task models and single-task models. The
        multi-task models are quite large (generally 400MB+), whereas the
        single-task models are only ~40MB. So, we use a single-task model by
        default.
        2. HanLP's pipeline mode allows processing of long texts (i.e. many
        sentences) efficiently in parallel. It splits long text into sentences
        and applies the tokenizer to the sentences in parallel.
    '''

    def __init__(self) -> None:
        super().__init__()
        tokenizer = hanlp.load(DEFAULT_TOKENIZER_WEIGHT)
        self.tokenizer_pipeline = hanlp.pipeline().\
            append(hanlp.utils.rules.split_sentence)  # type: ignore[reportGeneralTypeIssues] # NOQA: E501
        self.tokenizer_pipeline = self.tokenizer_pipeline.\
            append(tokenizer).\
            append(lambda sents: sum(sents, []))

    def _tokenize(self, text: str) -> Iterator[str]:
        tokens = self.tokenizer_pipeline(text)
        return tokens  # type: ignore[reportGeneralTypeIssues]
