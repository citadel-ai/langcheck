from __future__ import annotations

import abc
from collections.abc import Iterator
from rouge_score.tokenizers import Tokenizer as BaseTokenizer

try:
    import hanlp
    # size 43M+, fine grained tokenizer
    DEFAULT_TOKENIZER_WEIGHT = hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH
except ModuleNotFoundError:
    raise ModuleNotFoundError("No module named 'HanLP'.\n"
                              "Install it by pip install -U hanlp")

# Chinese
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

    The default tokenizer to calculate rouge score base on HanLP.

    .. note::
        HanLP https://github.com/hankcs/HanLP/tree/doc-zh
        HanLP is the newest tokenizer that have developer maintaince.
        1.HanLP have multi-task mode and single task mode. Multitask
        Model need download 400MB+  pretrained weight, in contrast, 
        single task only need 40MB+. Use single task mode in default. 
        2.LLM generated content have lot of sentences in most situtation.
        use HanLP pipeline mode for concurrency. 
    '''

    def __init__(self) -> None:
        super().__init__()
        tokenizer = hanlp.load(DEFAULT_TOKENIZER_WEIGHT)
        self.tokenzier_pipeline = hanlp.pipeline()\
                                    .append(hanlp.utils.rules.split_sentence) \
                                    .append(tokenizer)\
                                    .append(lambda sents: sum(sents, []))

    def _tokenize(self, text: str) -> Iterator[str]:
        tokens = self.tokenzier_pipeline(text)
        return tokens
