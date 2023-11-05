from __future__ import annotations

import abcip3 install
from collections.abc import Iterator
from rouge_score.tokenizers import Tokenizer as BaseTokenizer

# Chinese
# https://github.com/yikeke/zh-style-guide/blob/master/source/%E6%A0%87%E7%82%B9%E7%AC%A6%E5%8F%B7/%E5%B8%B8%E7%94%A8%E4%B8%AD%E6%96%87%E6%A0%87%E7%82%B9%E7%AC%A6%E5%8F%B7.md
_PUNCTUATIONS = ['、', '，','。',
                '：','；',
                '?','!', "？", "！",
                '～','-','—','——',
                '……','⋯⋯',
                '/'
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

class PKUTokenizer(_ChineseTokenizer):
    '''
    An optional tokenizer to calculate rouge score base PKUTokenizer. 

    .. note::
        website:https://github.com/lancopku/pkuseg-python. Without maintain for
        over 4 years.
    '''
    def __init__(self):
        try:
            import pkuseg
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "No module named 'jieba'.\n"
                )
        self.tokenizer = pkuseg(postag=True)
    
    def __tokenize(self,text):
        return self.tokenizer.cut(text)
