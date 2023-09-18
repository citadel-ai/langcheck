from __future__ import annotations

import abc
from collections.abc import Iterator
from janome.tokenizer import Tokenizer

from rouge_score.tokenizers import Tokenizer as BaseTokenizer

# Japanese punctuation characters
_PUNCTUATIONS = ['、', '。', '，', '．', ',', '.', "?", "!", "？", "！"]


class _JapaneseTokenizer(BaseTokenizer):

    @abc.abstractmethod
    def _tokenize(self, text: str) -> Iterator[str]:
        raise NotImplementedError(
            "Tokenizer for Japanese must override `_tokenize()` method")

    def tokenize(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        return [
            token for token in tokens if (token and token not in _PUNCTUATIONS)
        ]


class MeCabTokenizer(_JapaneseTokenizer):

    class _MeCabNodeSurfaceIterator(Iterator):

        def __init__(self, node) -> None:
            self._node = node

        def __next__(self):
            if self._node is None:
                raise StopIteration
            else:
                node = self._node
                self._node = self._node.next
                return node.surface

    def __init__(self):
        import MeCab
        self.tokenizer = MeCab.Tagger()

    def _tokenize(self, text):
        return MeCabTokenizer._MeCabNodeSurfaceIterator(
            self.tokenizer.parseToNode(text))


class JanomeTokenizer(_JapaneseTokenizer):
    '''Janome based Tokenizer for Japanese.

    The default tokenizer for caliculating rouge score.
    Janome is a pure python library and introduces no additional dependency.
    '''

    def __init__(self):
        self.tokenizer = Tokenizer()

    def _tokenize(self, text: str):
        return self.tokenizer.tokenize(text, wakati=True)
