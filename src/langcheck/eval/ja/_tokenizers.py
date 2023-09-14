import abc
from collections.abc import Iterator
from typing import List
from typing import Iterator
from rouge_score.tokenizers import Tokenizer

# Japanese punctuation characters
_PUNCTUATIONS = ['、', '。', '，', '．', ',', '.', "?", "!", "？", "！"]


class _JapaneseTokenizer(Tokenizer):

    @abc.abstractmethod
    def _tokenize(self, text: str) -> Iterator[str]:
        raise NotImplementedError(
            "Tokenizer for Japanese must override `_tokenize()` method")

    def tokenize(self, text: str) -> List[str]:
        tokens = self._tokenize(text)
        return [
            token for token in tokens if (token and token not in _PUNCTUATIONS)
        ]


class MecabTokeninzer(_JapaneseTokenizer):

    class _MecabNodeSurfaceIterator(Iterator):

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
        return MecabTokeninzer._MecabNodeSurfaceIterator(
            self.tokenizer.parseToNode(text))


class JanomeTokenizer(_JapaneseTokenizer):

    def __init__(self):
        from janome.tokenizer import Tokenizer
        self.tokenizer = Tokenizer()

    def _tokenize(self, text: str):
        return self.tokenizer.tokenize(text, wakati=True)
