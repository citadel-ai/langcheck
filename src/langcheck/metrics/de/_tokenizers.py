from typing import List

from nltk.stem.cistem import Cistem
from nltk.tokenize import word_tokenize
from rouge_score.tokenizers import Tokenizer as BaseTokenizer


class DeTokenizer(BaseTokenizer):
    """Tokenizer for German.

    This tokenizer is used to calculate rouge score for German.
    """

    def __init__(self, stemmer=False):
        self.stemmer = None
        if stemmer:
            self.stemmer = Cistem()

    def tokenize(self, text: str) -> List[str]:
        if self.stemmer:
            # use only the stem part of the word
            text, _ = self.stemmer.segment(text)
        return word_tokenize(text)
