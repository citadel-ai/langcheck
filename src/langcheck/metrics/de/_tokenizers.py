from nltk.stem.cistem import Cistem

from rouge_score.tokenizers import Tokenizer as BaseTokenizer


class DeTokenizer(BaseTokenizer):
    """Tokenizer for German.

    This tokenizer is used to calculate rouge score for German.
    """

    def __init__(self):
        self.tokenizer = Cistem()

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)
