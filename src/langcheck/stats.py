import string
from dataclasses import dataclass

import nltk
from nltk.corpus import cmudict
from nltk.tokenize import SyllableTokenizer


@dataclass
class TextStats:
    num_sentences: int
    num_words: int
    num_syllables: int


def compute_stats(input_text: str) -> TextStats:
    '''Compute statics about the given input text.

    Args:
        input_text: Text you want to compute the stats for

    Returns:
        A :class:`~langcheck.stats.TextStats` object
    '''

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        nltk.download('cmudict')

    sentences = nltk.tokenize.sent_tokenize(input_text)

    words = sum(
        [nltk.tokenize.word_tokenize(sentence) for sentence in sentences], [])

    # Filter out "words" like "!", ".", ... etc
    def _all_punctuations(input_str: str) -> bool:
        return all(c in string.punctuation for c in input_str)

    words = [word for word in words if not _all_punctuations(word)]

    tokenizer = SyllableTokenizer()
    syllable_dict = cmudict.dict()

    # Count syllables in a word by checking a dictionary first, and falling back
    # to best-effort SyllableTokenizer
    def _count_syllables(word):
        word = word.lower()
        if word in syllable_dict:
            return len([
                phoneme for phoneme in syllable_dict[word][0]
                if phoneme[-1] in ['0', '1', '2']
            ])
        else:
            syllables = tokenizer.tokenize(word)
            return len([
                syllable for syllable in syllables
                if not _all_punctuations(syllable)
            ])

    num_syllables = sum([_count_syllables(word) for word in words])

    return TextStats(num_sentences=len(sentences),
                     num_words=len(words),
                     num_syllables=num_syllables)
