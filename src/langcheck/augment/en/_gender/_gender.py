from __future__ import annotations

from collections.abc import Iterable
from random import choice
from types import MappingProxyType

import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from langcheck.augment.en._gender._gender_pronouns import (_PRONOUNS_DICT,
                                                           _BaseGenderPronouns)

# This dictionary is used to determine the form of the pronoun.
# Note that his and hers are not included in this dictionary because they can be
# either of two different forms depending on the context.
_PRONOUNS_FORM_DICT = MappingProxyType({
    "she": "subject",
    "he": "subject",
    "they": "subject",
    "their": "dependent_possessive",
    "him": "object",
    "them": "object",
    "hers": "independent_possessive",
    "theirs": "independent_possessive",
    "himself": "reflexive",
    "herself": "reflexive",
    "themselves": "reflexive"
})


def _get_pronoun_form(word: str, tag: str) -> str | None:
    """Get pronoun form of the word."""
    # Handle degenerated cases.
    if tag == "PRP$":
        # PRP$ tag denotes a possessive pronoun.
        return "dependent_possessive"
    if word.lower() == "her":
        return "object"
    if word.lower() == "his":
        return "independent_possessive"
    return _PRONOUNS_FORM_DICT.get(word.lower())


def _replace_pronoun(word: str, tag: str, target_pronouns: _BaseGenderPronouns):
    # When the word is not a pronoun, return the word itself.
    if (pronoun_form := _get_pronoun_form(word, tag)) is None:
        return word
    # Replace the pronoun with the target pronoun with the same form.
    replaced_pronoun = getattr(target_pronouns, pronoun_form)
    if word.isupper():
        return replaced_pronoun.upper()
    elif word.istitle():
        return replaced_pronoun.title()
    # When the word is not first letter capitalized or uppercase only, return
    # the word in lowercase, regardless of how irregularly the word is
    # capitalized.
    return replaced_pronoun


def _replace_gender_pronouns(
    text: str,
    target_pronouns: _BaseGenderPronouns,
) -> str:
    """Replace target pronouns in text with new pronouns.

    Args:
        target_pronouns (_BaseGenderPronouns): Pronouns to replace with.
        text (str): Text to be augmented.

    Returns:
        str: Augmented text.
    """
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    tagged_words = pos_tag(word_tokenize(text))
    augmented_words = [
        _replace_pronoun(word, tag, target_pronouns)
        for (word, tag) in tagged_words
    ]
    return TreebankWordDetokenizer().detokenize(augmented_words)


def gender(
    texts: Iterable[str] | str,
    *,
    to_gender: str = 'plural',
) -> list[str]:
    """Replace pronouns with that of specified gender.

        Args:
            texts: Iterable of texts to be augmented.
            to_gender: Replacing pronoun type string ('male', 'female',
            'neutral', or 'plural'). Default to `plural`.

        Returns:
            List of sentences with replaced pronouns.

        .. note::
            Replacing neopronouns with other neopronouns is not supported yet
            because `NLTK <https://www.nltk.org/>`_ does not recognize them.

    """
    if (to_gender is not None) and (to_gender not in [
            "female", "male", "neutral", "plural"
    ]):
        raise ValueError(
            f"The argument 'gender' must be one of 'female', 'male', 'neutral',"
            f" or 'plural', but got {to_gender}.")
    target_gender = choice(_PRONOUNS_DICT[to_gender])

    if isinstance(texts, str):
        return [_replace_gender_pronouns(texts, target_gender)]
    elif isinstance(texts, Iterable):
        return [_replace_gender_pronouns(text, target_gender) for text in texts]
    else:
        raise TypeError(
            f"Expected texts to be a string or iterable of strings but got "
            f"{type(texts)}.")
