from __future__ import annotations

from nlpaug.augmenter.char.ocr import OcrAug


def ocr_typo(
    texts: list[str] | str,
    **kwargs,
) -> list[str]:
    '''Generate OCR typo perturbed texts for augmentation.

    Args:
        texts: List of texts to be augmented.

    .. note::
        Any argument that can be passed to
        `nlpaug.augmenter.char.ocr.OcrAug
        <https://nlpaug.readthedocs.io/en/latest/augmenter/char/ocr.html#nlpaug.augmenter.char.ocr.OcrAug>`_
        is acceptable. Some of the more useful ones from `nlpaug` document are
        listed below:
            - ``aug_char_p`` (float): Percentage of character (per token)
              will be augmented.
            - ``aug_char_min`` (int): Minimum number of character will be
              augmented.
            - ``aug_char_max`` (int): Maximum number of character will be
              augmented.
            - ``aug_word_p`` (float): Percentage of word will be augmented.
            - ``aug_word_min`` (int): Minimum number of word will be augmented.
            - ``aug_word_max`` (int): Maximum number of word will be augmented.
        Note that the default values for these arguments are different from the
        ``nlpaug`` defaults. To be more specific, the default values for
        ``aug_char_p`` to be `0.1`, ``aug_char_max`` and ``aug_word_max`` to be
        `None`. See the documentation for more details.

    Returns:
        A list of perturbed texts.
    '''

    kwargs["aug_char_p"] = kwargs.get("aug_char_p", 0.1)
    kwargs["aug_char_max"] = kwargs.get("aug_char_max")
    kwargs["aug_word_max"] = kwargs.get("aug_word_max")

    aug = OcrAug(**kwargs)
    return aug.augment(texts)
