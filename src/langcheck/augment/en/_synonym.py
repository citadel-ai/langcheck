from __future__ import annotations

from nlpaug.augmenter.word import SynonymAug


def synonym(
    texts: list[str] | str,
    **kwargs,
) -> list[str]:
    '''Generate texts where some of the input words are replaced with synonyms.

    Args:
        texts: List of texts to be augmented.
        aug_p: Percentage of words which will be augmented. Default to `0.1`.
        aug_max: Maximum number of words which will be augmented. Default to
            `None`.

    .. note::
        Any argument that can be passed to
        `nlpaug.augmenter.word.SynonymAug
        <https://nlpaug.readthedocs.io/en/latest/_modules/nlpaug/augmenter/word/synonym.html>`_
        is acceptable. Some of the more useful ones from `nlpaug` document are
        listed below:

          - ``aug_p`` (float): Percentage of word will be augmented.
          - ``aug_min``: Minimum number of word will be augmented.
          - ``aug_max``: Maximum number of word will be augmented.

        Note that the default values for these arguments are different from the
        ``nlpaug`` defaults. To be more specific, the default values for
        ``aug_p`` to be `0.1` and ``aug_max`` to be `None`. See the source
        code for mode details.

    Returns:
        A list of perturbed texts.
    '''

    kwargs["aug_p"] = kwargs.get("aug_p", 0.1)
    kwargs["aug_max"] = kwargs.get("aug_max")

    aug = SynonymAug(**kwargs)
    return aug.augment(texts)
