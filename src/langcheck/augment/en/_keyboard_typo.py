from __future__ import annotations

import nlpaug.augmenter.char as nac


def keyboard_typo(
    texts: list[str] | str,
    **kwargs,
) -> list[str]:
    '''Generate keyboard typo perturbed texts for augmentation.

    Args:
        texts: List of texts to be augmented.
        **kwargs: Arbitrary keyword arguments, which will be passed to
            :class:`nlpaug.augmenter.char.keyboard.KeyboardAug`. See the
            documentation for more details.

    Returns:
        A list of perturbed texts.
    '''
    aug = nac.KeyboardAug()
    return aug.augment(texts, **kwargs)
