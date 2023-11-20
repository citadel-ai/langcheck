from __future__ import annotations

from nlpaug.augmenter.char.ocr import OcrAug


def ocr_typo(
    instances: list[str] | str,
    *,
    num_perturbations: int = 1,
    **kwargs,
) -> list[str]:
    '''Applies an OCR typo text perturbation to each string in instances
    (usually a list of prompts).

    Args:
        instances: A single string or a list of strings to be augmented.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances
        aug_char_p: Percentage of characters (per token) that will be augmented.
            Defaults to `0.1`.
        aug_char_max: Maximum number of characters which will be augmented.
            Defaults to `None`.
        aug_word_max: Maximum number of words which will be augmented. Defaults
            to `None`.

    .. note::
        Any argument that can be passed to
        `nlpaug.augmenter.char.ocr.OcrAug
        <https://nlpaug.readthedocs.io/en/latest/augmenter/char/ocr.html#nlpaug.augmenter.char.ocr.OcrAug>`_
        is acceptable. Some of the more useful ones from the `nlpaug`
        documentation are listed below:

          - ``aug_char_p`` (float): Percentage of characters (per token) that
            will be augmented.
          - ``aug_char_min`` (int): Minimum number of characters that will be
            augmented.
          - ``aug_char_max`` (int): Maximum number of characters that will be
            augmented.
          - ``aug_word_p`` (float): Percentage of words that will be augmented.
          - ``aug_word_min`` (int): Minimum number of words that will be
            augmented.
          - ``aug_word_max`` (int): Maximum number of words that will be
            augmented.

        Note that the default values for these arguments may be different from
        the ``nlpaug`` defaults.

    Returns:
        A list of perturbed instances.
    '''

    kwargs["aug_char_p"] = kwargs.get("aug_char_p", 0.1)
    kwargs["aug_char_max"] = kwargs.get("aug_char_max")
    kwargs["aug_word_max"] = kwargs.get("aug_word_max")

    instances = [instances] if isinstance(instances, str) else instances
    perturbed_instances = []
    aug = OcrAug(**kwargs)
    for instance in instances:
        for _ in range(num_perturbations):
            perturbed_instances += aug.augment(instance)
    return perturbed_instances
