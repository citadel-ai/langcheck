from __future__ import annotations

from nlpaug.augmenter.char.keyboard import KeyboardAug


def keyboard_typo(
    instances: list[str] | str,
    *,
    num_perturbations: int = 1,
    **kwargs,
) -> list[str]:
    '''Applies a keyboard typo text perturbation to each string in instances
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
        include_special_char: Allow special characters to be augmented. Defaults
            to `False`.
        include_numeric: Allow numeric characters to be augmented. Defaults to
            `False`.

    .. note::
        Any argument that can be passed to
        `nlpaug.augmenter.char.keyboard.KeyboardAug
        <https://nlpaug.readthedocs.io/en/latest/augmenter/char/keyboard.html#nlpaug.augmenter.char.keyboard.KeyboardAug>`_
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

        Note that the default values for these arguments may be different from
        the ``nlpaug`` defaults.

    Returns:
        A list of perturbed instances.
    '''

    kwargs["aug_char_p"] = kwargs.get("aug_char_p", 0.1)
    kwargs["aug_char_max"] = kwargs.get("aug_char_max")
    kwargs["aug_word_max"] = kwargs.get("aug_word_max")
    kwargs["include_special_char"] = kwargs.get("include_special_char", False)
    kwargs["include_numeric"] = kwargs.get("include_numeric", False)

    instances = [instances] if isinstance(instances, str) else instances
    perturbed_instances = []
    aug = KeyboardAug(**kwargs)
    for instance in instances:
        for _ in range(num_perturbations):
            perturbed_instances += aug.augment(instance)
    return perturbed_instances
