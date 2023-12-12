import random

from chikkarpy import Chikkar
from chikkarpy.dictionarylib import Dictionary as chikkardict
from sudachipy import Dictionary

_SudachiDict = Dictionary()


def synonym(
    instances: list[str] | str,
    *,
    num_perturbations: int = 1,
    **kwargs,
) -> list[str]:
    '''Applies a text perturbation to each string in instances (usually a list
    of prompts) where some words are replaced with synonyms.

    Args:
        instances: A single string or a list of strings to be augmented.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances
        aug_p: Percentage of words which will be augmented. Defaults to `0.1`.
        aug_max: Maximum number of words which will be augmented. Defaults to
            `None`.

    .. note::
        Any argument that can be passed to
        `nlpaug.augmenter.word.SynonymAug
        <https://nlpaug.readthedocs.io/en/latest/_modules/nlpaug/augmenter/word/synonym.html>`_
        is acceptable. Some of the more useful ones from the `nlpaug`
        documention are listed below:

          - ``aug_p`` (float): Percentage of words which will be augmented.

        Note that the default values for these arguments may be different from
        the ``nlpaug`` defaults.

    Returns:
        A list of perturbed instances.
    '''
    chikkar = Chikkar()
    chikkar.add_dictionary(chikkardict())
    sudachi_tokenizer = _SudachiDict.create()

    kwargs["aug_p"] = kwargs.get("aug_p", 0.8)

    instances = [instances] if isinstance(instances, str) else instances
    perturbed_instances = []
    for instance in instances:
        tokens = sudachi_tokenizer.tokenize(instance)
        for _ in range(num_perturbations):
            perturbed_instance = ""
            for token in tokens:
                synonym = token.surface()
                if (synonyms := chikkar.find(token.normalized_form())
                   ) and random.random() < kwargs["aug_p"]:
                    while synonym == token.surface():
                        synonym = random.choice(synonyms)
                perturbed_instance += synonym
            perturbed_instances.append(perturbed_instance)
    return perturbed_instances
