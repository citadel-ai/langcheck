from __future__ import annotations

import random
from contextlib import suppress

from chikkarpy import Chikkar
from chikkarpy.dictionarylib import Dictionary as chikkardict

with suppress(ImportError):
    from sudachipy import Dictionary  # type: ignore[reportMissingImports]


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
        aug_p: Percentage of words with synonymous which will be augmented.
            Defaults to `0.8`.

    Returns:
        A list of perturbed instances.


    .. note::
        This function requires `sudachidict_core` and `sudachipy` to be
        installed in your environment.
        Please refer to the `official instructions <https://github.com/
        WorksApplications/SudachiPy?tab=readme-ov-file#setup>`_ to install them.

    '''
    _SudachiDict = Dictionary()  # type: ignore[reportUnboundVariable]

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
                    synonym = random.choice(synonyms)
                perturbed_instance += synonym
            perturbed_instances.append(perturbed_instance)
    return perturbed_instances
