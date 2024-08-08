from __future__ import annotations

from .._common._payload_splitting import payload_splitting_common


def payload_splitting(
    instances: list[str] | str,
    *,
    num_perturbations: int = 1,
    seed: int | None = None,
) -> list[str]:
    """Applies payload splitting augmentation to each string in instances.

    Ref: https://arxiv.org/pdf/2302.05733

    Args:
        instances: A single string or a list of strings to be augmented.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances. Should be equal to or less than the number
            of templates.
        seed: The seed for the random number generator. You can fix the seed to
            deterministically choose the indices to split the instances.

    Returns:
        A list of perturbed instances.
    """

    return payload_splitting_common(
        instances,
        "en",
        num_perturbations=num_perturbations,
        seed=seed,
    )
