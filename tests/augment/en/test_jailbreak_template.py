from __future__ import annotations

import pytest

from langcheck.augment.en import jailbreak_template


def test_invalid_template():
    with pytest.raises(ValueError):
        jailbreak_template("Hello, world", templates=["invalid_template"])


def test_invalid_num_perturbations():
    with pytest.raises(ValueError):
        jailbreak_template(
            "Hello, world", ["chatgpt_dan", "john"], num_perturbations=3
        )


@pytest.mark.parametrize(
    "instances,templates,num_perturbations",
    [
        ("Hello, world", ["basic"], 1),
        ("Hello, world", None, 5),
        (
            ["Hello, world!", "Hello, world?"],
            [
                "chatgpt_dan",
                "chatgpt_good_vs_evil",
                "john",
                "universal_adversarial_suffix",
            ],
            4,
        ),
    ],
)
def test_jailbreak_template(
    instances: str | list[str],
    templates: list[str] | None,
    num_perturbations: int,
):
    results = jailbreak_template(
        instances, templates=templates, num_perturbations=num_perturbations
    )
    if isinstance(instances, str):
        instances = [instances]

    for i in range(len(instances)):
        for j in range(num_perturbations):
            idx = i * num_perturbations + j
            # assert that `idx`th template includes the original instance
            assert instances[i] in results[idx]
