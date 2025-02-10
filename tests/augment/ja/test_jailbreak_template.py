from __future__ import annotations

import pytest

from langcheck.augment.ja import jailbreak_template


def test_invalid_template():
    with pytest.raises(ValueError):
        jailbreak_template("こんにちは", templates=["invalid_template"])


def test_invalid_num_perturbations():
    with pytest.raises(ValueError):
        jailbreak_template(
            "こんにちは", ["chatgpt_good_vs_evil", "john"], num_perturbations=3
        )


@pytest.mark.parametrize(
    "instances,templates,num_perturbations",
    [
        ("こんにちは", ["basic"], 1),
        ("こんにちは", None, 3),
        (
            ["こんにちは!", "こんにちは?"],
            [
                "chatgpt_good_vs_evil",
                "john",
            ],
            2,
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

def test_custom_jailbreak_template(tmp_path):
    template_file = tmp_path / "custom_template.j2"
    template_file.write_text("Custom jailbreak: {}")
    custom_templates = [("custom", str(template_file))]
    results = jailbreak_template("Hello, world", templates=["custom"], custom_templates=custom_templates)
    assert len(results) == 1
    assert "Custom jailbreak:" in results[0]
