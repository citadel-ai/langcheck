import tempfile

import pytest

from langcheck.metrics import custom_evaluator, custom_pairwise_evaluator
from tests.utils import MockEvalClient

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    "generated_outputs,sources,additional",
    [
        (
            "Tokyo is the capital of Japan.",
            "Tokyo is Japan's capital city.",
            "Additional Information",
        ),
        (
            ["Tokyo is the capital of Japan.", "The Earth is flat."],
            ["Tokyo is Japan's capital city.", "The Earth is round."],
            ["Additional Information", "Additional Information"],
        ),
    ],
)
def test_custom_evaluator(generated_outputs, sources, additional):
    metric_name = "factual consistency"
    score_map = {
        "Fully Consistent": 1.0,
        "Partially Consistent": 0.5,
        "Not Consistent": 0.0,
    }
    # Create a temporary Jinja2 template file
    with tempfile.NamedTemporaryFile(suffix=".j2") as temp:
        temp.write(b"""
        Evaluate the submitted claim's factual consistency with the source:
        [Source]: {{ src }}
        [Submission]: {{ gen_output }}
        [Additional Information]: {{ additional_info }}
        """)
        template_path = temp.name

        for option in score_map:
            eval_client = MockEvalClient(option)
            metric_value = custom_evaluator(
                generated_outputs,
                None,
                sources,
                None,
                eval_client,
                metric_name,
                score_map,
                template_path,
                "en",
                additional_inputs={"additional": additional},
                additional_input_name_to_prompt_var_mapping={
                    "additional": "additional_info",
                },
            )
            assert metric_value == score_map[option]


@pytest.mark.parametrize(
    "generated_outputs,sources,additional",
    [
        (
            "Tokyo is the capital of Japan.",
            "Tokyo is Japan's capital city.",
            "Additional Information",
        ),
        (
            ["Tokyo is the capital of Japan.", "The Earth is flat."],
            ["Tokyo is Japan's capital city.", "The Earth is round."],
            ["Additional Information", "Additional Information"],
        ),
    ],
)
def test_custom_evaluator_with_template_str(
    generated_outputs, sources, additional
):
    metric_name = "factual consistency"
    score_map = {
        "Fully Consistent": 1.0,
        "Partially Consistent": 0.5,
        "Not Consistent": 0.0,
    }

    template_str = """
    Evaluate the submitted claim's factual consistency with the source:
    [Source]: {{ src }}
    [Submission]: {{ gen_output }}
    [Additional Information]: {{ additional_info }}
    """
    for option in score_map:
        eval_client = MockEvalClient(option)
        metric_value = custom_evaluator(
            generated_outputs,
            None,
            sources,
            None,
            eval_client,
            metric_name,
            score_map,
            None,
            "en",
            template_str=template_str,
            additional_inputs={"additional": additional},
            additional_input_name_to_prompt_var_mapping={
                "additional": "additional_info",
            },
        )
        assert metric_value == score_map[option]


@pytest.mark.parametrize(
    "generated_outputs_a,generated_outputs_b,prompts,sources_a,sources_b,reference_outputs",
    [
        (
            "Tokyo is Japan's capital city.",
            "New York is Japan's capital city.",
            "What is the capital of Japan?",
            None,
            None,
            None,
        ),
        (
            "Tokyo is Japan's capital city.",
            "New York is Japan's capital city.",
            "What is the capital of Japan?",
            None,
            None,
            "Tokyo",
        ),
        (
            "Tokyo is Japan's capital city.",
            "New York is Japan's capital city.",
            "What is the capital of Japan?",
            "Capital of Japan = Tokyo",
            None,
            None,
        ),
        (
            "Tokyo is Japan's capital city.",
            "New York is Japan's capital city.",
            "What is the capital of Japan?",
            "Capital of Japan = Tokyo",
            "Capital of Japan = Tokyo",
            None,
        ),
        (
            "Tokyo is Japan's capital city.",
            "New York is Japan's capital city.",
            "What is the capital of Japan?",
            "Capital of Japan = Tokyo",
            "Capital of Japan = Tokyo",
            "Tokyo",
        ),
    ],
)
def test_custom_pairwise_evaluator(
    generated_outputs_a,
    generated_outputs_b,
    prompts,
    sources_a,
    sources_b,
    reference_outputs,
):
    """Test the custom pairwise evaluator.

    Test 1: No sources or reference outputs are provided.
    Test 2: Reference output is provided.
    Test 3: Source A is provided.
    Test 4: Both Source A and Source B are provided.
    Test 5: Source A, Source B, and the reference output are provided.
    """
    metric_name = "pairwise factual correctness"
    score_map = {"Response B": 1.0, "Tie": 0.5, "Response A": 0.0}
    # Create a temporary Jinja2 template file
    with tempfile.NamedTemporaryFile(suffix=".j2") as temp:
        temp.write(b"""
        Determine which of the two responses is more factually correct:
        [User Query]: {{ user_query }}
        {% if src_a is not none %}
        [Source A]: {{ src_a }}
        {% endif %}
        {% if src_b is not none %}
        [Source B]: {{ src_b }}
        {% endif %}
        {% if ref_output is not none %}
        [Ideal Response]: {{ ref_output }}
        {% endif %}
        [Response A]: {{ gen_output_a }}
        [Response B]: {{ gen_output_b }}
        [Additional Information]: {{ additional_info }}
        [Additional Pairwise Information A]: {{ additional_pairwise_info_a }}
        [Additional Pairwise Information B]: {{ additional_pairwise_info_b }}
        """)
        template_path = temp.name

        # Return "Tie" so that it passes the consistency check
        eval_client = MockEvalClient("Tie")
        metric_value = custom_pairwise_evaluator(
            generated_outputs_a,
            generated_outputs_b,
            prompts,
            sources_a,
            sources_b,
            reference_outputs,
            eval_client,
            metric_name,
            score_map,
            template_path,
            "en",
            additional_inputs={
                "additional": "Additional Information",
                "additional_pairwise": (
                    "Additional Pairwise Information A",
                    "Additional Pairwise Information B",
                ),
            },
            additional_input_name_to_prompt_var_mapping={
                "additional": "additional_info",
                "additional_pairwise": "additional_pairwise_info",
            },
        )

        # MockEvalClient returns 0.5 for Tie
        assert metric_value == 0.5
