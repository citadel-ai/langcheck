import tempfile

import pytest
from langcheck.metrics import custom_evaluator

from tests.utils import MockEvalClient

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    "generated_outputs,sources",
    [("Tokyo is the capital of Japan.", "Tokyo is Japan's capital city."),
     (["Tokyo is the capital of Japan.", "The Earth is flat."
      ], ["Tokyo is Japan's capital city.", "The Earth is round."])])
def test_custom_evaluator(generated_outputs, sources):
    metric_name = "factual consistency"
    score_map = {
        "Fully Consistent": 1.0,
        "Partially Consistent": 0.5,
        "Not Consistent": 0.0
    }
    # Create a temporary Jinja2 template file
    with tempfile.NamedTemporaryFile(suffix=".j2") as temp:
        temp.write(b"""
        Evaluate the submitted claim's factual consistency with the source:
        [Source]: {{ src }}
        [Submission]: {{ gen_output }}
        """)
        template_path = temp.name

        for option in score_map:
            eval_client = MockEvalClient(option)
            metric_value = custom_evaluator(generated_outputs, None, sources,
                                            None, eval_client, metric_name,
                                            score_map, template_path, "en")
            assert metric_value == score_map[option]
