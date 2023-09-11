import pytest

from langcheck.eval import exact_match

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [(["The cat sat on the mat."], ["The cat sat on the mat."])]
)
def test_exact_match(generated_outputs, reference_outputs):
    eval_value = exact_match(generated_outputs, reference_outputs)
    exact_match_value = eval_value.metric_values[0]
    assert exact_match_value == 1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [(["The CAT sat on the MAT."], ["The cat sat on the mat."])]
)
def test_not_exact_match(generated_outputs, reference_outputs):
    eval_value = exact_match(generated_outputs, reference_outputs)
    exact_match_value = eval_value.metric_values[0]
    assert exact_match_value == 0
