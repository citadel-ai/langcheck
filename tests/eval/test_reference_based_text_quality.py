import pytest

from langcheck.eval import semantic_sim

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [(["The cat sat on the mat."], ["The cat sat on the mat."])])
def test_semantic_sim_identical(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.99 <= semantic_sim_value <= 1