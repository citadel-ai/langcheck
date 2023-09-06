import pytest

from langcheck.eval.en import semantic_sim

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


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [(["The CAT sat on the MAT."], ["The cat sat on the mat."])])
def test_semantic_sim_case_sensitivity(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.9 <= semantic_sim_value <= 1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [(["The cat sat on the mat."], ["I like to eat ice cream."])])
def test_semantic_sim_not_similar(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.0 <= semantic_sim_value <= 0.1