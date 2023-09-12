import pytest

from langcheck.eval.ja import semantic_sim
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が座っています。"], ["猫が座っています。"])])
def test_semantic_sim_identical(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.99 <= semantic_sim_value <= 1


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が座っています。"], ["ネコがすわっています。"])])
def test_semantic_sim_character_sensitivity(generated_outputs,
                                            reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.5 <= semantic_sim_value <= 1


@pytest.mark.parametrize('generated_outputs,reference_outputs',
                         [(["猫が座っています。"], ["僕はアイスクリームを食べるのが好きです。"])])
def test_semantic_sim_not_similar(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs, reference_outputs)
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.0 <= semantic_sim_value <= 0.5