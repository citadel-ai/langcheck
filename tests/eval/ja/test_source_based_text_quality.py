import pytest

from langcheck.eval.ja import factual_consistency

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [(['東京は日本の首都です。', '地球は平面です。'], ['東京は日本の首都です。', '地球は球体です。'])])
def test_factual_consistency(generated_outputs, sources):
    eval_value = factual_consistency(generated_outputs, sources)
    factual_consistency_high = eval_value.metric_values[0]
    assert 0.9 <= factual_consistency_high <= 1
    factual_consistency_low = eval_value.metric_values[1]
    assert 0.0 <= factual_consistency_low <= 0.1
