import pytest

from langcheck.eval.en import factual_consistency

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [(['Tokyo is the capital of Japan.', 'The Earth is flat.'
      ], ["Tokyo is Japan's capital city.", 'The Earth is round.'])])
def test_factual_consistency(generated_outputs, sources):
    eval_value = factual_consistency(generated_outputs, sources)
    factual_consistency_high = eval_value.metric_values[0]
    assert 0.9 <= factual_consistency_high <= 1
    factual_consistency_low = eval_value.metric_values[1]
    assert 0.0 <= factual_consistency_low <= 0.1
