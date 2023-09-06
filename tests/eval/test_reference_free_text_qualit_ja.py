import pytest

from langcheck.eval.ja import sentiment
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs', [["私は嬉しい", "私は悲しい"], ['こんにちは']])
def test_sentiment(generated_outputs):
    eval_value = sentiment(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)
