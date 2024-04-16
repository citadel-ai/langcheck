import pytest

from langcheck.metrics.ja import context_relevance, factual_consistency
from tests.utils import MockEvalClient

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,sources',
    [('東京は日本の首都です。', '東京は日本の首都です。'),
     (['東京は日本の首都です。', '地球は平面です。'], ['東京は日本の首都です。', '地球は球体です。'])])
def test_factual_consistency(generated_outputs, sources):
    metric_value = factual_consistency(generated_outputs, sources)
    factual_consistency_high = metric_value.metric_values[0]
    assert factual_consistency_high is not None
    assert 0.9 <= factual_consistency_high <= 1
    if len(metric_value.metric_values) == 2:
        factual_consistency_low = metric_value.metric_values[1]
        assert factual_consistency_low is not None
        assert 0.0 <= factual_consistency_low <= 0.1


@pytest.mark.parametrize('generated_outputs,sources',
                         [('東京は日本の首都です。', "東京は日本の首都です。"),
                          (['東京は日本の首都です。'], ["東京は日本の首都です。"])])
def test_factual_consistency_eval_client(generated_outputs, sources):
    eval_client = MockEvalClient(return_value=1.0)
    metric_value = factual_consistency(generated_outputs,
                                       sources,
                                       eval_model=eval_client)
    # MockEvalClient always returns 1.0
    assert metric_value == 1.0


@pytest.mark.parametrize('prompts,sources',
                         [('日本の首都は何ですか？', "東京は日本の首都です。"),
                          (['日本の首都は何ですか？'], ["東京は日本の首都です。"])])
def test_context_relevance_eval_client(prompts, sources):
    eval_client = MockEvalClient(return_value=1.0)
    metric_value = context_relevance(sources, prompts, eval_model=eval_client)
    # MockEvalClient always returns 1.0
    assert metric_value == 1.0
