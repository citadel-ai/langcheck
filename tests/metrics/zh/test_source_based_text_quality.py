import pytest

from langcheck.metrics.zh import factual_consistency
from tests.utils import MockEvalClient

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize("generated_outputs,sources", [
    ("北京是中国的首都。", "中国的首都是北京"),
    pytest.param("地球围绕着太阳转动。", "太阳是太阳系的中心。", marks=pytest.mark.xfail),
    pytest.param(["飞机在是一种空中交通工具。", "太阳围绕着地球转动。"], ["飞机在可以在天上飞。", "太阳是太阳系的中心。"]),
])
def test_factual_consistency(generated_outputs, sources):
    metric_value = factual_consistency(generated_outputs, sources)
    factual_consistency_high = metric_value.metric_values[0]
    assert factual_consistency_high is not None
    assert 0.8 <= factual_consistency_high <= 1
    if len(metric_value.metric_values) == 2:
        factual_consistency_low = metric_value.metric_values[1]
        assert factual_consistency_low is not None
        assert 0.0 <= factual_consistency_low <= 0.1


@pytest.mark.parametrize("generated_outputs,sources",
                         [("北京是中国的首都。", "中国的首都是北京"),
                          (["北京是中国的首都。"], ["中国的首都是北京"])])
def test_factual_consistency_eval_client(generated_outputs, sources):
    eval_client = MockEvalClient()
    metric_value = factual_consistency(generated_outputs,
                                       sources,
                                       eval_model=eval_client)
    # MockEvalClient without any argument returns None
    assert metric_value.metric_values[0] is None

    factual_consistency_assessment_to_score = {
        "Fully Consistent": 1.0,
        "Partially Consistent": 0.5,
        "Not Consistent": 0.0
    }

    for option in factual_consistency_assessment_to_score:
        eval_client = MockEvalClient(option)
        metric_value = factual_consistency(generated_outputs,
                                           sources,
                                           eval_model=eval_client)
        assert metric_value == factual_consistency_assessment_to_score[option]
