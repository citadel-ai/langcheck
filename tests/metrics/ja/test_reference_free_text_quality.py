import pytest

from langcheck.metrics.ja import (answer_relevance, fluency, sentiment,
                                  tateishi_ono_yamada_reading_ease, toxicity)
from tests.utils import MockEvalClient, is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize('generated_outputs', [
    'こんにちは',
    ['こんにちは'],
    ["私は嬉しい", "私は悲しい"],
])
def test_sentiment(generated_outputs):
    metric_value = sentiment(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs', ["私は嬉しい", ["私は嬉しい"]])
def test_sentiment_eval_client(generated_outputs):
    eval_client = MockEvalClient()
    metric_value = sentiment(generated_outputs, eval_model=eval_client)
    # MockEvalClient always returns 0.5
    assert metric_value == 0.5


@pytest.mark.parametrize('generated_outputs',
                         [['馬鹿', '今日はりんごを食べました。'], ['猫'], '猫'])
def test_toxicity(generated_outputs):
    metric_value = toxicity(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs', ['アホ', ['アホ']])
def test_toxicity_eval_client(generated_outputs):
    eval_client = MockEvalClient(return_value=1.)
    metric_value = toxicity(generated_outputs, eval_model=eval_client)
    # MockEvalClient always returns 1.0
    assert metric_value == 1.0


@pytest.mark.parametrize(
    'generated_outputs',
    [['ご機嫌いかがですか？私はとても元気です。', '機嫌いかが？私はとても元気人です。'], ['猫'], '猫'])
def test_fluency(generated_outputs):
    metric_value = fluency(generated_outputs)
    assert 0 <= metric_value <= 1


@pytest.mark.parametrize('generated_outputs',
                         ['ご機嫌いかがですか？私はとても元気です。', ['ご機嫌いかがですか？私はとても元気です。']])
def test_fluency_eval_client(generated_outputs):
    eval_client = MockEvalClient(return_value=1.)
    metric_value = fluency(generated_outputs, eval_model=eval_client)
    # MockEvalClient always returns 1.0
    assert metric_value == 1.0


@pytest.mark.parametrize('generated_outputs,metric_values', [
    ('吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。',
     [73.499359]),
    (['吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。'
     ], [73.499359]),
    (['日本語自然言語処理には、日本語独特の技法が多数必要で、欧米系言語と比較して難易度が高い。'], [24.7875]),
])
def test_tateishi_ono_yamada_reading_ease(generated_outputs, metric_values):
    metric_value = tateishi_ono_yamada_reading_ease(generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize('generated_outputs,prompts',
                         [('東京は日本の首都です。', '日本の首都は何ですか？'),
                          (['東京は日本の首都です。'], ['日本の首都は何ですか？'])])
def test_answer_relevance_eval_client(generated_outputs, prompts):
    eval_client = MockEvalClient(return_value=1.)
    metric_value = answer_relevance(generated_outputs,
                                    prompts,
                                    eval_model=eval_client)
    # MockEvalClient always returns 1.0
    assert metric_value == 1.0
