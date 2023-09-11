import pytest

from langcheck.eval.en import (
    flesch_kincaid_grade, flesch_reading_ease, fluency, sentiment, toxicity
)
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs',
    [["I'm fine!", "I'm feeling pretty bad today."], ['Hello']]
)
def test_sentiment(generated_outputs):
    eval_value = sentiment(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize(
    'generated_outputs',
    [["I'd appreciate your help.", 'Today I eats very much apples good.'],
     ['cat']]
)
def test_fluency(generated_outputs):
    eval_value = fluency(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize(
    'generated_outputs',
    [['I hate you. Shut your mouth!', 'Thank you so much for coming today!!'],
     ['foo bar']]
)
def test_toxicity(generated_outputs):
    eval_value = toxicity(generated_outputs)
    assert all(0 <= v <= 1 for v in eval_value.metric_values)


@pytest.mark.parametrize(
    'generated_outputs,metric_values', [
        ([
            '''
    My Friend. Welcome to the Carpathians. I am anxiously expecting you.
    Sleep well to-night. At three to-morrow the diligence will start for Bukovina;
    a place on it is kept for you.
    '''
        ], [75.00651612903226]),
        ([
            '''
    How slowly the time passes here, encompassed as I am by frost and snow!
    Yet a second step is taken towards my enterprise.
    '''
        ], [77.45815217391308]),
    ]
)
def test_flesch_reading_ease(generated_outputs, metric_values):
    eval_value = flesch_reading_ease(generated_outputs)
    assert is_close(eval_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,metric_values', [
        ([
            '''
    My Friend. Welcome to the Carpathians. I am anxiously expecting you.
    Sleep well to-night. At three to-morrow the diligence will start for Bukovina;
    a place on it is kept for you.
    '''
        ], [4.33767741935484]),
        ([
            '''
    How slowly the time passes here, encompassed as I am by frost and snow!
    Yet a second step is taken towards my enterprise.
    '''
        ], [5.312391304347827]),
    ]
)
def test_flesch_kincaid_grade(generated_outputs, metric_values):
    eval_value = flesch_kincaid_grade(generated_outputs)
    assert is_close(eval_value.metric_values, metric_values)
