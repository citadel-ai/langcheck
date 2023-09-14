from langcheck.eval import is_float


def test_eval_value_comparisons():
    eval_value = is_float(['1', '2', '3'], min=0, max=5)

    # Test all comparisons: ==, !=, <, >, <=, >=
    assert eval_value == 1
    assert (eval_value == 1).all()
    assert all((eval_value == 1).threshold_results)
    assert (eval_value == 1).pass_rate == 1
    assert not eval_value != 1
    assert not (eval_value != 1).any()
    assert not any((eval_value != 1).threshold_results)
    assert (eval_value != 1).pass_rate == 0
    assert eval_value < 1.5
    assert (eval_value < 1.5).all()
    assert all((eval_value < 1.5).threshold_results)
    assert (eval_value < 1.5).pass_rate == 1
    assert eval_value > 0.5
    assert (eval_value > 0.5).all()
    assert all((eval_value > 0.5).threshold_results)
    assert (eval_value > 0.5).pass_rate == 1
    assert eval_value <= 1
    assert (eval_value <= 1).all()
    assert all((eval_value <= 1).threshold_results)
    assert (eval_value <= 1).pass_rate == 1
    assert eval_value >= 0
    assert (eval_value >= 0).all()
    assert all((eval_value >= 0).threshold_results)
    assert (eval_value >= 0).pass_rate == 1
