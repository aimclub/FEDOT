from fedot_ind.tools.example_utils import evaluate_metric


def test_evaluate_metric():
    target = [0, 1, 0, 1, 0, 1]
    prediction = [0, 1, 0, 1, 0, 1]
    metric = evaluate_metric(target, prediction)
    assert metric == 1.0
