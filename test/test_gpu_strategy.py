from typing import Tuple

from fedot.core.data.data import InputData, OutputData

from test.unit.models.test_split_train_test import get_synthetic_input_data
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy
from fedot.core.operations.evaluation.gpu.classification import CuMLClassificationStrategy
from cuml.svm import SVC


def get_synthetic_data() -> Tuple[InputData, InputData]:
    train_data = get_synthetic_input_data(10000)
    test_data = get_synthetic_input_data(1000)

    return train_data, test_data


def test_gpu_evaluation_strategy_fit():
    train_data, _ = get_synthetic_data()
    strategy = CuMLEvaluationStrategy(operation_type='svc',
                                      params=dict(kernel='rbf', C=10, gamma=1, cache_size=2000, probability=True))

    operation = strategy.fit(train_data)

    assert isinstance(operation, SVC)


def test_gpu_evaluation_strategy_predict():
    train_data, test_data = get_synthetic_data()
    strategy = CuMLClassificationStrategy(operation_type='svc',
                                          params=dict(kernel='rbf', C=10, gamma=1, cache_size=2000, probability=True))

    operation = strategy.fit(train_data)
    prediction = strategy.predict(operation, test_data)

    assert isinstance(prediction, OutputData)
