from typing import Tuple

import pytest
import torch
from sklearn.datasets import make_classification

from fedot import create_data
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.strategies.models.cuml import CuMLClassificationStrategy
from fedot.core.operations.strategies.models.cuml import CuMLModelStrategy


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='cuML CUDA runtime is unavailable')
SVC = pytest.importorskip('cuml.svm').SVC


def get_synthetic_data() -> Tuple[TensorData, TensorData]:
    features, target = make_classification(n_samples=11000, random_state=42)
    train_data = create_data(features[:10000], target=target[:10000], task='classification')
    test_data = create_data(features[10000:], target=target[10000:], from_data=train_data)

    return train_data, test_data


def test_gpu_evaluation_strategy_fit():
    train_data, _ = get_synthetic_data()
    strategy = CuMLModelStrategy(operation_type='svc',
                                      params=dict(kernel='rbf', C=10, gamma=1, cache_size=2000, probability=True))

    operation = strategy.fit(train_data)

    assert isinstance(operation, SVC)


def test_gpu_evaluation_strategy_predict():
    train_data, test_data = get_synthetic_data()
    strategy = CuMLClassificationStrategy(operation_type='svc',
                                          params=dict(kernel='rbf', C=10, gamma=1, cache_size=2000, probability=True))

    operation = strategy.fit(train_data)
    prediction = strategy.predict(operation, test_data)

    assert isinstance(prediction, TensorData)
