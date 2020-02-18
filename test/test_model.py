import numpy as np
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from core.data import Data
from core.model import LogRegression
from core.repository.dataset_types import NumericalDataTypesEnum


@pytest.fixture()
def log_function_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = Data(features=x, target=classes, idx=np.arange(0, len(x)))

    return data


def test_log_regression_types_correct():
    log_reg = LogRegression()

    assert log_reg.input_type is NumericalDataTypesEnum.table
    assert log_reg.output_type is NumericalDataTypesEnum.vector


def test_log_regression_fit_correct(log_function_dataset):
    data = log_function_dataset
    log_reg = LogRegression()

    log_reg.fit(data=data)
    predicted = log_reg.predict(data=data)

    train_to = int(len(predicted) * 0.8)

    roc_on_train = roc_auc(y_true=data.target[:train_to],
                           y_score=predicted[:train_to])
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold
