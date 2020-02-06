import numpy as np

from core.data import Data
from core.model import LogRegression
from core.repository.dataset_types import NumericalDataTypesEnum


def test_log_regression_types_correct():
    log_reg = LogRegression()

    assert log_reg.input_type is NumericalDataTypesEnum.table
    assert log_reg.output_type is NumericalDataTypesEnum.vector


# TODO: refactor this test
def test_log_regression_fit_correct():
    log_reg = LogRegression()
    x = 10.0 * np.random.rand(100, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= 0.5 else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = Data(features=x, target=classes)

    log_reg.fit(data=data)
    predicted = log_reg.predict(data=data)
    print(np.array_equal(classes, predicted))
