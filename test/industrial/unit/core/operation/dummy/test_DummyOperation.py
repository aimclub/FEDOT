import pandas as pd

from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
import pytest
import numpy as np
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data


@pytest.fixture()
def input_data():
    features = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))
    target = np.array([1, 2])
    return init_input_data(features, target)


def test_dummy_operation(input_data):
    operation = DummyOperation(dict())
    operation.fit(input_data)
    predict = operation.transform(input_data)
    assert predict.features.shape == (2, 3)
    assert predict.target.shape == (2, 1)
