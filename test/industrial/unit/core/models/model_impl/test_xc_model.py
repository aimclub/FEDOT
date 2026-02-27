import warnings

import numpy as np
import pytest
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from matplotlib import get_backend, pyplot as plt

from fedot_ind.core.models.nn.network_impl.feature_extraction.explainable_convolution_model import XCModel


@pytest.fixture(scope='session')
def input_data():
    features_train, features_test = np.random.randn(100, 3, 50, 50), np.random.randn(100, 3, 50, 50)
    target_train, target_test = np.random.randint(0, 2, 100), np.random.randint(0, 2, 100)
    train_input = init_input_data(features_train, target_train)
    test_input = init_input_data(features_test, target_test)
    return train_input, test_input


def test_xcm_model(input_data):
    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
    train, test = input_data
    model = XCModel(params=OperationParameters(num_classes=2,
                                               epochs=10,
                                               batch_size=64))
    model._fit_model(train)
    predict = model._predict_model(test.features)

    model.model_for_inference.explain(train)

    assert model is not None
    assert predict.predict.shape[0] == test.target.shape[0]
