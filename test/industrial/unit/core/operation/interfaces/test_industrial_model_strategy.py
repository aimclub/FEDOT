import numpy as np
import pandas as pd
import pytest

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.repository.model_repository import AtomizedModel

NN_MODELS = AtomizedModel.NEURAL_MODEL.value


@pytest.fixture()
def torch_classification_data():
    features = np.random.rand(10, 10)
    target = np.random.randint(2, size=10)
    return init_input_data(
        X=pd.DataFrame(features),
        y=target,
        task='classification')


@pytest.fixture()
def torch_regression_data():
    features = np.random.rand(10, 10)
    target = np.random.rand(10, 1)
    return init_input_data(
        X=pd.DataFrame(features),
        y=target,
        task='regression')

#
# # TODO: add more models from NN_MODELS
# @pytest.mark.parametrize('model_name', ['omniscale_model', 'inception_model'])
# def test_fedot_nn_classification_strategy(torch_classification_data, model_name):
#     params = OperationParameters()
#     params._parameters.update({'problem': 'regression', 'timeout': 0.1})
#     strategy = FedotNNClassificationStrategy(operation_type=model_name,
#                                              params=params)
#     trained_operation = strategy.fit(torch_classification_data)
#
#     assert trained_operation is not None
#
#
# # TODO: add more models from NN_MODELS
# @pytest.mark.parametrize('model_name', ['inception_model'])
# def test_fedot_nn_regression_strategy(torch_regression_data, model_name):
#     params = OperationParameters()
#     params._parameters.update({'problem': 'regression', 'timeout': 0.1})
#     strategy = FedotNNRegressionStrategy(operation_type=model_name,
#                                          params=params)
#     trained_operation = strategy.fit(torch_regression_data)
#
#     assert trained_operation is not None
