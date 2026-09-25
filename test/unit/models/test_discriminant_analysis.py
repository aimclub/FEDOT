from unittest.mock import Mock, patch

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.models.discriminant_analysis import QDAImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def classification_data(samples=10, features=2):
    return InputData(idx=np.arange(samples),
                     features=np.zeros((samples, features)),
                     target=np.arange(samples) % 2,
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)


def test_qda_regularizes_singular_covariance():
    implementation = QDAImplementation(OperationParameters())
    implementation.model = Mock()
    implementation.model.fit.side_effect = np.linalg.LinAlgError

    implementation.fit(classification_data())

    assert implementation.params.get('reg_param') == implementation._MIN_REG_PARAM
    assert implementation.model.reg_param == implementation._MIN_REG_PARAM


def test_qda_uses_eigen_solver_when_regularization_is_not_enough():
    implementation = QDAImplementation(OperationParameters())
    implementation.model = Mock()
    implementation.model.fit.side_effect = np.linalg.LinAlgError
    regularized_model = Mock()
    regularized_model.fit.side_effect = np.linalg.LinAlgError
    regularized_model.get_params.return_value = {'solver': 'svd'}
    eigen_model = Mock()

    with patch('fedot.core.operations.evaluation.operation_implementations.models.discriminant_analysis.'
               'QuadraticDiscriminantAnalysis', side_effect=[regularized_model, eigen_model]):
        fitted_model = implementation.fit(classification_data(samples=6, features=10))

    assert fitted_model is eigen_model
    assert implementation.params.get('solver') == 'eigen'
    assert implementation.params.get('shrinkage') == 'auto'
