import golem
import numpy as np
import pytest
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task

from fedot_ind.core.operation.interfaces.fedot_automl_evaluation_strategy import FedotAutoMLClassificationStrategy, \
    FedotAutoMLRegressionStrategy
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tests.unit.api.fixtures import get_data_by_task


def mock_message(self, msg: str, **kwargs):
    level = 40
    self.log(level, msg, **kwargs)


@pytest.mark.parametrize('task', ('classification', 'regression'))
def test_fedot_automl_strategy_fit_predict(task, monkeypatch):
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)
    repo = IndustrialModels()
    repo.setup_default_repository()
    (x_train, y_train), _ = get_data_by_task(task)
    x_train, y_train = x_train.values, y_train
    input_data = InputData(idx=np.arange(len(x_train)),
                           features=x_train,
                           target=y_train,
                           task=Task(TaskTypesEnum(task)),
                           data_type=DataTypesEnum.table)

    params = OperationParameters(problem=task, timeout=0.1, n_jobs=1)
    if task == 'classification':
        strategy = FedotAutoMLClassificationStrategy(operation_type='fedot_cls', params=params)
    elif task == 'regression':
        strategy = FedotAutoMLRegressionStrategy(operation_type='fedot_regr', params=params)
    else:
        return

    trained_operation = strategy.fit(input_data)

    predict = strategy.predict(trained_operation, input_data)
    predict_for_fit = strategy.predict_for_fit(trained_operation, input_data)
    repo.setup_repository()

    assert predict.predict is not None
    assert predict_for_fit.predict is not None
    assert strategy.operation_impl is not None
    assert trained_operation is not None
