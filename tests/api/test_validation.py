import pytest
import torch

from fedot import Fedot
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.validation.errors import FedotValidationError


def _tensor_metric_data() -> TensorData:
    return TensorData(
        state=StateEnum.FIT,
        features=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        target=torch.arange(5),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
    )


def test_get_metrics_tensordata_validates_fitted_pipeline():
    model = Fedot.__new__(Fedot)
    model.current_pipeline = None
    model.metrics = ['rmse']
    model._is_in_sample_prediction = False

    with pytest.raises(FedotValidationError, match='Pipeline is not fitted yet'):
        Fedot.get_metrics(model, tensor_data=_tensor_metric_data())
