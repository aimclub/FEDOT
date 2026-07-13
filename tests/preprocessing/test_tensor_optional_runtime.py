import numpy as np
import pytest

from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.core.architecture.preprocessing.ts_optional_service import OptionalTSService
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.service.tensor_optional_runtime import (
    TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE,
    get_optional_runtime_spec_for_tensor_data,
)
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum


def _tabular_tensor_data() -> TensorData:
    return TensorData(
        state=StateEnum.FIT,
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        idx=np.array([0, 1]),
        target=np.array([0, 1]),
    )


def _ts_tensor_data() -> TensorData:
    return TensorData(
        state=StateEnum.FIT,
        features=np.array([[1.0], [2.0]], dtype=np.float32),
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
        data_type=DataTypesEnum.ts,
        idx=np.array([0, 1]),
        target=np.array([1.0, 2.0]),
    )


@pytest.mark.unit
def test_tensor_optional_runtime_registry_shares_default_steps():
    tabular_steps = TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE[DataTypesEnum.tabular].default_steps
    ts_steps = TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE[DataTypesEnum.ts].default_steps

    assert tabular_steps is ts_steps
    assert tabular_steps == {
        PreprocessingStepEnum.imputation: None,
        PreprocessingStepEnum.scaling: None,
    }


@pytest.mark.parametrize(
    ('tensor_data_factory', 'service_cls'),
    [
        (_tabular_tensor_data, OptionalTabularService),
        (_ts_tensor_data, OptionalTSService),
    ],
)
@pytest.mark.unit
def test_get_optional_runtime_spec_for_tensor_data_returns_registry_entry(
    tensor_data_factory,
    service_cls,
):
    tensor_data = tensor_data_factory()
    runtime_spec = get_optional_runtime_spec_for_tensor_data(tensor_data)

    assert runtime_spec.service_cls is service_cls
    assert runtime_spec.default_steps == {
        PreprocessingStepEnum.imputation: None,
        PreprocessingStepEnum.scaling: None,
    }


@pytest.mark.unit
def test_get_optional_runtime_spec_for_tensor_data_raises_for_unsupported_type():
    tensor_data = _tabular_tensor_data()
    tensor_data.data_type = DataTypesEnum.image

    with pytest.raises(ValueError, match='Optional preprocessing is not supported'):
        get_optional_runtime_spec_for_tensor_data(tensor_data)


