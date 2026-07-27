from dataclasses import dataclass
from typing import Any, Dict, Mapping, Type

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.industrial.core.architecture.preprocessing.ts_optional_service import OptionalTSService
from fedot.preprocessing.service.optional_service import OptionalService
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum


_DEFAULT_OPTIONAL_STEPS: Dict[PreprocessingStepEnum, None] = {
    PreprocessingStepEnum.imputation: None,
    PreprocessingStepEnum.scaling: None,
}


@dataclass(frozen=True)
class TensorOptionalRuntimeSpec:
    service_cls: Type[OptionalService]
    default_steps: Mapping[PreprocessingStepEnum, Any]


TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE: Dict[DataTypesEnum, TensorOptionalRuntimeSpec] = {
    DataTypesEnum.tabular: TensorOptionalRuntimeSpec(
        service_cls=OptionalTabularService,
        default_steps=_DEFAULT_OPTIONAL_STEPS,
    ),
    DataTypesEnum.ts: TensorOptionalRuntimeSpec(
        service_cls=OptionalTSService,
        default_steps=_DEFAULT_OPTIONAL_STEPS,
    ),
}


def get_optional_runtime_spec_for_tensor_data(
    tensor_data: TensorData,
) -> TensorOptionalRuntimeSpec:
    runtime_spec = TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE.get(tensor_data.data_type)
    if runtime_spec is None:
        raise ValueError(
            f'Optional preprocessing is not supported for data type '
            f'{tensor_data.data_type!r}.'
        )
    return runtime_spec
