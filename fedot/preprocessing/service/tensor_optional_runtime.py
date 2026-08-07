from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.schemas import validate_optional_preprocessing_data_type
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
}


def register_optional_runtime(
    data_type: DataTypesEnum,
    service_cls: Type[OptionalService],
    default_steps: Optional[Mapping[PreprocessingStepEnum, Any]] = None,
) -> None:
    """Register an optional-preprocessing runtime for ``data_type``.

    Core ships tabular only. Industrial (and other extensions) should call this
    at their init to plug in additional data types without an eager import from core.
    """
    TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE[data_type] = TensorOptionalRuntimeSpec(
        service_cls=service_cls,
        default_steps=_DEFAULT_OPTIONAL_STEPS if default_steps is None else default_steps,
    )


def get_optional_runtime_spec_for_tensor_data(
    tensor_data: TensorData,
) -> TensorOptionalRuntimeSpec:
    data_type = validate_optional_preprocessing_data_type(tensor_data.data_type)
    return TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE[data_type]
