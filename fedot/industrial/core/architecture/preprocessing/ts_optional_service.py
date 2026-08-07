from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.service.optional_service import OptionalService
from fedot.preprocessing.service.tensor_optional_runtime import register_optional_runtime
from fedot.preprocessing.tools.methods_mapping import TS_PREPROCESSING_MAPPING


class OptionalTSService(OptionalService):
    """OptionalTSService implementation."""
    handler_mapping = TS_PREPROCESSING_MAPPING


register_optional_runtime(DataTypesEnum.ts, OptionalTSService)
