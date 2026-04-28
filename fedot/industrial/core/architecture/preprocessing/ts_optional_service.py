from fedot.preprocessing.service.optional_service import OptionalService
from fedot.preprocessing.tools.methods_mapping import TS_PREPROCESSING_MAPPING


class TSPreprocessingService(OptionalService):
    """TSPreprocessingService implementation."""
    handler_mapping = TS_PREPROCESSING_MAPPING
