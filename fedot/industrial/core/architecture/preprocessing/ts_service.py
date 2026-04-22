from fedot.preprocessing.service.service import OptionalPreprocessingService
from fedot.preprocessing.tools.methods_mapping import TS_PREPROCESSING_MAPPING


class TSPreprocessingService(OptionalPreprocessingService):
    handler_mapping = TS_PREPROCESSING_MAPPING
