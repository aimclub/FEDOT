from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OPTIONAL_MAPPING
from fedot.preprocessing.service.optional_service import OptionalService


class OptionalTabularService(OptionalService):
    handler_mapping = PREPROCESSING_OPTIONAL_MAPPING
