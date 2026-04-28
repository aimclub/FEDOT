from fedot.preprocessing.service.obligatory_service import ObligatoryService
from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OBLIGATORY_MAPPING


class ObligatoryTabularService(ObligatoryService):
    handler_mapping = PREPROCESSING_OBLIGATORY_MAPPING
