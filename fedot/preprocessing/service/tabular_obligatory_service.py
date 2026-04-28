from fedot.preprocessing.service.obligatory_service import ObligatoryService
from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OBLIGATORY_MAPPING


class ObligatoryTabularService(ObligatoryService):
    """Service for mandatory preprocessing steps on tabular data.

    The service runs obligatory transformations required for TensorData creation
    (categorical encoding and embedding).

    Example of usage and creating custom steps in tests/preprocessing/test_obligatory_preprocessing.py.

    Example:
        prepared = ObligatoryTabularService().fit_transform(features, target, params)
    """
    handler_mapping = PREPROCESSING_OBLIGATORY_MAPPING
