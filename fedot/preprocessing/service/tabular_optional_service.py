from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OPTIONAL_MAPPING
from fedot.preprocessing.service.optional_service import OptionalService


class OptionalTabularService(OptionalService):
    """Service for optional preprocessing steps on tabular data.

    The service applies user-defined optional strategy (imputation, scaling,
    filtering, or custom handlers) via `fit_transform`.

    Example of usage and creating custom steps in tests/preprocessing/test_optional_preprocessing.py.

    Example:
        X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
        td = TensorData.create(X, backend_name="cpu")
        strategy = {PreprocessingStepEnum.imputation: None}
        prepared = OptionalTabularService().fit_transform(td, strategy)
    """
    handler_mapping = PREPROCESSING_OPTIONAL_MAPPING
