import warnings

from golem.utilities.requirements_notificator import warn_requirement

from fedot.core.operations.operation_parameters import OperationParameters

try:
    from cuml import KMeans
    import cudf
except ModuleNotFoundError:
    warn_requirement('cudf / cuml', 'cudf / cuml')
    cudf = None
    KMeans = None

from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy
from fedot.utilities.random import ImplementationRandomStateHandler


class CumlClusteringStrategy(CuMLEvaluationStrategy):
    _operations_by_types = {
        'kmeans': KMeans
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: InputData):
        """
        Fit method for clustering task

        :param train_data: data used for model training
        :return:
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit.to_dict())
        else:
            operation_implementation = self.operation_impl(n_clusters=2)

        features = cudf.DataFrame(train_data.features.astype('float32'))
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(features)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for regression task for predict stage
        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return:
        """

        features = cudf.DataFrame(predict_data.features.astype('float32'))

        prediction = trained_operation.predict(features)
        converted = self._convert_to_output(prediction, predict_data)

        return converted
