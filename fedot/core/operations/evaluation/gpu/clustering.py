import warnings

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utilities.random import RandomStateHandler
from fedot.utilities.requirements_notificator import warn_requirement

try:
    from cuml import KMeans
    import cudf
except ModuleNotFoundError:
    warn_requirement('cudf / cuml')
    cudf = None
    KMeans = None

from typing import Optional

from fedot.core.data.data import InputData, OutputData

from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy


class CumlClusteringStrategy(CuMLEvaluationStrategy):
    __operations_by_types = {
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
        with RandomStateHandler():
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

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain SkLearn clustering strategy for {operation_type}')
