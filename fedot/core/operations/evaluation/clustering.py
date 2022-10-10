import warnings
from typing import Optional

from sklearn.cluster import KMeans as SklearnKmeans

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.evaluation.evaluation_interfaces import SkLearnEvaluationStrategy
from fedot.core.utilities.random import RandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnClusteringStrategy(SkLearnEvaluationStrategy):
    __operations_by_types = {
        'kmeans': SklearnKmeans
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        if not params:
            self.params_for_fit.update(n_clusters=2)
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: InputData):
        """
        Fit method for clustering task

        :param train_data: data used for model training
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(**self.params_for_fit.to_dict())
        with RandomStateHandler():
            operation_implementation.fit(train_data.features)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for clustering task
        :param trained_operation: operation object
        :param predict_data: data used for prediction
        """
        prediction = trained_operation.predict(predict_data.features)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain SkLearn clustering strategy for {operation_type}')
