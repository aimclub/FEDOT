import warnings

from cuml import KMeans
import cudf
from typing import Optional

from fedot.core.data.data import InputData, OutputData

from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy


class CumlClusteringStrategy(CuMLEvaluationStrategy):
    __operations_by_types = {
        'kmeans': KMeans,
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.params_for_fit = params

    def fit(self, train_data: InputData):
        """
        Fit method for clustering task

        :param train_data: data used for model training
        :return:
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl(n_clusters=2)

        features = cudf.DataFrame(train_data.features.astype('float32'))
        operation_implementation.fit(features)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool) -> OutputData:
        """
        Predict method for clustering task
        :param trained_operation: operation object
        :param predict_data: data used for prediction
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return :
        """
        features = cudf.DataFrame(predict_data.features.astype('float32'))
        prediction = trained_operation.predict(features)

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain SkLearn clustering strategy for {operation_type}')
