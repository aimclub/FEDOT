from fedot.utilities.requirements_notificator import warn_requirement

try:
    import cudf
except ModuleNotFoundError:
    warn_requirement('cudf')
    cudf = None

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy


class CuMLRegressionStrategy(CuMLEvaluationStrategy):
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
