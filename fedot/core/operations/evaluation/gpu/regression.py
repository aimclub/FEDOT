import cudf

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy


class CuMLRegressionStrategy(CuMLEvaluationStrategy):
    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool):
        """
        Predict method for regression task
        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return:
        """

        features = cudf.DataFrame(predict_data.features.astype('float32'))

        prediction = trained_operation.predict(features)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)

        return converted
