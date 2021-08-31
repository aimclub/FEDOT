import warnings

import cudf

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy

warnings.filterwarnings("ignore", category=UserWarning)


class CuMLClassificationStrategy(CuMLEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool):
        """
        Predict method for classification task

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return: prediction target
        """
        features = cudf.DataFrame(predict_data.features.astype('float32'))

        prediction = self._sklearn_compatible_prediction(trained_operation,
                                                         features)

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted
