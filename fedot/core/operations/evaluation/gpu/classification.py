import warnings

import cudf

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy

warnings.filterwarnings("ignore", category=UserWarning)


class CuMLClassificationStrategy(CuMLEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Predict method for classification task

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return: prediction target
        """
        n_classes = len(trained_operation.classes_)
        features = cudf.DataFrame(predict_data.features.astype('float32'))
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(features)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs':
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted
