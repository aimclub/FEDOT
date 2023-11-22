import warnings
from typing import Optional

from examples.simple.classification.cust.cnn_impls import MyCNNImplementation
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class ImageClassificationStrategy(EvaluationStrategy):
    _operations_by_types = {
        'cnn_1': MyCNNImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained data operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for classification task for predict stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """
        n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data)
            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            elif n_classes == 2 and self.output_mode != 'full_probs' and len(prediction.shape) > 1:
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted
