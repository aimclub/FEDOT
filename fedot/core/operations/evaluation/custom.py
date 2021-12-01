import warnings
from typing import Optional
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.custom_model import CustomModelImplementation
warnings.filterwarnings("ignore", category=UserWarning)


class CustomModelStrategy(EvaluationStrategy):
    """
    This class defines the default model container for custom of domain-specific implementations
    :param str operation_type: rudimentary of parent - type of the operation defined in operation or
           data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    def __init__(self, operation_type: Optional[str], params: dict = None):
        super().__init__(operation_type, params)
        self.params = params
        self.operation_impl = CustomModelImplementation(params)

    def fit(self, train_data: InputData):
        """ Fit method for custom strategy"""
        self.operation_impl.fit(train_data)
        return self.operation_impl

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool) -> OutputData:
        prediction = trained_operation.predict(predict_data, is_fit_pipeline_stage)
        # Convert prediction to output
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        return CustomModelImplementation
