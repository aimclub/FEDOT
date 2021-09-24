import warnings
from typing import Optional
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models. \
    default_model import DefaultImplementation

warnings.filterwarnings("ignore", category=UserWarning)


class CustomDefaultModelStrategy(EvaluationStrategy):
    """
    This class defines the default model container for custom of domain-specific implementations

    :param dict params: hyperparameters to fit the model with
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.params_for_fit = params
        self.operation = DefaultImplementation

    def fit(self, train_data: InputData):
        """
        This strategy does not support fitting the operation
        """
        return self.operation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool) -> OutputData:
        pass
