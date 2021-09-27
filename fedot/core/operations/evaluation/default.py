import warnings
from typing import Optional
from typing import Callable
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy

warnings.filterwarnings("ignore", category=UserWarning)


class CustomDefaultModelStrategy(EvaluationStrategy):
    """
    This class defines the default model container for custom of domain-specific implementations

    :param str operation_type: rudimentary of parent - type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    def __init__(self, operation_type: Optional[str], params: dict = None):
        super().__init__(operation_type, params)

        if not 'params' not in params.keys():
            raise KeyError('There is no key word "params" for custom model parameters in input dictionary')
        else:
            self.params_for_fit = params.get('params')

        if not 'model' not in params.keys():
            raise KeyError('There is no key word "model" for model definition in input dictionary')
        else:
            self.model = params.get('model')
        if not isinstance(self.model, Callable):
            raise ValueError('Input model is not Callable')

    def fit(self, train_data: InputData):
        """
        This strategy does not support fitting the operation
        """
        return self.model

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool) -> OutputData:
        train_data = predict_data.features
        target_data = predict_data.target

        try:
            result = trained_operation(train_data, target_data, self.params_for_fit)
        except Exception as e:
            print(e)
            raise AttributeError('Input model has incorrect behaviour. Check type hints model: \
                                  Callable[[np.array, np.array, dict], np.array]')
        output = self._convert_to_output(result, predict_data)

        return output

    def _convert_to_operation(self, operation_type: str):
        raise NotImplementedError('For this strategy there are no available operations')
