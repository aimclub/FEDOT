from fedot.core.data.data import InputData
from typing import Callable
from typing import Optional
from fedot.core.log import Log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


class DefaultModelImplementation(ModelImplementation):
    """
    Implementation of container for custom model
    """
    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        if 'params' not in params.keys():
            raise KeyError('There is no key word "params" for custom model parameters in input dictionary')
        else:
            self.params_for_fit = params.get('params')

        if 'model' not in params.keys():
            raise KeyError('There is no key word "model" for model definition in input dictionary')
        else:
            self.model = params.get('model')
        if not isinstance(self.model, Callable):
            raise ValueError('Input model is not Callable')

    def fit(self, input_data):
        """
        Default implementation does not support fitting,
        it should be presented inside model parameter
        """
        return self.model

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        train_data = input_data.features
        target_data = input_data.target
        if is_fit_pipeline_stage:
            predict = train_data
        else:
            try:
                predict = self.model(train_data, target_data, self.params_for_fit)
            except Exception as e:
                raise AttributeError(f'Error: {e}\nInput model has incorrect behaviour. Check type hints model: \
                                      Callable[[np.array, np.array, dict], np.array]')

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params_for_fit
