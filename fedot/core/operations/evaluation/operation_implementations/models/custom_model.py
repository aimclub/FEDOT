from typing import Callable
from typing import Optional
import warnings
from fedot.core.log import Log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data import InputData


class CustomModelImplementation(ModelImplementation):
    """
    Implementation of container for custom model, which is presented as function with
    input train_data(np.array), test_data(np.array), parameters(dict)
    output type specification DataTypesEnum (string - 'ts', 'table', 'image', 'text')
    into parameters dictionary {'model_predict': function, 'model_fit': function}
    """
    def __init__(self, params: dict = None, log: Log = None):
        super().__init__(log)
        self.params = params
        self.fitted_model = None
        if not self.params:
            raise ValueError('There is no specified parameters for custom model!')
        else:
            # init model
            if 'model_predict' not in self.params.keys():
                raise ValueError('There is no key word "model_predict" for model definition in input dictionary.')
            if 'model_fit' not in self.params.keys():
                raise ValueError('There is no key word "model_fit" for model definition in input dictionary.')
            else:
                self.model_predict = self.params.get('model_predict')
                self.model_fit = self.params.get('model_fit')
                if not isinstance(self.model_fit, Callable) or not isinstance(self.model_predict, Callable):
                    raise ValueError('Input model is not Callable')

    def fit(self, input_data):
        """ Fit method for custom model implementation """
        self.fitted_model = self.model_fit(input_data.features, input_data.target, self.params)
        return self.fitted_model

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        self.output_type = input_data.data_type
        predict = input_data.features
        # If custom model has exceptions inviolate train data goes to Output
        if not is_fit_pipeline_stage:
            try:
                predict, output_type = self.model_predict(self.fitted_model,
                                                          input_data.features,
                                                          input_data.target,
                                                          self.params)
                self.output_type = DataTypesEnum[output_type]
            except Exception as e:
                raise TypeError(f'{e}\nInput model has incorrect behaviour. Check type hints for model: \
                                      Callable[[any, np.array, np.array, dict], np.array, str]')

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=self.output_type)
        return output_data

    def get_params(self):
        return self.params
