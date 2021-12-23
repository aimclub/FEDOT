import warnings
from typing import Callable, Optional

from fedot.core.log import Log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


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
        self.model_fit = None
        self.model_predict = None
        self.fitted_model = None
        if not self.params:
            raise ValueError('There is no specified parameters for custom model!')
        else:
            # init model
            if 'model_predict' in self.params.keys():
                self.model_predict = self.params.get('model_predict')
                if not isinstance(self.model_predict, Callable):
                    warnings.warn('Input model_predict is not Callable')
            else:
                raise ValueError('There is no key word "model_predict" for model definition in input dictionary.')

            # custom model can be without fitting
            if 'model_fit' in self.params.keys():
                self.model_fit = self.params.get('model_fit')
                if not isinstance(self.model_fit, Callable):
                    raise ValueError('Input model is not Callable')

    def fit(self, input_data):
        """ Fit method for custom model implementation """
        if self.model_fit:
            self.fitted_model = self.model_fit(input_data.idx, input_data.features, input_data.target, self.params)
        return self.fitted_model

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        output_type = input_data.data_type
        # if there is no need in fitting custom model and it is fit call
        if is_fit_pipeline_stage and not self.model_fit:
            predict = input_data.features
            # If custom model has exceptions inviolate train data goes to Output
        # make prediction if predict call or there is need to fit custom model
        else:
            try:
                predict, output_type = self.model_predict(self.fitted_model,
                                                          input_data.idx,
                                                          input_data.features,
                                                          self.params)
                if (input_data.data_type == DataTypesEnum.ts and
                        input_data.target is not None and len(input_data.target.shape) > 1):
                    # change target after custom model is processed
                    input_data.target = input_data.target[:, 0]
                output_type = DataTypesEnum[output_type]
            except Exception as e:
                raise TypeError(f'{e}\nInput model has incorrect behaviour. Check type hints for model: \
                                        Callable[[any, np.array, dict], np.array, str]')

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=output_type)
        return output_data

    def get_params(self):
        return self.params
