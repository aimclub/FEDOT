from typing import Callable, Optional

from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum


class CustomModelImplementation(ModelImplementation):
    """
    Implementation of container for custom model, which is presented as function with
    input train_data(np.array), test_data(np.array), parameters(dict)
    output type specification DataTypesEnum (string - 'ts', 'table', 'image', 'text')
    into parameters dictionary {'model_predict': function, 'model_fit': function}
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        if not self.params:
            raise ValueError('There is no specified parameters for custom model!')
        self.fitted_model = None

    @property
    def model_predict(self) -> Callable:
        if 'model_predict' in self.params.keys():
            model_predict = self.params.get('model_predict')
            if not isinstance(model_predict, Callable):
                self.log.warning('Input model_predict is not Callable')
            return model_predict
        else:
            raise ValueError('There is no key word "model_predict" for model definition in input dictionary.')

    @property
    def model_fit(self) -> Optional[Callable]:
        # custom model can be without fitting
        model_fit = None
        if 'model_fit' in self.params.keys():
            model_fit = self.params.get('model_fit')
            if not isinstance(model_fit, Callable):
                raise ValueError('Input model is not Callable')
        return model_fit

    def fit(self, input_data):
        """ Fit method for custom model implementation """
        if self.model_fit:
            self.fitted_model = self.model_fit(input_data.idx, input_data.features, input_data.target, self.params)
        return self.fitted_model

    def predict(self, input_data):
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

    def predict_for_fit(self, input_data):
        output_type = input_data.data_type
        # if there is no need in fitting custom model and it is fit call
        if not self.model_fit:
            predict = input_data.features
            output_data = self._convert_to_output(input_data,
                                                  predict=predict,
                                                  data_type=output_type)
            return output_data
        else:
            return self.predict(input_data)
