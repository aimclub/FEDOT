from typing import Callable
from typing import Optional
import warnings
from fedot.core.log import Log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


class DefaultModelImplementation(ModelImplementation):
    """
    Implementation of container for custom model, which is presented as function with
    input train_data(np.array), test_data(np.array), parameters(dict)
    output type specification DataTypesEnum
    into parameters dictionary {'model': function, 'output_type': 'table'}
    """
    def __init__(self, params: dict = None, log: Log = None):
        super().__init__(log)
        self.params = params
        if not self.params:
            warnings.warn('There is no specified parameters for custom model! Skip node.')
        else:
            # init output_type
            if 'output_type' not in self.params.keys():
                self.output_type = DataTypesEnum.table
            else:
                self.output_type = DataTypesEnum[self.params.get('output_type')]

            # init model
            if 'model' not in self.params.keys():
                warnings.warn('There is no key word "model" for model definition in input dictionary. Model set to None')
            else:
                self.model = self.params.get('model')
                if not isinstance(self.model, Callable):
                    raise ValueError('Input model is not Callable')

    def fit(self, input_data):
        """
        Default implementation does not support fitting,
        it should be presented inside wrappers.model parameter
        """
        pass

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        train_data = input_data.features
        target_data = input_data.target
        predict = train_data
        # If custom model has exceptions inviolate train data goes to Output
        if not is_fit_pipeline_stage:
            try:
                predict = self.model(train_data, target_data, self.params)
            except Exception as e:
                warnings.warn(f'{e}\nInput model has incorrect behaviour. Check type hints for model: \
                                      Callable[[np.array, np.array, dict], np.array]')

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=self.output_type)
        return output_data

    def get_params(self):
        return self.params
