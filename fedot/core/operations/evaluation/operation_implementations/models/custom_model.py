from typing import Optional
from fedot.core.log import Log, default_log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum

# to time_series.py CustomTsForecastingStrategy(EvaluationStrategy) as 'custom': DomainSpecificModelImplementation

class DomainSpecificModelImplementation(ModelImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.model = None
        self.input_data = None

        # Define logger object
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def _check_model_structure(self):

        # check if custom model is in class which can be called by .predict() method
        pass

    def fit(self, input_data):
        """ Class doesn't support fit operation
        :param input_data: data with features, target and ids to process
        """
        # as we use physical models there is no train data and fitting

        self.input_data = input_data
        pass

    def predict(self):

    def predict(self, input_data, is_fit_chain_stage: Optional[bool]):




        predicted = self.model.predict()

        output_data = self._convert_to_output(self.input_data,
                                              predict=predicted,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params




