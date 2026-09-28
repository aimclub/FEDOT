from copy import deepcopy
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum

from fedot.industrial.core.models.classification.freq_domain_clf import FrequencyClassificator


class FrequencyRegressor(FrequencyClassificator):
    """Implementation of a composite of a topological extractor and autoreg atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.channel_model = params.get('channel_model', 'treg')
        self.tuning_params['metric'] = RegressionMetricsEnum.RMSE

    def _define_tuning_data(self, train_data):
        tuning_data = deepcopy(train_data)
        tuning_data.data_type = DataTypesEnum.image
        tuning_data.task.task_type = TaskTypesEnum.regression
        return tuning_data

    def _predict(self, input_data, output_mode='labels'):
        prediction = self.tuned_model.predict(input_data)
        return prediction
