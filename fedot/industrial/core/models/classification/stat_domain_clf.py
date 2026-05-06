import json
from copy import deepcopy
from typing import Optional

import numpy as np
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.tuning.sequential import SequentialTuner

from fedot.industrial.core.models.ts_forecasting.lagged_strategy.lagged_forecaster import LaggedAR
from fedot.industrial.tools.serialisation.path_lib import PATH_TO_DEFAULT_PARAMS


class StatClassificator(LaggedAR):
    """Implementation of a composite of a stat_extractor and classification atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.tuning_params['metric'] = ClassificationMetricsEnum.f1
        self.tuning_params['tuner'] = SequentialTuner
        self.model_name = params.get('transformation_model', 'quantile_extractor')
        self.tuning_params['tuning_iterations'] = 1

    def _define_model(self):
        with open(PATH_TO_DEFAULT_PARAMS) as json_data:
            self.default_operation_params = json.load(json_data)
        self.default_model_params = self.default_operation_params[self.model_name]
        self.default_channel_model_params = self.default_operation_params[self.channel_model]
        self.model = PipelineBuilder().add_node(self.model_name, params=self.default_model_params). \
            add_node(self.channel_model, params=self.default_channel_model_params).build()
        return self.model

    def _define_tuning_data(self, train_data):
        tuning_data = deepcopy(train_data)
        tuning_data.data_type = DataTypesEnum.image
        tuning_data.task.task_type = TaskTypesEnum.classification
        return tuning_data

    def fit(self, input_data):
        model_to_tune = self._define_model()
        self.classes_ = np.unique(input_data.target)
        self.tuned_model = self.build_tuner(model_to_tune=model_to_tune,
                                            tuning_params=self.tuning_params,
                                            train_data=input_data)
        del self.tuning_params
        return self

    def _predict(self, input_data: InputData, output_mode='labels') -> OutputData:
        prediction = self.tuned_model.predict(input_data, output_mode)
        return prediction

    def predict_for_fit(self, input_data: InputData, output_mode='labels') -> OutputData:
        return self._predict(input_data, output_mode=output_mode)

    def predict(self, input_data: InputData, output_mode='labels') -> OutputData:
        return self._predict(input_data, output_mode=output_mode)

    def predict_proba(self, input_data: InputData) -> OutputData:
        return self._predict(input_data, output_mode='probs')
