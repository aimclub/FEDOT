from copy import deepcopy

import numpy as np
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.industrial.core.models.ts_forecasting.lagged_strategy.lagged_forecaster import LaggedAR
from fedot.industrial.core.tuning.search_space import get_industrial_search_space


class EigenAR(LaggedAR):
    """ Generalized linear models implementation """

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.eigen_ts = PipelineBuilder().add_node('eigen_basis', params={
            'low_rank_approximation': False,
            'rank_regularization': 'hard_thresholding'})
        self.ts_component_bound = 3

    def _define_data_and_search_space(self, train_data):
        tuning_data = deepcopy(train_data)
        custom_search_space = get_industrial_search_space(self)
        search_space = PipelineSearchSpace(
            custom_search_space=custom_search_space, replace_default_search_space=True
        )
        return tuning_data, search_space

    def fit(self, input_data):
        self.eigen_ts = self.eigen_ts.build()
        decomposed_time_series = self.eigen_ts.fit(
            input_data).predict.squeeze()
        self.fitted_model_dict = {}
        if len(decomposed_time_series.shape) == 1:
            decomposed_time_series = decomposed_time_series.reshape(1, -1)
        else:
            decomposed_time_series = decomposed_time_series[:self.ts_component_bound, :]
        for component_idx, ts_component in enumerate(decomposed_time_series):
            copy_input_data = deepcopy(input_data)
            copy_input_data.features = ts_component
            tuned_model = self.build_tuner(model_to_tune=PipelineBuilder().add_node(self.channel_model).build(),
                                           tuning_params=self.tuning_params,
                                           train_data=copy_input_data)
            self.fitted_model_dict.update({component_idx: tuned_model})

        del self.tuning_params
        return self

    def _predict(self, input_data):
        flatten_predict = self.eigen_ts.predict(input_data).predict
        table_predict = flatten_predict.reshape(int(flatten_predict.shape[0] / input_data.features.shape[0]),
                                                input_data.features.shape[0])
        decomposed_time_series = table_predict[:self.ts_component_bound, :]
        prediction = []
        for component_idx, ts_component in enumerate(decomposed_time_series):
            copy_input_data = deepcopy(input_data)
            copy_input_data.features = ts_component
            prediction.append(self.fitted_model_dict[component_idx].predict(
                copy_input_data).predict)
        prediction = np.stack(prediction)
        output_data = self._convert_to_output(input_data,
                                              predict=np.sum(
                                                  prediction, axis=0),
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def predict(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)
