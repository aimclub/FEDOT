from datetime import datetime
from typing import Dict, Union
from typing import Optional

import numpy as np
import torch
from golem.core.log import default_log

from fedot.api.api_utils.api_data_rules import (
    build_tensordata_definition_plan,
    iter_shared_index_assignments,
    normalize_features_for_definition,
    plan_fit_preprocessing,
    plan_prediction,
    plan_predict_preprocessing,
)
from fedot.api.api_utils.data_definition import data_strategy_selector, FeaturesType, TargetType
from fedot.core.data.input_data.data import InputData, OutputData, data_type_is_table
from fedot.core.data.bridges.input_to_tensor import input_data_to_tensordata
from fedot.core.data.bridges.tensor_to_input import tensordata_to_input_data
from fedot.core.data.common.enums import StateEnum
from fedot.preprocessing.data_preprocessing import convert_into_column
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.pipelines.ensembling.pipeline_ensemble import PipelineEnsemble
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast, convert_forecast_to_output
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import convert_memory_size
from fedot.preprocessing.dummy_preprocessing import DummyPreprocessor
from fedot.preprocessing.preprocessing import DataPreprocessor
from fedot.core.data.tensor_data import TensorData


class ApiDataProcessor:
    """
    Class for selecting optimal data processing strategies based on type of data.
    Available data sources are:
        * numpy array
        * pandas DataFrame
        * string (path to csv file)
        * InputData (FEDOT dataclass)

    Data preprocessing such a class performing also
    """

    def __init__(self, task: Task, use_input_preprocessing: bool = True):
        self.task = task

        self._recommendations = {}

        if use_input_preprocessing:
            self.preprocessor = DataPreprocessor()

            # Dictionary with recommendations (e.g. 'cut' for cutting dataset, 'label_encoded'
            # to encode features using label encoder). Parameters for transformation provided also
            self._recommendations = {
                'cut': self.preprocessor.cut_dataset,
                'label_encoded': self.preprocessor.label_encoding_for_fit
            }

        else:
            self.preprocessor = DummyPreprocessor()

        self.log = default_log(self)

    def define_predictions_tensordata(self,
                                      current_pipeline: Union[Pipeline, PipelineEnsemble],
                                      test_data: TensorData,
                                      in_sample: bool = False,
                                      validation_blocks: int = None) -> TensorData:
        """ Prepare predictions """
        forecast_length = getattr(
            self.task.task_params, 'forecast_length', None)
        prediction_plan = plan_prediction(
            task_type=self.task.task_type,
            in_sample=in_sample,
            validation_blocks=validation_blocks,
            forecast_length=forecast_length,
        )

        # TODO @lopa10ko: it should be refactored for TD
        # if prediction_plan.use_in_sample_forecast:
        #     forecast = in_sample_ts_forecast(
        #         current_pipeline, test_data, prediction_plan.horizon)
        #     idx = test_data.idx[-prediction_plan.horizon:]
        #     return convert_forecast_to_output(test_data, forecast, idx=idx)

        prediction = current_pipeline.predict_tensordata(test_data)
        if prediction_plan.flatten_prediction:
            prediction.predict = torch.flatten(prediction.predict)
        return prediction

    def correct_predictions(self, real: InputData, prediction: OutputData):
        """ Change shape for models predictions if its necessary. Apply """
        if self.task == TaskTypesEnum.ts_forecasting:
            real.target = real.target[~np.isnan(prediction.predict)]
            prediction.predict = prediction.predict[~np.isnan(
                prediction.predict)]

        if data_type_is_table(prediction):
            # Check dimensions for real and predicted values
            if len(real.target.shape) != len(prediction.predict.shape):
                prediction.predict = convert_into_column(prediction.predict)
                real.target = convert_into_column(np.array(real.target))

    def accept_and_apply_recommendations(self, input_data: Union[InputData, MultiModalData], recommendations: Dict):
        """
        Accepts recommendations for preprocessing from DataAnalyser

        :param input_data - data for preprocessing
        :param recommendations - dict with recommendations
        """
        if isinstance(input_data, MultiModalData):
            for data_source_name, values in input_data.items():
                self.accept_and_apply_recommendations(
                    input_data[data_source_name], recommendations[data_source_name])
        else:
            for name, rec in recommendations.items():
                # Apply desired preprocessing function
                self._recommendations[name](input_data, *rec.values())
