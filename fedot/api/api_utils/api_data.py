from datetime import datetime
from typing import Dict, Union
from typing import Optional

import numpy as np
from golem.core.log import default_log

from fedot.api.api_utils.api_data_rules import (
    iter_shared_index_assignments,
    normalize_features_for_definition,
    plan_fit_preprocessing,
    plan_prediction,
    plan_predict_preprocessing,
)
from fedot.api.api_utils.data_definition import data_strategy_selector, FeaturesType, TargetType
from fedot.core.data.data import InputData, OutputData, data_type_is_table
from fedot.core.data.data_preprocessing import convert_into_column
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast, convert_forecast_to_output
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import convert_memory_size
from fedot.preprocessing.dummy_preprocessing import DummyPreprocessor
from fedot.preprocessing.preprocessing import DataPreprocessor


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

    def define_data(self,
                    features: FeaturesType,
                    target: Optional[TargetType] = None,
                    is_predict=False):
        """ Prepare data for FEDOT pipeline composing.
        Obligatory preprocessing steps are applying also. If features is dictionary
        there is a need to process MultiModalData
        """
        normalized_features = normalize_features_for_definition(features)

        try:
            data = data_strategy_selector(features=normalized_features.features,
                                          target=target,
                                          task=self.task,
                                          is_predict=is_predict)
            for data_source_name, shared_index in iter_shared_index_assignments(data, normalized_features.shared_index):
                data[data_source_name].idx = shared_index
        except Exception as ex:
            raise ValueError('Please specify the "features" as path to csv file/'
                             'Numpy array/Pandas DataFrame/FEDOT InputData/dict for multimodal data, '
                             f'Exception: {ex}')

        # Perform obligatory steps of data preprocessing
        if is_predict:
            data = self.preprocessor.obligatory_prepare_for_predict(data)
        else:
            data = self.preprocessor.obligatory_prepare_for_fit(data)
        return data

    def define_predictions(self, current_pipeline: Pipeline, test_data: Union[InputData, MultiModalData],
                           in_sample: bool = False, validation_blocks: int = None) -> OutputData:
        """ Prepare predictions """
        forecast_length = getattr(test_data.task.task_params, 'forecast_length', None)
        prediction_plan = plan_prediction(
            task_type=self.task.task_type,
            in_sample=in_sample,
            validation_blocks=validation_blocks,
            forecast_length=forecast_length,
        )

        if prediction_plan.output_mode is not None:
            return current_pipeline.predict(test_data, output_mode=prediction_plan.output_mode)

        if prediction_plan.use_in_sample_forecast:
            forecast = in_sample_ts_forecast(current_pipeline, test_data, prediction_plan.horizon)
            idx = test_data.idx[-prediction_plan.horizon:]
            return convert_forecast_to_output(test_data, forecast, idx=idx)

        prediction = current_pipeline.predict(test_data)
        if prediction_plan.flatten_prediction:
            prediction.predict = np.ravel(np.array(prediction.predict))
        return prediction

    def correct_predictions(self, real: InputData, prediction: OutputData):
        """ Change shape for models predictions if its necessary. Apply """
        if self.task == TaskTypesEnum.ts_forecasting:
            real.target = real.target[~np.isnan(prediction.predict)]
            prediction.predict = prediction.predict[~np.isnan(prediction.predict)]

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
                self.accept_and_apply_recommendations(input_data[data_source_name], recommendations[data_source_name])
        else:
            for name, rec in recommendations.items():
                # Apply desired preprocessing function
                self._recommendations[name](input_data, *rec.values())

    def fit_transform(self, train_data: InputData) -> InputData:
        start_time = datetime.now()
        self.log.message('Preprocessing data')
        memory_usage = convert_memory_size(train_data.memory_usage)
        features_shape = train_data.features.shape
        target_shape = train_data.target.shape
        self.log.message(
            f'Train Data (Original) Memory Usage: {memory_usage} Data Shapes: {features_shape, target_shape}')

        train_data = self._apply_preprocessing_plan(
            data=train_data,
            current_pipeline=Pipeline(),
            plan=plan_fit_preprocessing(),
        )

        memory_usage = convert_memory_size(train_data.memory_usage)

        features_shape = train_data.features.shape
        target_shape = train_data.target.shape
        self.log.message(
            f'Train Data (Processed) Memory Usage: {memory_usage} Data Shape: {features_shape, target_shape}')
        self.log.message(f'Data preprocessing runtime = {datetime.now() - start_time}')

        return train_data

    def transform(self, test_data: InputData, current_pipeline) -> InputData:
        start_time = datetime.now()
        self.log.message('Preprocessing data')
        memory_usage = convert_memory_size(test_data.memory_usage)
        features_shape = test_data.features.shape
        target_shape = test_data.target.shape
        self.log.message(
            f'Test Data (Original) Memory Usage: {memory_usage} Data Shapes: {features_shape, target_shape}')

        test_data = self._apply_preprocessing_plan(
            data=test_data,
            current_pipeline=current_pipeline,
            plan=plan_predict_preprocessing(),
        )

        memory_usage = convert_memory_size(test_data.memory_usage)
        features_shape = test_data.features.shape
        target_shape = test_data.target.shape
        self.log.message(
            f'Test Data (Processed) Memory Usage: {memory_usage} Data Shape: {features_shape, target_shape}')
        self.log.message(f'Data preprocessing runtime = {datetime.now() - start_time}')

        return test_data

    def _apply_preprocessing_plan(self,
                                  data: InputData,
                                  current_pipeline: Pipeline,
                                  plan) -> InputData:
        for step_name in plan.steps:
            self.log.debug(f'- {step_name} started')
            step = getattr(self.preprocessor, step_name)
            if step_name.startswith('optional_prepare') or step_name.startswith('convert_indexes'):
                data = step(pipeline=current_pipeline, data=data)
            else:
                data = step(data=data)

        if plan.mark_auto_preprocessed:
            data.supplementary_data.is_auto_preprocessed = True

        return data
