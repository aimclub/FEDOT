from typing import Dict, Union
from typing import Optional

import numpy as np

from fedot.api.api_utils.data_definition import data_strategy_selector, FeaturesType, TargetType
from fedot.core.data.data import InputData, OutputData, data_type_is_table
from fedot.core.data.data_preprocessing import convert_into_column
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast, convert_forecast_to_output
from fedot.core.repository.tasks import Task, TaskTypesEnum
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

    def __init__(self, task: Task):
        self.task = task
        self.preprocessor = DataPreprocessor()

        # Dictionary with recommendations (e.g. 'cut' for cutting dataset, 'label_encode'
        # to encode features using label encoder). Parameters for transformation provided also
        self.recommendations = {'cut': self.preprocessor.cut_dataset,
                                'label_encoded': self.preprocessor.label_encoding_for_fit}

    def define_data(self,
                    features: FeaturesType,
                    target: Optional[TargetType] = None,
                    is_predict=False):
        """ Prepare data for FEDOT pipeline composing.
        Obligatory preprocessing steps are applying also. If features is dictionary
        there is a need to process MultiModalData
        """
        try:
            # TODO remove workaround
            idx = None
            if isinstance(features, dict) and 'idx' in features:
                idx = features['idx']
                del features['idx']
            data = data_strategy_selector(features=features,
                                          target=target,
                                          ml_task=self.task,
                                          is_predict=is_predict)
            if isinstance(data, dict) and idx is not None:
                for key in data:
                    data[key].idx = idx
        except Exception as ex:
            raise ValueError('Please specify the "features" as path to as path to csv file/'
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
        if self.task.task_type == TaskTypesEnum.classification:
            # Prediction should be converted into source labels
            output_prediction = current_pipeline.predict(test_data, output_mode='labels')
        elif self.task.task_type == TaskTypesEnum.ts_forecasting:
            if in_sample:
                forecast_length = test_data.task.task_params.forecast_length
                validation_blocks = validation_blocks or 1
                horizon = forecast_length * validation_blocks
                forecast = in_sample_ts_forecast(current_pipeline, test_data, horizon)
                idx = test_data.idx[-horizon:]
                prediction = convert_forecast_to_output(test_data, forecast, idx=idx)
            else:
                prediction = current_pipeline.predict(test_data)
                # Convert forecast into one-dimensional array
                forecast = np.ravel(np.array(prediction.predict))
                prediction.predict = forecast
            output_prediction = prediction
        else:
            output_prediction = current_pipeline.predict(test_data)

        return output_prediction

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
            for name in recommendations:
                rec = recommendations[name]
                # Apply desired preprocessing function
                self.recommendations[name](input_data, *rec.values())
