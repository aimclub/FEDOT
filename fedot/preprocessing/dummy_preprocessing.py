from typing import Union

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from .base_preprocessing import BasePreprocessor


class DummyPreprocessor(BasePreprocessor):
    """
    Just uses base class methods as is, passing through input data without modification
    """

    def __init__(self):
        super().__init__()

        self.log = default_log(self)

    def obligatory_prepare_for_fit(self, data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
        return super().obligatory_prepare_for_fit(data)

    def obligatory_prepare_for_predict(self, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return super().obligatory_prepare_for_predict(data)

    def optional_prepare_for_fit(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return super().optional_prepare_for_fit(pipeline, data)

    def optional_prepare_for_predict(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return super().optional_prepare_for_predict(pipeline, data)

    def label_encoding_for_fit(self, data: InputData, source_name: str = ...):
        return super().label_encoding_for_fit(data, source_name)

    def cut_dataset(self, data: InputData, border: int):
        return super().cut_dataset(data, border)

    def apply_inverse_target_encoding(self, column_to_transform: np.ndarray) -> np.ndarray:
        return super().apply_inverse_target_encoding(column_to_transform)

    def convert_indexes_for_fit(self, pipeline: Pipeline, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return super().convert_indexes_for_fit(pipeline, data)

    def convert_indexes_for_predict(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return super().convert_indexes_for_predict(pipeline, data)

    def restore_index(self, input_data: InputData, result: OutputData) -> OutputData:
        return super().restore_index(input_data, result)

    def update_indices_for_time_series(self, test_data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return super().update_indices_for_time_series(test_data)
