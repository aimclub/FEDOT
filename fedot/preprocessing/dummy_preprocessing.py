from typing import Union, TYPE_CHECKING

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from .base_preprocessing import BasePreprocessor

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class DummyPreprocessor(BasePreprocessor):
    """
    Just uses base class methods as is, passing through input data without modification
    """

    def __init__(self):
        super().__init__()

        self.log = default_log(self)

    def obligatory_prepare_for_fit(self, data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
        BasePreprocessor.mark_as_preprocessed(data)
        return data

    def obligatory_prepare_for_predict(self, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        BasePreprocessor.mark_as_preprocessed(data)
        return data

    def optional_prepare_for_fit(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return data

    def optional_prepare_for_predict(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return data

    def label_encoding_for_fit(self, data: InputData, source_name: str = ...):
        pass

    def cut_dataset(self, data: InputData, border: int):
        pass

    def apply_inverse_target_encoding(self, column_to_transform: np.ndarray) -> np.ndarray:
        return column_to_transform

    def convert_indexes_for_fit(self, pipeline: 'Pipeline', data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return data

    def convert_indexes_for_predict(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return data

    def restore_index(self, input_data: InputData, result: OutputData) -> OutputData:
        return result

    def update_indices_for_time_series(self, test_data: Union[InputData, MultiModalData]) -> Union[
        InputData, MultiModalData]:
        return test_data
