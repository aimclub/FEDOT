from typing import Protocol

from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.preprocessing_state import PreprocessingState

class PreprocessingService(Protocol):
    def fit(self, pipeline, data) -> PreparedData:
        ...

    def transform(self, pipeline, data, state: PreprocessingState) -> PreparedData:
        ...