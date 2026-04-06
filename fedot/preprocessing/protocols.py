from typing import Protocol

from fedot.core.data.prepared_data import PreparedData


class PreprocessingService(Protocol):
    def fit(self, pipeline, data) -> PreparedData:
        ...

    def transform(self, pipeline, data, state) -> PreparedData:
        ...