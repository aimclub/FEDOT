from abc import ABC, abstractmethod
from typing import Sequence

from fedot.core.data.prepared_data import PreparedData


class AbstractPreprocessingHandler(ABC):
    @abstractmethod
    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: PreparedData) -> PreparedData:
        raise NotImplementedError

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]) -> PreparedData:
        return self.fit(data, features_idx).transform(data)
