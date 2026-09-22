from abc import ABC, abstractmethod
from typing import Sequence

from fedot.core.data.prepared_data.prepared_data import PreparedData


class AbstractPreprocessingHandler(ABC):
    """Abstract interface for preprocessing handlers."""

    @abstractmethod
    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Fit the handler on selected feature columns.

        Args:
            data: Input prepared data with feature tensor and optional target.
            features_idx: Indices of feature columns to process.

        Returns:
            Fitted preprocessing handler instance.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: PreparedData) -> PreparedData:
        """Apply fitted transformation to input data.

        Args:
            data: Input prepared data to transform.

        Returns:
            Transformed prepared data.
        """
        raise NotImplementedError

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]) -> PreparedData:
        """Fit the handler and transform input data in one call.

        Args:
            data: Input prepared data to fit and transform.
            features_idx: Indices of feature columns to process.

        Returns:
            Transformed prepared data.
        """
        return self.fit(data, features_idx).transform(data)
