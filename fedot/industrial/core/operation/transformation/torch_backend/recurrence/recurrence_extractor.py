from typing import Optional

import torch

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.industrial.core.models.base_extractor import BaseExtractor
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.operation.transformation.data.kernel_matrix import TorchTSTransformer
from fedot.industrial.core.operation.transformation.torch_backend.recurrence.sequences import RecurrenceFeatureExtractorTorch


class RecurrenceExtractor(BaseExtractor):
    """
    A feature extractor for time series based on recurrence plots and recurrence quantification analysis (RQA).

    This class transforms time series into recurrence matrices and extracts statistical features
    such as recurrence rate, determinism, laminarity, and line-based metrics.
    It supports both feature-based and image-based representations of recurrence plots.

    Attributes:
        window_size (int): The size of the sliding window for trajectory matrix construction.
        stride (int): The stride for the sliding window.
        rec_metric (str): The distance metric used for recurrence plot construction (e.g., 'cosine').
        image_mode (bool): If True, returns the recurrence plot as a 3D tensor (image-like representation).
                          If False, returns RQA features as a 1D tensor.
        transformer: The transformer class used to convert time series to recurrence matrices.
        extractor: The extractor class used to compute RQA features from recurrence matrices.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        # TODO romankuklo: add threshold for other metrics
        self.rec_metric = params.get('rec_metric', 'cosine')
        self.image_mode = params.get('image_mode', False)
        self.transformer = TorchTSTransformer
        self.extractor = RecurrenceFeatureExtractorTorch

    def __repr__(self):
        return 'Reccurence Class for TS representation'

    def _generate_features_from_ts(self, ts: torch.Tensor) -> torch.Tensor:
        if self.window_size != 0:
            trajectory_transformer = HankelMatrix(time_series=ts,
                                                  window_size=self.window_size,
                                                  strides=self.stride)
            ts = trajectory_transformer.trajectory_matrix
            self.ts_length = trajectory_transformer.ts_length
        specter = self.transformer(time_series=ts,
                                   rec_metric=self.rec_metric)

        if not self.image_mode:
            feature_df = specter.ts_to_recurrence_matrix()
            feature_df = self.extractor(
                recurrence_matrix=feature_df).quantification_analysis()
            features = torch.tensor(list(feature_df.values()))
            features = torch.nan_to_num(features, nan=0.0)
        else:
            features = specter.ts_to_3d_recurrence_matrix()
        return features

    def generate_recurrence_features(self, ts: torch.Tensor):
        if ts.ndim < 3:
            aggregation_df = self._generate_features_from_ts(ts)
        else:
            aggregation_df = self._get_torch_feature_matrix(
                self._generate_features_from_ts, ts)
        return aggregation_df
