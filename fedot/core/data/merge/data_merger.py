from dataclasses import replace
from typing import List, Iterable, Union, Optional

import numpy as np
import torch

from golem.core.log import default_log
from golem.utilities.data_structures import are_same_length

from fedot.core.data.common.array_utils import find_common_elements, atleast_2d, atleast_4d, flatten_extra_dim
from fedot.core.data.input_data.data import OutputData, InputData
from fedot.core.data.merge.supplementary_data_merger import SupplementaryDataMerger
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum


class DataMerger:
    """
    Base class for merging a number of OutputData from one or several parent nodes
    into a single InputData for the next level node in the computation graph.

    Main method of the interface is `merge` that delegates merge stages to other methods.
    Merge process can be customized for data types by overriding these methods, primarily:
    `merge_targets`, `preprocess_predicts`, `merge_predicts`, `postprocess_predicts'.

    :param outputs: list with OutputData from parent nodes for merging
    """

    def __init__(self, outputs: List['OutputData'], data_type: DataTypesEnum = None):
        self.log = default_log(self)
        self.outputs = outputs
        self.data_type = data_type or DataMerger.get_datatype_for_merge(
            output.data_type for output in outputs)

        # Ensure outputs are of equal length, find common index if it is not
        idx_list = [np.asarray(output.idx) for output in outputs]
        self.common_indices = find_common_elements(*idx_list)
        if len(self.common_indices) == 0:
            raise ValueError('There are no common indices for outputs')

        # Find first output with the main target & resulting task
        self.main_output = DataMerger.find_main_output(outputs)

    @staticmethod
    def get(outputs: List['OutputData']) -> 'DataMerger':
        """ Construct appropriate data merger for the outputs. """

        # Ensure outputs can be merged
        data_type = DataMerger.get_datatype_for_merge(
            output.data_type for output in outputs)
        if data_type is None:
            raise ValueError("Can't merge different data types")

        merger_by_type = {
            DataTypesEnum.table: DataMerger,
            DataTypesEnum.ts: TSDataMerger,
            DataTypesEnum.multi_ts: TSDataMerger,
            DataTypesEnum.image: ImageDataMerger,
            DataTypesEnum.text: TextDataMerger,
        }
        cls = merger_by_type.get(data_type)
        if not cls:
            raise ValueError(f'Unable to merge data type {cls}')
        return cls(outputs, data_type)

    @staticmethod
    def get_datatype_for_merge(data_types: Iterable[DataTypesEnum]) -> Optional[DataTypesEnum]:
        # Check is all data types can be merged or not
        distinct = set(data_types)
        return distinct.pop() if len(distinct) == 1 else None

    def merge(self) -> 'InputData':
        common_idx = self.select_common(self.main_output.idx)

        filtered_main_target = self.merge_targets()

        common_predicts = self.find_common_predicts()
        mergeable_predicts = self.preprocess_predicts(common_predicts)
        merged_features = self.merge_predicts(mergeable_predicts)
        merged_features = self.postprocess_predicts(merged_features)

        updated_metadata = SupplementaryDataMerger(
            self.outputs, self.main_output).merge()

        return InputData(idx=common_idx, features=merged_features, target=filtered_main_target,
                         task=self.main_output.task, data_type=self.data_type,
                         numerical_idx=self.main_output.numerical_idx,
                         categorical_idx=self.main_output.categorical_idx,
                         encoded_idx=self.main_output.encoded_idx,
                         categorical_features=self.main_output.categorical_features,
                         features_names=self.main_output.features_names,
                         supplementary_data=updated_metadata)

    def merge_targets(self) -> np.array:
        filtered_main_target = self.main_output.target
        # if target has the same form as index
        #  then it makes sense to extract target with common indices
        if filtered_main_target is not None and len(self.main_output.idx) == len(filtered_main_target):
            filtered_main_target = self.select_common(
                self.main_output.idx, filtered_main_target)
        return filtered_main_target

    def find_common_predicts(self) -> List[np.array]:
        """ Selects and returns only those elements of predicts that are common to all outputs. """

        # Forecast index is index with a length different from that of features/predictions.
        # Such index can't be used for extracting common predictions, and it must be
        # handled separately. This case arises for timeseries after lagged transform,
        # where the datatype becomes 'table', but we still must merge it as timeseries.
        is_forecast_indices = map(DataMerger.is_forecast_index, self.outputs)

        if any(is_forecast_indices):
            # Cut prediction length to minimum length
            predict_len = min(len(output.predict) for output in self.outputs)
            common_predicts = [output.predict[:predict_len]
                               for output in self.outputs]
        else:
            common_predicts = [self.select_common(
                output.idx, output.predict) for output in self.outputs]
            if not are_same_length(common_predicts):
                raise ValueError(
                    'Indices of merged data are not equal and not unique. Check validity of the pipeline.')
        return common_predicts

    def preprocess_predicts(self, predicts: List[np.array]) -> List[np.array]:
        """ Pre-process (e.g. equalizes sizes, reshapes) and return list of arrays that can be merged. """
        return list(map(atleast_2d, predicts))

    def merge_predicts(self, predicts: List[np.array]) -> np.array:
        # Finally, merge predictions into features for the next stage
        return np.concatenate(predicts, axis=-1)

    def postprocess_predicts(self, merged_predicts: np.array) -> np.array:
        """ Post-process merged predictions (e.g. reshape). """
        return merged_predicts

    def select_common(self, idx: Union[list, np.array], data: Union[list, np.array] = None):
        """ Select elements from data according to index for data.
         Includes only elements with index from self.common_indices. """
        index_mask = np.isin(idx, self.common_indices)
        sliced = data if data is not None else idx
        sliced = np.asarray(sliced)[index_mask]
        return sliced

    @staticmethod
    def is_forecast_index(output: 'OutputData'):
        return len(output.idx) != len(output.predict)

    @staticmethod
    def find_main_output(outputs: List['OutputData']) -> 'OutputData':
        """ Returns first output with main target or (if there are
        no main targets) the output with priority secondary target. """
        priority_output = next((output for output in outputs
                                if output.supplementary_data.is_main_target), None)
        if not priority_output:
            flow_lengths = [
                output.supplementary_data.data_flow_length for output in outputs]
            i_priority_secondary = np.argmin(flow_lengths)
            priority_output = outputs[i_priority_secondary]
        return priority_output


class TensorDataMerger:
    """
    Merges TensorData objects from parent nodes into a TensorData for the next node.

    TensorData runtime has no TensorOutputData, so parent outputs are merged by
    their ``features`` field. The resulting container keeps the metadata from the
    first parent and clears ``predict`` so only final model predictions are stored
    in ``predict``.
    """

    def __init__(self, outputs: List[TensorData]):
        if not outputs:
            raise ValueError('No TensorData outputs to merge')
        self.outputs = outputs
        self.main_output = self._find_main_output(outputs)
        self.data_type = DataMerger.get_datatype_for_merge(
            output.data_type for output in outputs)
        # TODO romankuklo: add ValueError schema
        if self.data_type is None:
            raise ValueError("Can't merge different TensorData data types")
        self.common_indices = self._find_common_indices()

    def merge(self) -> TensorData:
        merged_features = self.merge_features(self.find_common_features())

        return replace(
            self.main_output,
            idx=self.common_indices,
            features=merged_features,
            target=self.merge_target(),
            predict=None,
        )

    def _find_common_indices(self):
        idx_list = [output.idx for output in self.outputs]
        if any(idx is None for idx in idx_list):
            self._check_equal_rows([output.features for output in self.outputs])
            return self.main_output.idx

        common_indices = find_common_elements(*[np.asarray(idx) for idx in idx_list])
        if len(common_indices) == 0:
            raise ValueError('There are no common indices for TensorData outputs')
        return common_indices

    def find_common_features(self) -> List[torch.Tensor]:
        features = [
            self._select_common(output, output.features)
            for output in self.outputs
        ]
        self._check_equal_rows(features)
        return self._normalize_feature_shapes(features)

    def merge_target(self) -> Optional[torch.Tensor]:
        target = self.main_output.target
        if target is None:
            return None
        if self.main_output.idx is None or len(self.main_output.idx) != len(target):
            return target
        return self._select_common(self.main_output, target)

    @staticmethod
    def merge_features(features: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=-1)

    def _select_common(self, output: TensorData, tensor: torch.Tensor) -> torch.Tensor:
        if self.common_indices is None or output.idx is None:
            return tensor
        index_mask = np.isin(np.asarray(output.idx), self.common_indices)
        tensor_mask = torch.as_tensor(index_mask, dtype=torch.bool, device=tensor.device)
        return tensor[tensor_mask]

    @staticmethod
    def _check_equal_rows(tensors: List[torch.Tensor]):
        row_counts = [tensor.shape[0] for tensor in tensors]
        if len(set(row_counts)) != 1:
            raise ValueError(
                f"Can't merge TensorData objects with different row counts: {row_counts}")

    @staticmethod
    def _normalize_feature_shapes(features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize TensorData branch outputs before concatenation.

        Contract:
        - all tensors must have equal sample count in dim 0 (validated earlier);
        - 1D tensors are promoted to 2D as ``[n_samples, 1]``;
        - if branch shapes are incompatible for ``torch.cat(..., dim=-1)``,
          tensors are flattened to ``[n_samples, -1]``.
        """
        normalized = [TensorDataMerger._atleast_2d_tensor(tensor) for tensor in features]
        if TensorDataMerger._can_cat_by_last_axis(normalized):
            return normalized
        return [tensor.reshape(tensor.shape[0], -1) for tensor in normalized]

    @staticmethod
    def _atleast_2d_tensor(tensor: torch.Tensor) -> torch.Tensor:
        while tensor.ndim < 2:
            tensor = tensor.unsqueeze(-1)
        return tensor

    @staticmethod
    def _can_cat_by_last_axis(features: List[torch.Tensor]) -> bool:
        if not features:
            return True
        base_shape = features[0].shape[:-1]
        return all(tensor.shape[:-1] == base_shape for tensor in features[1:])

    @staticmethod
    def _find_main_output(outputs: List[TensorData]) -> TensorData:
        """Choose branch metadata holder for merged TensorData.

        TensorData runtime has no SupplementaryData, so main output selection
        falls back to the first branch with available target. If all targets are
        absent, the first branch is used.
        """
        return next((output for output in outputs if output.target is not None), outputs[0])


class ImageDataMerger(DataMerger):

    def preprocess_predicts(self, predicts: List[np.array]) -> List[np.array]:
        # Reshape predicts to 4d (idx, width, height, channels)
        reshaped_predicts = list(map(atleast_4d, predicts))

        # And check image sizes
        img_wh = [predict.shape[1:3] for predict in reshaped_predicts]
        # Can merge only images of the same size
        invalid_sizes = len(set(img_wh)) > 1
        if invalid_sizes:
            raise ValueError(
                "Can't merge images of different sizes: " + str(img_wh))

        return reshaped_predicts


class TSDataMerger(DataMerger):

    def postprocess_predicts(self, merged_predicts: np.array) -> np.array:
        # Ensure that 1d-column timeseries remains 1d timeseries
        return flatten_extra_dim(merged_predicts)


class TextDataMerger(DataMerger):

    def merge_predicts(self, predicts: List[np.array]) -> np.array:
        if any(len(pred.shape) > 2 for pred in predicts):
            raise ValueError(
                'Merge of arrays with more than 2 dimensions is not supported')
        if len(predicts) > 1:
            predicts = [predict.astype(str) for predict in predicts]
            result = predicts[0]
            for i in range(1, len(predicts)):
                result = np.core.defchararray.add(result, predicts[i])
            return result
        return predicts[0]

    def postprocess_predicts(self, merged_predicts: np.array) -> np.array:
        return merged_predicts
