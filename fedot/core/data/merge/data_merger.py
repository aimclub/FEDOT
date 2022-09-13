from typing import List, Iterable, Union

from fedot.core.data.data import OutputData, InputData
from fedot.core.data.merge.supplementary_data_merger import SupplementaryDataMerger
from fedot.core.log import default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.array_utilities import *
from fedot.core.utilities.data_structures import are_same_length


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
        self.data_type = data_type or DataMerger.get_datatype_for_merge(output.data_type for output in outputs)

        # Ensure outputs are of equal length, find common index if it is not
        idx_list = [np.asarray(output.idx) for output in outputs]
        self.common_indices = find_common_elements(*idx_list)
        if len(self.common_indices) == 0:
            raise ValueError(f'There are no common indices for outputs')

        # Find first output with the main target & resulting task
        self.main_output = DataMerger.find_main_output(outputs)

    @staticmethod
    def get(outputs: List['OutputData']) -> 'DataMerger':
        """ Construct appropriate data merger for the outputs. """

        # Ensure outputs can be merged
        data_type = DataMerger.get_datatype_for_merge(output.data_type for output in outputs)
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

        updated_metadata = SupplementaryDataMerger(self.outputs, self.main_output).merge()

        return InputData(idx=common_idx, features=merged_features, target=filtered_main_target,
                         task=self.main_output.task, data_type=self.data_type,
                         supplementary_data=updated_metadata)

    def merge_targets(self) -> np.array:
        filtered_main_target = self.main_output.target
        # if target has the same form as index
        #  then it makes sense to extract target with common indices
        if filtered_main_target is not None and len(self.main_output.idx) == len(filtered_main_target):
            filtered_main_target = self.select_common(self.main_output.idx, filtered_main_target)
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
            common_predicts = [output.predict[:predict_len] for output in self.outputs]
        else:
            common_predicts = [self.select_common(output.idx, output.predict) for output in self.outputs]
            if not are_same_length(common_predicts):
                raise ValueError('Indices of merged data are not equal and not unique. Check validity of the pipeline.')
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
            flow_lengths = [output.supplementary_data.data_flow_length for output in outputs]
            i_priority_secondary = np.argmin(flow_lengths)
            priority_output = outputs[i_priority_secondary]
        return priority_output


class ImageDataMerger(DataMerger):

    def preprocess_predicts(self, predicts: List[np.array]) -> List[np.array]:
        # Reshape predicts to 4d (idx, width, height, channels)
        reshaped_predicts = list(map(atleast_4d, predicts))

        # And check image sizes
        img_wh = [predict.shape[1:3] for predict in reshaped_predicts]
        invalid_sizes = len(set(img_wh)) > 1  # Can merge only images of the same size
        if invalid_sizes:
            raise ValueError("Can't merge images of different sizes: " + str(img_wh))

        return reshaped_predicts


class TSDataMerger(DataMerger):

    def postprocess_predicts(self, merged_predicts: np.array) -> np.array:
        # Ensure that 1d-column timeseries remains 1d timeseries
        return flatten_extra_dim(merged_predicts)


class TextDataMerger(DataMerger):

    def merge_predicts(self, predicts: List[np.array]) -> np.array:
        if len(predicts) > 1:
            raise ValueError("Text tables and merge of text data is not supported")
        return predicts[0]

    def postprocess_predicts(self, merged_predicts: np.array) -> np.array:
        return merged_predicts
