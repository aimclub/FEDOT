from typing import List, Dict

from fedot.core.data.data import OutputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.log import default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.data_types import TableTypesCorrector


class SupplementaryDataMerger:
    def __init__(self, outputs: List['OutputData'], main_output: 'OutputData'):
        self.outputs = outputs
        self.main_output = main_output
        self.log = default_log(self)

    def merge(self) -> SupplementaryData:
        return SupplementaryData(
            is_main_target=self.main_output.supplementary_data.is_main_target,
            data_flow_length=self.calculate_dataflow_len(),
            features_mask=self.prepare_parent_mask(),
            previous_operations=None,  # is set by Node after merge
            was_preprocessed=self.all_preprocessed(),
            non_int_idx=None,  # is set elsewhere (by preprocessor or during pipeline fit/predict)
            column_types=self.merge_column_types()
        )

    def calculate_dataflow_len(self) -> int:
        """ Number of visited nodes is the max number among outputs plus 1 (the next operation). """
        return 1 + max(output.supplementary_data.data_flow_length for output in self.outputs)

    def all_preprocessed(self) -> bool:
        return all(output.supplementary_data.was_preprocessed for output in self.outputs)

    def prepare_parent_mask(self) -> Dict:
        """ The method for OutputData from multiple parent nodes prepares a field
        with encoded values. This allows distinguishing the source ancestor to
        which the data was attached.

        For example, a mask for two ancestors, each giving predictions
        in the form of tabular data with two columns will look like this:
        {'input_ids': [0, 0, 1, 1], 'flow_lens': [2, 2, 1, 1]}

        Similarly, for images the mask will specify where the channels came from.
        Mask for two images (1st with 2 channels and the 2nd with 3) will look like:
        {'input_ids': [0, 0, 1, 1, 1], 'flow_lens': [1, 1, 3, 3, 3]}

        :return features_mask: dict with mask for features/channels.
        A composite ID of two lists is used:
            - index of parent operation
            - dataflow length (number of nodes that data visited)
        """
        input_ids = []
        flow_lens = []
        for input_id, output in enumerate(self.outputs):
            # Get number of features (for tables and series) or channels (for images)
            table_shape = output.predict.shape
            if len(table_shape) > 1:
                num_features = table_shape[-1]  # we need last dimension
            else:
                num_features = 1

            # Each index specifies the source of the feature
            id_mask = [input_id] * num_features
            input_ids.extend(id_mask)

            # Keep dataflow length for each feature separately
            flow_mask = [output.supplementary_data.data_flow_length] * num_features
            flow_lens.extend(flow_mask)

        features_mask = {'input_ids': input_ids, 'flow_lens': flow_lens}
        return features_mask

    def merge_column_types(self) -> Dict:
        """ Store information about column types in tabular data for merged data """
        if self.main_output.data_type is not DataTypesEnum.table:
            # Data is not tabular
            return self.main_output.supplementary_data.column_types

        # Concatenate types for features columns and
        #  choose target type of the main target as the new target type
        new_features_types = []
        new_target_types = None
        for output in self.outputs:
            if output.supplementary_data.column_types is None:
                self.log.debug(f'Perform determination of column types in DataMerger')
                table_corr = TableTypesCorrector()
                output.supplementary_data.column_types = table_corr.prepare_column_types_info(output.predict,
                                                                                              output.target,
                                                                                              output.task)
            col_types = output.supplementary_data.column_types['features']
            new_features_types.extend(col_types)

            if output.supplementary_data.is_main_target:
                # Target can be None for predict stage
                new_target_types = output.supplementary_data.column_types.get('target')

        column_types = {'features': new_features_types, 'target': new_target_types}
        return column_types
