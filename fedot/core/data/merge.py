from typing import List

import numpy as np

from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.data_types import TableTypesCorrector


class DataMerger:
    """
    Class for merging data, when it comes from different nodes and there is a
    need to merge it into next level node

    :param outputs: list with OutputData from parent nodes
    """

    def __init__(self, outputs: list, log: Log = None):
        self.outputs = outputs
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def merge(self):
        """
        Method automatically determine which merge function should be applied
        """

        if len(self.outputs) == 1 and self.outputs[0].data_type in [DataTypesEnum.image]:
            # TODO implement correct merge
            idx = self.outputs[0].idx
            features = self.outputs[0].features
            target = self.outputs[0].target
            task = self.outputs[0].task
            data_type = self.outputs[0].data_type

            if self.outputs[0].supplementary_data is None:
                updated_info = SupplementaryData(is_main_target=True)
            else:
                updated_info = self.outputs[0].supplementary_data
            updated_info.calculate_data_flow_len(self.outputs)
            return idx, features, target, task, data_type, updated_info

        merge_function_by_type = {DataTypesEnum.ts: self.combine_datasets_ts,
                                  DataTypesEnum.multi_ts: self.combine_datasets_table,
                                  DataTypesEnum.table: self.combine_datasets_table,
                                  DataTypesEnum.text: self.combine_datasets_table}

        output_data_types = [output.data_type for output in self.outputs]
        first_data_type = output_data_types[0]

        # Check is all data types can be merged or not
        if len(set(output_data_types)) > 1:
            raise ValueError('There is no ability to merge different data types')

        # Define appropriate strategy
        merge_func = merge_function_by_type.get(first_data_type)
        if merge_func is None:
            message = f"For data type '{first_data_type}' doesn't exist merge function"
            raise NotImplementedError(message)
        else:
            idx, features, target, is_main_target, task = merge_func()

        updated_info = SupplementaryData(is_main_target=is_main_target)
        # Calculate amount of visited nodes for data
        updated_info.calculate_data_flow_len(self.outputs)
        # Prepare mask with predict from different parent nodes
        if first_data_type == DataTypesEnum.table and len(self.outputs) > 1:
            updated_info.prepare_parent_mask(self.outputs)

        updated_info = self.update_column_types(updated_info, first_data_type)
        return idx, features, target, task, first_data_type, updated_info

    def combine_datasets_table(self):
        """ Function for combining datasets from parents to make features to
        another node. Features are tabular data.

        :return idx: updated indices
        :return features: new features obtained from predictions at previous level
        :return target: updated target
        """
        are_lengths_equal, idx_list = self._check_size_equality(self.outputs)

        if are_lengths_equal:
            idx, features, target, is_main_target, task = self._merge_equal_outputs(self.outputs)
        else:
            idx, features, target, is_main_target, task = self._merge_non_equal_outputs(self.outputs,
                                                                                        idx_list)

        return idx, features, target, is_main_target, task

    def combine_datasets_ts(self):
        """ Function for combining datasets from parents to make features to
        another node. Features are time series data.

        :return idx: updated indices
        :return features: new features obtained from predictions at previous level
        :return target: updated target
        """
        are_lengths_equal, idx_list = self._check_size_equality(self.outputs)

        if are_lengths_equal:
            idx, features, target, is_main_target, task = self._merge_equal_outputs(self.outputs)
        else:
            idx, features, target, is_main_target, task = self._merge_non_equal_outputs(self.outputs,
                                                                                        idx_list)

        if len(features.shape) > 1 and features.shape[1] == 1:
            features = np.ravel(np.array(features))
        if target is not None and len(target.shape) > 1 and target.shape[1] == 1:
            target = np.ravel(np.array(target))
        return idx, features, target, is_main_target, task

    def update_column_types(self, supplementary_data: SupplementaryData, data_type: DataTypesEnum):
        """ Store information about column types in tabular data for merged data """
        if data_type is not DataTypesEnum.table:
            # Data is not tabular
            return supplementary_data

        # Types for features columns
        new_features_types = []
        for output in self.outputs:
            if output.supplementary_data.column_types is None:
                self.log.debug(f'Perform determination of column types in DataMerger')
                table_corr = TableTypesCorrector()
                output.supplementary_data.column_types = table_corr.prepare_column_types_info(output.predict,
                                                                                              output.target,
                                                                                              output.task)
            col_types = output.supplementary_data.column_types['features']
            new_features_types.extend(col_types)

        # Target(s) types
        new_target_types = None
        for output in self.outputs:
            if output.supplementary_data.column_types is None:
                self.log.debug(f'Perform determination of column types in DataMerger')
                table_corr = TableTypesCorrector()
                output.supplementary_data.column_types = table_corr.prepare_column_types_info(output.predict,
                                                                                              output.target,
                                                                                              output.task)

            # Search for main target
            if output.supplementary_data.is_main_target:
                # Target can be None for predict stage
                new_target_types = output.supplementary_data.column_types.get('target')
        supplementary_data.column_types = {'features': new_features_types,
                                           'target': new_target_types}
        return supplementary_data

    @staticmethod
    def _merge_equal_outputs(outputs: list):
        """ Method merge datasets with equal amount of rows """

        features = []
        for elem in outputs:
            if len(elem.predict.shape) == 1:
                features.append(elem.predict)
            else:
                # If the model prediction is multivariate
                number_of_variables_in_prediction = elem.predict.shape[1]
                for i in range(number_of_variables_in_prediction):
                    features.append(elem.predict[:, i])

        features = np.array(features).T
        idx = outputs[0].idx

        # Update target from multiple parents
        target, is_main_target, task = TaskTargetMerger(outputs).obtain_equal_target()

        return idx, features, target, is_main_target, task

    @staticmethod
    def _merge_non_equal_outputs(outputs: list, idx_list: List):
        """ Method merge datasets with different amount of rows by idx field """

        # Search overlapping indices in data
        for i, idx in enumerate(idx_list):
            idx = set(idx)
            if i == 0:
                common_idx = idx
            else:
                common_idx = common_idx & idx

        if len(common_idx) == 0:
            raise ValueError(f'There are no common indices for outputs')

        idx_list = [list(output.idx) for output in outputs]
        predicts = [output.predict for output in outputs]

        # Generate feature table with overlapping ids
        features = tables_mapping(idx_list, predicts, common_idx)
        # Link tables with features into one table - rotate array
        features = np.hstack(features)

        # Merge tasks and targets
        t_merger = TaskTargetMerger(outputs)
        filtered_target, is_main_target, task = t_merger.obtain_non_equal_target(common_idx)
        return common_idx, features, filtered_target, is_main_target, task

    @staticmethod
    def _check_size_equality(outputs: list):
        """ Function check the size of combining datasets """
        idx_lengths = []
        idx_list = []
        for elem in outputs:
            idx_lengths.append(len(elem.idx))
            idx_list.append(elem.idx)

        # Check amount of unique lengths of datasets
        if len(set(idx_lengths)) == 1:
            are_lengths_equal = True
        else:
            are_lengths_equal = False

        return are_lengths_equal, idx_list


class TaskTargetMerger:
    """ Class for merging target and tasks """

    def __init__(self, outputs):
        self.outputs = outputs

    def obtain_equal_target(self):
        """ Method can merge different targets if the amount of objects in the
        training sample are equal
        """
        # If there is only one parent
        if len(self.outputs) == 1:
            target = self.outputs[0].target
            is_main_target = self.outputs[0].supplementary_data.is_main_target
            task = self.outputs[0].task
            return target, is_main_target, task

        # Get target flags, targets and tasks
        t_flags, targets, tasks = self._disintegrate_outputs()

        # If all t_flags are True - there is no need to merge targets
        if all(flag is True for flag in t_flags):
            main_outputs = [o for o in self.outputs if o.supplementary_data.is_main_target]

            target = main_outputs[0].target
            task = main_outputs[0].task
            is_main_target = True
            return target, is_main_target, task
        # If there is an "ignore" (False) flag - need to apply intelligent merge
        elif any(flag is False for flag in t_flags):
            target, is_main_target, task = self.ignored_merge(targets, t_flags, tasks)
            return target, is_main_target, task

    def obtain_non_equal_target(self, common_idx):
        """ Method for merging targets which have different amount of objects
        (amount of rows)

        :param common_idx: array with indices of common objects
        """

        t_flags, targets, tasks = self._disintegrate_outputs()

        if targets[0] is None:
            mapped_targets = [None]
        else:
            # Match targets - make them equal
            idx_list = [output.idx for output in self.outputs]
            mapped_targets = tables_mapping(idx_list, targets, common_idx)

        # If all t_flags are True - there is no need to merge targets
        if all(flag is True for flag in t_flags):
            # Just applying merge operation for common_idx
            filtered_target = mapped_targets[0]

            task = tasks[0]
            is_main_target = True
            return filtered_target, is_main_target, task
        elif any(flag is False for flag in t_flags):

            filtered_target, is_main_target, task = self.ignored_merge(mapped_targets,
                                                                       t_flags,
                                                                       tasks)
            return filtered_target, is_main_target, task

    def _disintegrate_outputs(self):
        """
        Method extract target flags, targets and tasks from list with OutputData
        """
        t_flags = [output.supplementary_data.is_main_target for output in self.outputs]
        targets = [output.target for output in self.outputs]
        tasks = [output.task for output in self.outputs]

        return t_flags, targets, tasks

    @staticmethod
    def ignored_merge(targets, t_flags, tasks):
        """ Method merge targets with False labels. False - means that such
        branch target must be ignored """
        # PEP8 fix through converting boolean into string
        t_flags = np.array(t_flags, dtype=str)
        main_ids = np.ravel(np.argwhere(t_flags == 'True'))
        tasks = np.array(tasks)

        # Is there is pipeline predict stage without target at all
        if targets[0] is None:
            target = None
            is_main_target = True
        # If there are several known targets
        else:
            # Take first non-ignored target
            main_id = main_ids[0]
            target = targets[main_id]
            if len(target.shape) == 1:
                target = target.reshape((-1, 1))
            is_main_target = True

        task = tasks[main_ids]
        task = task[0]
        return target, is_main_target, task


def tables_mapping(idx_list, object_list, common_idx):
    """ The function maps tables by matching object indices

    :param idx_list: list with indices for mapping
    :param object_list: list with tables (with features, targets or predictions)
     for mapping
    :param common_idx: list with common indices

    :return : list with matched tables
    """

    common_tables = []

    # if indices repeats (for multi_ts data type)
    repeats_num = int(min(obj.shape[0] for obj in object_list)/len(common_idx))
    if repeats_num > 1:
        common_idx_full = list(common_idx) * repeats_num
    else:
        common_idx_full = list(common_idx)

    for number in range(len(idx_list)):
        current_idx = list(idx_list[number])
        current_object = object_list[number]

        if len(current_idx) < current_object.shape[0]:
            repeats_num = int(current_object.shape[0]/len(current_idx))
            current_idx = list(current_idx) * repeats_num
        # Create mask where True - appropriate objects
        current_idx = np.array(current_idx, dtype=int)
        common_idx_full = np.array(common_idx_full)
        mask = np.isin(current_idx, common_idx_full)

        if len(current_object.shape) == 1:
            filtered_predict = current_object[mask]
            filtered_predict = filtered_predict.reshape((-1, 1))
            common_tables.append(filtered_predict)
        else:
            # If the table object has many columns
            number_of_variables_in_prediction = current_object.shape[1]
            for i in range(number_of_variables_in_prediction):
                predict = current_object[:, i]
                filtered_predict = predict[mask]

                # Convert to column
                filtered_predict = filtered_predict.reshape((-1, 1))
                if i == 0:
                    filtered_table = filtered_predict
                else:
                    filtered_table = np.hstack((filtered_table, filtered_predict))
            common_tables.append(filtered_table)
    return common_tables
