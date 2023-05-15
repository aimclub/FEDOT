from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Tuple, Optional, List, Dict, Sequence

import numpy as np
import pandas as pd
from golem.core.log import LoggerAdapter, default_log

from fedot.core.repository.tasks import Task, TaskTypesEnum

if TYPE_CHECKING:
    from fedot.core.data.data import InputData

_convertable_types = (bool, float, int, str, type(None))  # preserve lexicographical order
_types_ids = range(len(_convertable_types))

TYPE_TO_ID = dict(zip(_convertable_types, _types_ids))

_TYPES = 'types'
_FLOAT_NUMBER = 'float_number'
_INT_NUMBER = 'int_number'
_STR_NUMBER = 'str_number'
_NAN_NUMBER = 'nan_number'
_NAN_IDS = 'nan_ids'

FEDOT_STR_NAN = 'fedot_nan'
# If unique values in the feature column is less than 13 - convert column into string type else to numerical
CATEGORICAL_MAX_UNIQUE_TH = 13
# column must be removed if failed rate is between these constants below
# because it means that in the column there are approximately the same number of truly string and ints/floats
ACCEPTABLE_CONVERSION_FAILED_RATE_BOTTOM = 0.4
ACCEPTABLE_CONVERSION_FAILED_RATE_TOP = 0.65


class TableTypesCorrector:
    """
    Class for checking types in input data. Also perform conversion for columns with types conflicts
    """

    def __init__(self):
        # Threshold to convert numerical into categorical column
        self.categorical_max_uniques_th = CATEGORICAL_MAX_UNIQUE_TH

        self.acceptable_failed_rate_bottom = ACCEPTABLE_CONVERSION_FAILED_RATE_BOTTOM
        self.acceptable_failed_rate_top = ACCEPTABLE_CONVERSION_FAILED_RATE_TOP

        self.features_columns_info = pd.DataFrame()
        self.target_columns_info = pd.DataFrame()

        # Dictionary with information about converted during fitting columns
        self.features_converted_columns = {}
        self.target_converted_columns = {}

        # Columns to delete due to types conflicts
        self.columns_to_del = []
        # Column ids for transformation due to number of unique values
        self.numerical_into_str = []
        self.categorical_into_float = []

        # Indices of columns with filed string into numerical transformation
        self.string_columns_transformation_failed = {}

        # Is target column contains non-numerical cells during conversion
        self.target_converting_has_errors = False

        # Lists with column types for converting calculated on source input data
        self.features_types = None
        self.target_types = None
        self.log = default_log(self)

    def convert_data_for_fit(self, data: InputData):
        """ If column contain several data types - perform correction procedure """
        # Convert features to have an ability to insert str into float table or vice versa
        data.features = data.features.astype(object)

        # Determine types for each column in features and target if it is necessary
        self.features_columns_info = define_column_types(data.features)
        self.target_columns_info = define_column_types(data.target)

        # Correct types in features table
        data.features = self.features_types_converting(features=data.features)
        # Remain only correct columns
        data.features = self.remove_incorrect_features(data.features, self.features_converted_columns)

        # And in target(s)
        data.target = self.target_types_converting(target=data.target, task=data.task)
        data.supplementary_data.column_types = self.prepare_column_types_info(predictors=data.features,
                                                                              target=data.target,
                                                                              task=data.task)

        self._into_numeric_features_transformation_for_fit(data)
        # Launch conversion float and integer features into categorical
        self._into_categorical_features_transformation_for_fit(data)
        # Save info about features and target types
        self.features_types = copy(data.supplementary_data.column_types['features'])
        self.target_types = copy(data.supplementary_data.column_types['target'])

        self._retain_columns_info_without_types_conflicts(data)
        return data

    def convert_data_for_predict(self, data: InputData):
        """ Prepare data for predict stage. Include only column types transformation """
        # Ordering is important because after removing incorrect features - indices are obsolete
        data.features = data.features.astype(object)
        data.features = self.remove_incorrect_features(data.features, self.features_converted_columns)
        data.features = apply_type_transformation(data.features, self.features_types, self.log)
        data.target = apply_type_transformation(data.target, self.target_types, self.log)
        data.supplementary_data.column_types = self.prepare_column_types_info(predictors=data.features,
                                                                              target=data.target,
                                                                              task=data.task)

        # Convert column types
        self._into_numeric_features_transformation_for_predict(data)
        self._into_categorical_features_transformation_for_predict(data)
        self._retain_columns_info_without_types_conflicts(data)
        return data

    def remove_incorrect_features(self, table: np.ndarray, converted_columns: dict):
        """
        Remove from the table columns with conflicts with types were not resolved

        :param table: tabular dataset based on which new dataset will be generated
        :param converted_columns: dictionary with actions with table
        """
        self.columns_to_del = [col_id for col_id, new_type_id in converted_columns.items() if new_type_id is None]
        table = np.delete(table, self.columns_to_del, 1)
        return table

    def features_types_converting(self, features: np.ndarray) -> np.ndarray:
        """ Convert all elements in the data in every feature column into one type

        :param features: tabular features array
        """
        mixed_types_columns = _find_mixed_types_columns(self.features_columns_info)
        cols_with_strings_or_floats = _select_from_rows_if_any(mixed_types_columns, [_STR_NUMBER, _FLOAT_NUMBER])

        def _update_converted_columns_and_data(column_info: pd.Series):
            updated_column, new_type_id = self._convert_feature_into_one_type(features[:, column_info.name],
                                                                              column_info)
            self.features_converted_columns[column_info.name] = new_type_id
            if updated_column is not None:
                features[:, column_info.name] = updated_column

        cols_with_strings_or_floats.apply(_update_converted_columns_and_data)

        return features

    def target_types_converting(self, target: np.ndarray, task: Task) -> np.ndarray:
        """ Convert all elements in every target column into one type

        :param target: tabular target array
        :param task: task to solve
        """
        mixed_types_columns = _find_mixed_types_columns(self.target_columns_info)
        cols_with_strings = _select_from_rows_if_any(mixed_types_columns, [_STR_NUMBER])

        def _update_converted_columns_and_data(column_info: pd.Series):
            updated_column, new_type_id = self._convert_target_into_one_type(target[:, column_info.name], column_info,
                                                                             task)
            self.target_converted_columns[column_info.name] = new_type_id
            if updated_column is not None:
                target[:, column_info.name] = updated_column

        cols_with_strings.apply(_update_converted_columns_and_data)

        return target

    def prepare_column_types_info(self, predictors: np.ndarray, target: np.ndarray = None,
                                  task: Task = None) -> dict:
        """ Prepare information about columns in a form of dictionary
        Dictionary has two keys: 'target' and 'features'
        """
        if self.features_columns_info.empty:
            # Information about column types is empty - there is a need to launch algorithm to collect info
            self.features_columns_info = define_column_types(predictors)
            predictors = self.features_types_converting(features=predictors)
        if self.target_columns_info.empty and task.task_type is not TaskTypesEnum.ts_forecasting:
            self.target_columns_info = define_column_types(target)
            target = self.target_types_converting(target=target, task=task)

        features_types = _generate_list_with_types(self.features_columns_info, self.features_converted_columns)
        self._check_columns_vs_types_number(predictors, features_types)

        if target is None or task.task_type is TaskTypesEnum.ts_forecasting:
            return {'features': features_types}
        else:
            target_types = _generate_list_with_types(self.target_columns_info, self.target_converted_columns)
            self._check_columns_vs_types_number(target, target_types)
            return {'features': features_types, 'target': target_types}

    def _retain_columns_info_without_types_conflicts(self, data: InputData):
        """ Update information in supplementary info - retain info only about remained columns.
        Such columns have no conflicts with types converting.
        """
        if self.string_columns_transformation_failed:
            self.log.warning(f'Columns with indices {self.string_columns_transformation_failed} were '
                             f'removed during mixed types column converting due to conflicts.')

            data.features = self.remove_incorrect_features(data.features, self.string_columns_transformation_failed)

            data.supplementary_data.column_types['features'] = np.delete(
                data.supplementary_data.column_types['features'],
                list(self.string_columns_transformation_failed)
            )

    def _check_columns_vs_types_number(self, table: np.ndarray, column_types: list):
        # Check if columns number correct
        _, n_cols = table.shape
        if n_cols != len(column_types):
            # There is an incorrect types calculation
            self.log.warning('Columns number and types numbers do not match.')

    @staticmethod
    def _remove_pseudo_str_values_from_str_column(data: InputData, columns: pd.Index):
        """ Removes from truly str column all pseudo str values """
        for col_id in columns:
            for row_id, item in enumerate(data.features[:, col_id]):
                try:
                    float(item)
                except ValueError:
                    continue
                else:
                    # item is numeric, remove its value
                    data.features[row_id, col_id] = np.nan

    def _convert_feature_into_one_type(self, mixed_column: np.ndarray, column_info: pd.Series):
        """ Determine new type for current feature column based on the string ratio. And then convert column into it.

        :param mixed_column: one-dimensional array with several data types
        :param column_info: dictionary with information about types in the column
        :param mixed_column_id: index of column in dataset
        """
        if len(column_info[_TYPES]) == 2 and TYPE_TO_ID[type(None)] in column_info[_TYPES]:
            # Column contain only one data type and nans
            filtered_types = [x for x in column_info[_TYPES] if x != TYPE_TO_ID[type(None)]]
            return mixed_column, filtered_types[0]

        string_objects_number = column_info[_STR_NUMBER]
        all_elements_number = string_objects_number + column_info[[_INT_NUMBER, _FLOAT_NUMBER]].sum()
        string_ratio = string_objects_number / all_elements_number

        if string_ratio > 0:
            suggested_type = str
        else:
            suggested_type = _obtain_new_column_type(column_info)

        try:
            mixed_column = mixed_column.astype(suggested_type)
            # If there were nans in the column - paste nan
            if column_info[_NAN_NUMBER]:
                mixed_column = mixed_column.astype(object)
                mixed_column[column_info[_NAN_IDS]] = np.nan
                del column_info[_NAN_IDS]
            return mixed_column, TYPE_TO_ID[suggested_type]
        except ValueError:
            # Cannot convert string objects into int or float (for example 'a' into int)
            prefix = f'Feature column with index {column_info.name} contains ' \
                     f'following data types: {column_info[_TYPES]}.'
            self.log.warning(f'{prefix} String cannot be converted into {suggested_type}. Drop column.')
            return None, None

    def _convert_target_into_one_type(self, mixed_column: np.ndarray, column_info: pd.Series,
                                      task: Task) -> Tuple[np.ndarray, str]:
        """ Convert target columns into one type based on column proportions of object and task """
        if task.task_type is TaskTypesEnum.classification:
            # For classification labels are string if at least one element is a string
            suggested_type = str
        else:
            suggested_type = _obtain_new_column_type(column_info)

        try:
            mixed_column = mixed_column.astype(suggested_type)
            return mixed_column, TYPE_TO_ID[suggested_type]
        except ValueError:
            # Cannot convert string objects into int or float (for example 'a' into int)
            target_column = pd.Series(mixed_column)
            converted_column = pd.to_numeric(target_column, errors='coerce')

            prefix = (f'Target column with index {column_info.name} contains '
                      f'following data types: {column_info[_TYPES]}.')
            log_message = f'{prefix} String cannot be converted into {suggested_type}. Ignore non converted values.'
            self.log.debug(log_message)
            self.target_converting_has_errors = True
            return converted_column.values, TYPE_TO_ID[suggested_type]

    def _into_categorical_features_transformation_for_fit(self, data: InputData):
        """
        Perform automated categorical features determination. If feature column
        contains int or float values with few unique values (less than 13)
        """
        features_types = data.supplementary_data.column_types['features']
        is_numeric_type = np.isin(features_types, [TYPE_TO_ID[int], TYPE_TO_ID[float]])
        numeric_type_ids = np.flatnonzero(is_numeric_type)
        num_df = pd.DataFrame(data.features[:, numeric_type_ids], columns=numeric_type_ids)
        nuniques = num_df.nunique(dropna=True)
        # reduce dataframe to include only categorical features
        num_df = num_df.loc[:, (2 < nuniques) & (nuniques < self.categorical_max_uniques_th)]
        cat_col_ids = num_df.columns
        # Convert into string
        data.features[:, cat_col_ids] = num_df.apply(convert_num_column_into_string_array).to_numpy()
        # Columns need to be transformed into categorical (string) ones
        self.numerical_into_str.extend(cat_col_ids.difference(self.numerical_into_str))
        # Update information about column types (in-place)
        features_types[cat_col_ids] = TYPE_TO_ID[str]

    def _into_categorical_features_transformation_for_predict(self, data: InputData):
        """ Apply conversion into categorical string column for every signed column """
        if not self.numerical_into_str:
            # There is no transformation for current table
            return data

        # Get numerical columns
        num_df = pd.DataFrame(data.features[:, self.numerical_into_str], columns=self.numerical_into_str)

        # Convert and apply categorical transformation
        data.features[:, self.numerical_into_str] = num_df.apply(convert_num_column_into_string_array).to_numpy()

        # Update information about column types (in-place)
        features_types = data.supplementary_data.column_types['features']
        features_types[self.numerical_into_str] = TYPE_TO_ID[str]

    def _into_numeric_features_transformation_for_fit(self, data: InputData):
        """
        Automatically determine categorical features which should be converted into float
        """
        str_columns = np.flatnonzero(
            np.isin(data.supplementary_data.column_types['features'], TYPE_TO_ID[str])
        )
        str_cols_df = pd.DataFrame(data.features[:, str_columns], columns=str_columns)
        orig_nans_cnt = str_cols_df.isna().sum(axis=0)

        converted_str_cols_df = str_cols_df.apply(pd.to_numeric, errors='coerce')
        conv_nans_cnt = converted_str_cols_df.isna().sum(axis=0)

        failed_objects_cnt = conv_nans_cnt - orig_nans_cnt
        non_nan_all_objects_cnt = len(data.features) - orig_nans_cnt
        failed_ratio = failed_objects_cnt / non_nan_all_objects_cnt

        # Check if the majority of objects can be converted into numerical
        is_numeric = failed_ratio < self.acceptable_failed_rate_bottom
        is_numeric_ids = is_numeric[is_numeric].index
        data.features[:, is_numeric_ids] = converted_str_cols_df[is_numeric_ids].to_numpy()
        self.categorical_into_float.extend(is_numeric_ids.difference(self.categorical_into_float))

        # Update information about column types (in-place)
        features_types = data.supplementary_data.column_types['features']
        features_types[is_numeric_ids] = TYPE_TO_ID[float]

        # The columns consists mostly of truly str values and has a few ints/floats in it
        is_mixed = (self.acceptable_failed_rate_top <= failed_ratio) & (failed_ratio != 1)
        self._remove_pseudo_str_values_from_str_column(data, is_mixed[is_mixed].index)

        # If column contains a lot of '?' or 'x' as nans equivalents
        # add it remove list
        is_of_mistakes = (
                (self.acceptable_failed_rate_bottom <= failed_ratio)
                & (failed_ratio < self.acceptable_failed_rate_top))
        self.string_columns_transformation_failed.update(dict.fromkeys(is_of_mistakes[is_of_mistakes].index))

    def _into_numeric_features_transformation_for_predict(self, data: InputData):
        """ Apply conversion into float string column for every signed column """
        str_cols_ids = list(set(self.categorical_into_float)
                            .difference(self.string_columns_transformation_failed))
        str_cols_df = pd.DataFrame(data.features[:, str_cols_ids], columns=str_cols_ids)
        data.features[:, str_cols_ids] = str_cols_df.apply(pd.to_numeric, errors='coerce').to_numpy()

        # Update information about column types (in-place)
        features_types = data.supplementary_data.column_types['features']
        features_types[str_cols_ids] = TYPE_TO_ID[float]


def define_column_types(table: Optional[np.ndarray]) -> pd.DataFrame:
    """ Prepare information about types per columns. For each column store unique
    types, which column contains.
    """
    if table is None:
        return pd.DataFrame()

    table_of_types = pd.DataFrame(table, copy=True)
    table_of_types = table_of_types.applymap(lambda el: TYPE_TO_ID[type(None if pd.isna(el) else el)]).astype(np.int8)

    # Build dataframe with unique types for each column
    uniques = table_of_types.apply([pd.unique]).rename(index={'unique': _TYPES})

    # Build dataframe with amount of each type
    counts_index_mapper = {
        TYPE_TO_ID[float]: _FLOAT_NUMBER,
        TYPE_TO_ID[int]: _INT_NUMBER,
        TYPE_TO_ID[str]: _STR_NUMBER,
        TYPE_TO_ID[type(None)]: _NAN_NUMBER
    }
    types_counts = (
        table_of_types
            .apply(pd.value_counts, dropna=False)
            .reindex(counts_index_mapper.keys(), copy=False)
            .replace(np.nan, 0)
            .rename(index=counts_index_mapper, copy=False)
            .astype(int)
    )

    # Build dataframe with nans indices
    nans_ids = (table_of_types == TYPE_TO_ID[type(None)]).apply(np.where).rename(index={0: _NAN_IDS})

    # Combine all dataframes
    return pd.concat([uniques, types_counts, nans_ids])


def _find_mixed_types_columns(columns_info: pd.DataFrame) -> pd.DataFrame:
    """ Search for columns with several types in them """
    has_mixed_types = [] if columns_info.empty else columns_info.loc[_TYPES].apply(len) > 1
    return columns_info.loc[:, has_mixed_types]


def _select_from_rows_if_any(frame: pd.DataFrame, rows_to_select: List[str]) -> pd.DataFrame:
    _cols_have_any = [] if frame.empty else frame.loc[rows_to_select].any()
    return frame.loc[:, _cols_have_any]


def apply_type_transformation(table: np.ndarray, column_types: Sequence, log: LoggerAdapter):
    """
    Apply transformation for columns in dataset into desired type. Perform
    transformation on predict stage when column types were already determined
    during fit
    """

    def type_by_id(current_type_id: int):
        """ Return type by its ID """
        if current_type_id == TYPE_TO_ID[int]:
            return int
        elif current_type_id == TYPE_TO_ID[str]:
            return str
        return float

    if table is None:
        # Occurs if for predict stage there is no target info
        return None

    _, n_cols = table.shape
    for column_id in range(n_cols):
        current_column = table[:, column_id]
        current_type = type_by_id(column_types[column_id])
        _convert_predict_column_into_desired_type(table=table, current_column=current_column, current_type=current_type,
                                                  column_id=column_id, log=log)

    return table


def convert_num_column_into_string_array(numerical_column: pd.Series) -> pd.Series:
    """ Convert pandas column into numpy one-dimensional array """
    # convert only non-nans values
    true_nums = numerical_column[numerical_column.notna()]
    numerical_column[true_nums.index] = true_nums.astype(str, copy=False)
    return numerical_column


def _obtain_new_column_type(column_info: pd.Series):
    """ Suggest in or float type based on the presence of nan and float values """
    if column_info[[_FLOAT_NUMBER, _NAN_NUMBER]].any():
        # Even if one of types are float - all elements should be converted into float
        return float
    # It is available to convert numerical into integer type
    return int


def _convert_predict_column_into_desired_type(table: np.ndarray, current_column: np.ndarray,
                                              column_id: int, current_type: type, log: LoggerAdapter):
    try:
        table[:, column_id] = current_column.astype(current_type)
        if current_type is str:
            is_any_comma = any(',' in el for el in current_column)
            is_any_dot = any('.' in el for el in current_column)
            # Most likely case: '20,000' must be converted into '20.000'
            if is_any_comma and is_any_dot:
                warning = f'Column {column_id} contains both "." and ",". Standardize it.'
                log.warning(warning)
    except ValueError:
        table[:, column_id] = _process_predict_column_values_one_by_one(current_column=current_column,
                                                                        current_type=current_type)


def _generate_list_with_types(columns_types_info: pd.DataFrame,
                              converted_columns: Dict[int, Optional[int]]) -> np.ndarray:
    """ Create list with types for all remained columns

    :param columns_types_info: dictionary with initial column types
    :param converted_columns: dictionary with transformed column types
    """
    updated_column_types = []

    for column_id, column_type_ids in columns_types_info.loc[_TYPES].items():
        if len(column_type_ids) == 1:
            # Column initially contain only one type
            updated_column_types.append(column_type_ids[0])
        elif len(column_type_ids) == 2 and TYPE_TO_ID[type(None)] in column_type_ids:
            # Column with one type and nans
            filtered_types = [x for x in column_type_ids if x != TYPE_TO_ID[type(None)]]
            updated_column_types.append(filtered_types[0])
        else:
            if TYPE_TO_ID[str] in column_type_ids:
                # Mixed-types column with string
                new_col_id = converted_columns[column_id]
                if new_col_id is not None:
                    updated_column_types.append(new_col_id)
            else:
                # Mixed-types with float and integer
                updated_column_types.append(TYPE_TO_ID[float])

    return np.array(updated_column_types)


def _process_predict_column_values_one_by_one(current_column: np.ndarray, current_type: type):
    """ Process column values one by one and try to convert them into desirable type.
    If not successful replace with np.nan """

    def _process_str_numbers_with_dots_and_commas(value: str):
        """ Try to process str with replacing ',' by '.' in case it was meant to be a number """
        value = value.replace(',', '.')
        new_value = np.nan
        try:
            # Since "10.6" can not be converted to 10 straightforward using int()
            if current_type is int:
                new_value = int(float(value))
        except ValueError:
            pass
        return new_value

    new_column = []
    for value in current_column:
        new_value = np.nan
        try:
            new_value = current_type(value)
        except ValueError:
            if isinstance(value, str) and ('.' in value or ',' in value):
                new_value = _process_str_numbers_with_dots_and_commas(value=value)
        new_column.append(new_value)
    return new_column
