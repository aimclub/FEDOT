from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from golem.core.log import LoggerAdapter, default_log

from fedot.core.repository.tasks import Task, TaskTypesEnum

if TYPE_CHECKING:
    from fedot.core.data.data import InputData

_convertable_types = (bool, float, int, str, type(None))  # preserve lexicographical order
_type_ids = range(len(_convertable_types))

TYPE_TO_ID = dict(zip(_convertable_types, _type_ids))
ID_TO_TYPE = dict(zip(_type_ids, _convertable_types))

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
        self.feature_type_ids = None
        self.target_type_ids = None
        self.log = default_log(self)

    def convert_data_for_fit(self, data: InputData):
        """ If column contain several data types - perform correction procedure """
        # Convert features to have an ability to insert str into float table or vice versa
        data.features = data.features.astype(object)

        # Determine types for each column in features and target if it is necessary
        self.features_columns_info = define_column_types(data.features)
        self.target_columns_info = define_column_types(data.target)

        # Correct types in features table
        data.features = self.feature_types_converting(features=data.features)
        # Remain only correct columns
        data.features = self.remove_incorrect_features(data.features, self.features_converted_columns)

        # And in target(s)
        data.target = self.target_types_converting(target=data.target, task=data.task)
        column_types_info = self.prepare_column_types_info(predictors=data.features, target=data.target, task=data.task)
        data.supplementary_data.col_type_ids = column_types_info
        col_types_info_message = prepare_log_message_with_cols_types(column_types_info, data.features_names)
        self.log.debug(f'--- The detected types of data are as follows: {col_types_info_message}')
        self._into_numeric_features_transformation_for_fit(data)
        # Launch conversion float and integer features into categorical
        self._into_categorical_features_transformation_for_fit(data)
        # Save info about features and target types
        self.feature_type_ids = data.supplementary_data.col_type_ids['features'].copy()
        self.target_type_ids = data.supplementary_data.col_type_ids.get(
            'target', np.empty((self.feature_type_ids.shape[0], 1), dtype=float)
        ).copy()

        self._retain_columns_info_without_types_conflicts(data)
        return data

    def convert_data_for_predict(self, data: InputData):
        """ Prepare data for predict stage. Include only column types transformation """
        # Ordering is important because after removing incorrect features - indices are obsolete
        data.features = data.features.astype(object)
        data.features = self.remove_incorrect_features(data.features, self.features_converted_columns)
        data.features = apply_type_transformation(data.features, self.feature_type_ids, self.log)
        if data.target is not None:
            data.target = apply_type_transformation(data.target, self.target_type_ids, self.log)
        data.supplementary_data.col_type_ids = self.prepare_column_types_info(predictors=data.features,
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

    def feature_types_converting(self, features: np.ndarray) -> np.ndarray:
        """ Convert all elements in the data in every feature column into one type

        :param features: tabular features array
        """
        mixed_types_columns = _find_mixed_types_columns(self.features_columns_info)
        cols_with_strings_or_floats = _select_from_rows_if_any(mixed_types_columns, [_STR_NUMBER, _FLOAT_NUMBER])
        cols_with_strings_or_floats.apply(self._convert_feature_into_one_type, features=features)

        return features

    def target_types_converting(self, target: np.ndarray, task: Task) -> np.ndarray:
        """ Convert all elements in every target column into one type

        :param target: tabular target array
        :param task: task to solve
        """
        mixed_types_columns = _find_mixed_types_columns(self.target_columns_info)
        cols_with_strings = _select_from_rows_if_any(mixed_types_columns, [_STR_NUMBER])
        cols_with_strings.apply(self._convert_target_into_one_type, target=target, task=task)

        return target

    def prepare_column_types_info(self, predictors: np.ndarray, target: np.ndarray = None,
                                  task: Task = None) -> dict:
        """ Prepare information about columns in a form of dictionary.
        Dictionary has two keys: 'target' and 'features'
        """
        if self.features_columns_info.empty:
            # Information about column types is empty - there is a need to launch algorithm to collect info
            self.features_columns_info = define_column_types(predictors)
            predictors = self.feature_types_converting(features=predictors)
        if self.target_columns_info.empty and task.task_type is not TaskTypesEnum.ts_forecasting:
            self.target_columns_info = define_column_types(target)
            target = self.target_types_converting(target=target, task=task)

        feature_type_ids = _generate_list_with_types(self.features_columns_info, self.features_converted_columns)
        self._check_columns_vs_types_number(predictors, feature_type_ids)

        if target is None or task.task_type is TaskTypesEnum.ts_forecasting:
            return {'features': feature_type_ids}
        else:
            target_type_ids = _generate_list_with_types(self.target_columns_info, self.target_converted_columns)
            self._check_columns_vs_types_number(target, target_type_ids)
            return {'features': feature_type_ids, 'target': target_type_ids}

    def _retain_columns_info_without_types_conflicts(self, data: InputData):
        """ Update information in supplementary info - retain info only about remained columns.
        Such columns have no conflicts with types converting.
        """
        if self.string_columns_transformation_failed:
            self.log.message(f'Columns with indices {self.string_columns_transformation_failed} were '
                             f'removed during mixed types column converting due to conflicts.')

            data.features = self.remove_incorrect_features(data.features, self.string_columns_transformation_failed)

            data.supplementary_data.col_type_ids['features'] = np.delete(
                data.supplementary_data.col_type_ids['features'],
                list(self.string_columns_transformation_failed)
            )

    def _check_columns_vs_types_number(self, table: np.ndarray, col_type_ids: Sequence):
        # Check if columns number correct
        _, n_cols = table.shape
        if n_cols != len(col_type_ids):
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

    def _convert_feature_into_one_type(self, column_info: pd.Series, features: np.ndarray):
        """ Determine new type for current feature column based on the string ratio. And then convert column into it.

        :param features: one-dimensional array with several data types
        :param column_info: dictionary with information about types in the column
        :param mixed_column_id: index of column in dataset
        """
        new_type_id = None
        if len(column_info[_TYPES]) == 2 and TYPE_TO_ID[type(None)] in column_info[_TYPES]:
            # Column contain only one data type and nans
            non_nan_type_lst = [x for x in column_info[_TYPES] if x != TYPE_TO_ID[type(None)]]
            new_type_id = non_nan_type_lst[0]
        else:
            string_objects_number = column_info[_STR_NUMBER]
            all_elements_number = string_objects_number + column_info[[_INT_NUMBER, _FLOAT_NUMBER]].sum()
            string_ratio = string_objects_number / all_elements_number

            if string_ratio > 0:
                suggested_type = str
            else:
                suggested_type = _obtain_new_column_type(column_info)

            try:
                converted = features[:, column_info.name].astype(suggested_type)
                # If there were nans in the column - paste nan
                if column_info[_NAN_NUMBER]:
                    converted = converted.astype(object)
                    converted[column_info[_NAN_IDS]] = np.nan
                    del column_info[_NAN_IDS]
                features[:, column_info.name] = converted
            except ValueError:
                # Cannot convert string objects into int or float (for example 'a' into int)
                prefix = (f'Feature column with index {column_info.name} contains '
                          f'the following data types: {column_info[_TYPES]}.')
                self.log.warning(f'{prefix} String cannot be converted into {suggested_type}. Drop column.')
            else:
                new_type_id = TYPE_TO_ID[suggested_type]
        self.features_converted_columns[column_info.name] = new_type_id

    def _convert_target_into_one_type(self, column_info: pd.Series, target: np.ndarray,
                                      task: Task) -> Tuple[np.ndarray, str]:
        """ Convert target columns into one type based on column proportions of object and task """
        if task.task_type is TaskTypesEnum.classification:
            # For classification labels are string if at least one element is a string
            suggested_type = str
        else:
            suggested_type = _obtain_new_column_type(column_info)
        self.target_converted_columns[column_info.name] = TYPE_TO_ID[suggested_type]

        mixed_column = target[:, column_info.name]
        try:
            target[:, column_info.name] = mixed_column.astype(suggested_type)
        except ValueError:
            # Cannot convert string objects into int or float (for example 'a' into int)
            converted_column = pd.to_numeric(mixed_column, errors='coerce')

            prefix = (f'Target column with index {column_info.name} contains '
                      f'the following data types: {column_info[_TYPES]}.')
            log_message = f'{prefix} String cannot be converted into {suggested_type}. Set unconverted values to NaN.'
            self.log.debug(log_message)
            self.target_converting_has_errors = True
            target[:, column_info.name] = converted_column

    def _into_categorical_features_transformation_for_fit(self, data: InputData):
        """
        Perform automated categorical features determination. If feature column
        contains int or float values with few unique values (less than 13)
        """
        if data.categorical_idx is None:
            feature_type_ids = data.supplementary_data.col_type_ids['features']
            is_numeric_type = np.isin(feature_type_ids, [TYPE_TO_ID[int], TYPE_TO_ID[float]])
            numeric_type_ids = np.flatnonzero(is_numeric_type)
            num_df = pd.DataFrame(data.features[:, numeric_type_ids], columns=numeric_type_ids)
            nuniques = num_df.nunique(dropna=True)

            # TODO: Improve the naive approach (with categorical_max_uniques_th) of identifying categorical data
            #  to a smarter approach (eg. numeric, features naming with llm)
            # reduce dataframe to include only categorical features
            num_df = num_df.loc[:, (2 < nuniques) & (nuniques < self.categorical_max_uniques_th)]

            if data.categorical_idx is not None:
                # If cats features were defined take it
                cat_col_ids = data.categorical_idx
            else:
                # Else cats features are selected by heuristic rule
                cat_col_ids = num_df.columns

            if np.size(cat_col_ids) > 0:
                # Convert into string
                data.features[:, cat_col_ids] = num_df.apply(
                    convert_num_column_into_string_array).to_numpy()
                # Columns need to be transformed into categorical (string) ones
                self.numerical_into_str.extend(cat_col_ids.difference(self.numerical_into_str))
                # Update information about column types (in-place)
                feature_type_ids[cat_col_ids] = TYPE_TO_ID[str]

            # Update cat cols idx in data
            is_cat_type = np.isin(feature_type_ids, [TYPE_TO_ID[str]])
            all_cat_col_ids = np.flatnonzero(is_cat_type)
            data.categorical_idx = all_cat_col_ids

            # Update num cols idx in data
            is_numeric_type = np.isin(feature_type_ids, [TYPE_TO_ID[int], TYPE_TO_ID[float]])
            all_numeric_type_ids = np.flatnonzero(is_numeric_type)
            data.numerical_idx = all_numeric_type_ids

            if np.size(all_cat_col_ids) > 0:
                if data.features_names is not None:
                    cat_features_names = data.features_names[all_cat_col_ids]
                    self.log.info(
                        f'Preprocessing defines the following columns as categorical: {cat_features_names}'
                    )
                else:
                    self.log.info(
                        f'Preprocessing defines the following columns as categorical: {all_cat_col_ids}'
                    )
            else:
                self.log.info('Preprocessing was unable to define the categorical columns')

    def _into_categorical_features_transformation_for_predict(self, data: InputData):
        """ Apply conversion into categorical string column for every signed column """
        # Get numerical columns
        num_df = pd.DataFrame(data.features[:, self.numerical_into_str], columns=self.numerical_into_str)

        # Convert and apply categorical transformation
        data.features[:, self.numerical_into_str] = num_df.apply(convert_num_column_into_string_array).to_numpy()

        # Update information about column types (in-place)
        feature_type_ids = data.supplementary_data.col_type_ids['features']
        feature_type_ids[self.numerical_into_str] = TYPE_TO_ID[str]

    def _into_numeric_features_transformation_for_fit(self, data: InputData):
        """
        Automatically determine categorical features which should be converted into float
        """
        is_str_type = data.supplementary_data.col_type_ids['features'] == TYPE_TO_ID[str]
        str_col_ids = np.flatnonzero(is_str_type)
        str_cols_df = pd.DataFrame(data.features[:, str_col_ids], columns=str_col_ids)
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
        feature_type_ids = data.supplementary_data.col_type_ids['features']
        feature_type_ids[is_numeric_ids] = TYPE_TO_ID[float]

        # The columns consist mostly of truly str values and has a few ints/floats in it
        is_mixed = (self.acceptable_failed_rate_top <= failed_ratio) & (failed_ratio != 1)
        self._remove_pseudo_str_values_from_str_column(data, is_mixed[is_mixed].index)

        # If column contains a lot of '?' or 'x' as nans equivalents
        # add it to remove list
        is_of_mistakes = (
            (self.acceptable_failed_rate_bottom <= failed_ratio) &
            (failed_ratio < self.acceptable_failed_rate_top))
        self.string_columns_transformation_failed.update(dict.fromkeys(is_of_mistakes[is_of_mistakes].index))
        data.numerical_idx = is_numeric_ids

    def _into_numeric_features_transformation_for_predict(self, data: InputData):
        """ Apply conversion into float string column for every signed column """
        str_col_ids = np.setdiff1d(
            self.categorical_into_float,
            list(self.string_columns_transformation_failed)
        ).astype(int)
        str_cols_df = pd.DataFrame(data.features[:, str_col_ids], columns=str_col_ids)
        data.features[:, str_col_ids] = str_cols_df.apply(pd.to_numeric, errors='coerce').to_numpy()

        # Update information about column types (in-place)
        feature_type_ids = data.supplementary_data.col_type_ids['features']
        feature_type_ids[str_col_ids] = TYPE_TO_ID[float]


def define_column_types(table: Optional[np.ndarray]) -> pd.DataFrame:
    """ Prepare information about types per columns. For each column store unique
    types, which column contains.
    """
    table_of_types = pd.DataFrame(table, copy=True)
    table_of_types = table_of_types.replace({np.nan: None}).applymap(lambda el: TYPE_TO_ID[type(el)])

    # Build dataframe with unique types for each column
    uniques = table_of_types.apply(pd.unique, result_type='reduce').to_frame(_TYPES).T

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
        .reindex(counts_index_mapper.keys(), copy=False)  # Sets all type ids
        .replace(np.nan, 0)
        .rename(index=counts_index_mapper, copy=False)  # Renames all type ids to strs
        .astype(int)
    )

    # Build dataframe with nans indices
    nans_ids = (
        (table_of_types == TYPE_TO_ID[type(None)])
        .apply(np.flatnonzero, result_type='reduce')
        .to_frame(_NAN_IDS).T
    )

    # Combine all dataframes
    return pd.concat([uniques, types_counts, nans_ids])


def _find_mixed_types_columns(columns_info: pd.DataFrame) -> pd.DataFrame:
    """ Search for columns with several types in them """
    has_mixed_types = columns_info.loc[_TYPES].apply(len) > 1
    return columns_info.loc[:, has_mixed_types]


def _select_from_rows_if_any(frame: pd.DataFrame, rows_to_select: List[str]) -> pd.DataFrame:
    cols_have_any = frame.loc[rows_to_select].any()
    return frame.loc[:, cols_have_any]


def apply_type_transformation(table: np.ndarray, col_type_ids: Sequence, log: LoggerAdapter):
    """
    Apply transformation for columns in dataset into desired type. Perform
    transformation on predict stage when column types were already determined
    during fit
    """
    table_df = pd.DataFrame(table, copy=False)
    types_sr = pd.Series(col_type_ids).map({
        **{TYPE_TO_ID[t]: t for t in [int, str]},
        **{TYPE_TO_ID[t]: float for t in [bool, type(None), float]}
    })

    return table_df.apply(_convert_predict_column_into_desired_type, types_sr=types_sr, log=log).to_numpy()


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


def _convert_predict_column_into_desired_type(current_column: pd.Series, types_sr: pd.Series, log: LoggerAdapter):
    current_type = types_sr.loc[current_column.name]
    try:
        converted_column = current_column.astype(current_type)
        if current_type is str:
            has_comma_and_dot = np.isin(['.', ','], current_column).all()
            if has_comma_and_dot:
                # Most likely case: '20,000' must be converted into '20.000'
                warning = f'Column {current_column.name} contains both "." and ",". Standardize it.'
                log.warning(warning)
    except ValueError:
        converted_column = current_column.apply(_process_predict_column_values_one_by_one, current_type=current_type)
    return converted_column


def _generate_list_with_types(columns_types_info: pd.DataFrame,
                              converted_columns: Dict[int, Optional[int]]) -> np.ndarray:
    """ Create list with types for all remained columns

    :param columns_types_info: dictionary with initial column types
    :param converted_columns: dictionary with transformed column types
    """
    updated_col_type_ids = []

    for column_id, column_type_ids in columns_types_info.loc[_TYPES].items():
        if len(column_type_ids) == 1:
            # Column initially contain only one type
            updated_col_type_ids.append(column_type_ids[0])
        elif len(column_type_ids) == 2 and TYPE_TO_ID[type(None)] in column_type_ids:
            # Column with one type and nans
            filtered_types = [x for x in column_type_ids if x != TYPE_TO_ID[type(None)]]
            updated_col_type_ids.append(filtered_types[0])
        else:
            if TYPE_TO_ID[str] in column_type_ids:
                # Mixed-types column with string
                new_col_id = converted_columns[column_id]
                if new_col_id is not None:
                    updated_col_type_ids.append(new_col_id)
            else:
                # Mixed-types with float and integer
                updated_col_type_ids.append(TYPE_TO_ID[float])

    return np.array(updated_col_type_ids)


def _process_predict_column_values_one_by_one(value, current_type: type):
    """ Process column values one by one and try to convert them into desirable type.
    If not successful replace with np.nan """
    new_value = np.nan
    try:
        new_value = current_type(value)
    except ValueError:
        if isinstance(value, str) and ('.' in value or ',' in value):
            value = value.replace(',', '.')
            try:
                # Since "10.6" can not be converted to 10 straightforward using int()
                if current_type is int:
                    new_value = int(float(value))
            except ValueError:
                pass
    return new_value


def prepare_log_message_with_cols_types(col_types_info, features_names):
    message = '\n' + 'Features\n'
    for type_name, type_id in TYPE_TO_ID.items():
        count_types = np.count_nonzero(col_types_info['features'] == type_id)
        features_idx = np.where(col_types_info['features'] == type_id)[0]
        names_or_indexes = features_names[features_idx] if features_names is not None else features_idx
        message += f'\tTYPE {type_name} - count {count_types} - features {names_or_indexes} \n' \

    message += '-' * 10 + '\n'
    message += f'Target: TYPE {_convertable_types[col_types_info["target"][0]]}'

    return message
