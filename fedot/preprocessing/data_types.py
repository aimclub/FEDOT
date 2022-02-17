import numpy as np
import pandas as pd

from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import Task, TaskTypesEnum

NAME_CLASS_STR = "<class 'str'>"
NAME_CLASS_INT = "<class 'int'>"
NAME_CLASS_FLOAT = "<class 'float'>"
NAME_CLASS_NONE = "<class 'NoneType'>"
FEDOT_STR_NAN = 'fedot_nan'
# If unique values in the feature column is less than 13 - convert column into string type
CATEGORICAL_UNIQUE_TH = 13
MAX_CATEGORIES_TH = 30


class TableTypesCorrector:
    """
    Class for checking types in input data. Also perform conversion for columns with types conflicts
    """

    def __init__(self, log: Log = None):
        # Maximum allowed unique categories in categorical table (if more - transform it into float)
        self.categorical_max_classes_th = MAX_CATEGORIES_TH
        # Threshold to convert numerical into categorical column
        self.numerical_min_uniques = CATEGORICAL_UNIQUE_TH

        self.features_columns_info = {}
        self.target_columns_info = {}

        # Dictionary with information about converted during fitting columns
        self.features_converted_columns = {}
        self.target_converted_columns = {}

        # Columns to delete due to types conflicts
        self.columns_to_del = []
        # Column ids for transformation due to number of unique values
        self.numerical_into_str = []
        self.categorical_into_float = []

        # Is target column contains non-numerical cells during conversion
        self.target_converting_has_errors = False

        # Lists with column types for converting
        self.features_types = None
        self.target_types = None
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def convert_data_for_fit(self, data: 'InputData'):
        """ If column contain several data types - perform correction procedure """
        # Convert features to have an ability to insert str into float table or vice versa
        data.features = data.features.astype(object)

        # Determine types for each column in features and target if it is necessary
        self.features_columns_info = define_column_types(data.features)
        self.target_columns_info = define_column_types(data.target)

        # Correct types in features table
        table = self.features_types_converting(features=data.features)
        # Remain only correct columns
        data.features = self.remove_incorrect_features(table, self.features_converted_columns)

        # And in target(s)
        data.target = self.target_types_converting(target=data.target, task=data.task)
        data.supplementary_data.column_types = self.prepare_column_types_info(predictors=data.features,
                                                                              target=data.target,
                                                                              task=data.task)

        # Launch conversion float and integer features into categorical
        self._into_categorical_features_transformation_for_fit(data)
        self._into_numeric_features_transformation_for_fit(data)

        # Save info about features and target types
        self.features_types = data.supplementary_data.column_types['features']
        self.target_types = data.supplementary_data.column_types['target']
        return data

    def convert_data_for_predict(self, data: 'InputData'):
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
        self._into_categorical_features_transformation_for_predict(data)
        self._into_numeric_features_transformation_for_predict(data)
        return data

    def remove_incorrect_features(self, table: np.array, converted_columns: dict):
        """
        Remove from the table columns with conflicts with types were not resolved

        :param table: tabular dataset based on which new dataset will be generated
        :param converted_columns: dictionary with actions with table
        """
        if len(converted_columns) == 0:
            return table

        self.columns_to_del = [column_id for column_id, new_type_name in converted_columns.items() if
                               new_type_name == 'removed']
        if not self.columns_to_del:
            # There are no columns to delete
            return table

        # Remove all "bad" columns
        table = np.delete(table, self.columns_to_del, 1)
        return table

    def features_types_converting(self, features: np.array) -> np.array:
        """ Convert all elements in the data in every feature column into one type

        :param features: tabular features array
        """
        features_with_mixed_types = find_mixed_types_columns(self.features_columns_info)

        if not features_with_mixed_types:
            return features

        if features_with_mixed_types:
            # There are mixed-types columns in features table - convert them
            for mixed_column_id in features_with_mixed_types:
                column_info = self.features_columns_info[mixed_column_id]

                if column_info.get('str_number') > 0:
                    # There are string elements in the array
                    mixed_column = features[:, mixed_column_id]
                    updated_column, new_type_name = self._convert_feature_into_one_type(mixed_column, column_info,
                                                                                        mixed_column_id)
                    # Store information about converted columns
                    self.features_converted_columns.update({mixed_column_id: new_type_name})

                    if updated_column is not None:
                        features[:, mixed_column_id] = updated_column

        return features

    def target_types_converting(self, target: np.array, task: Task) -> np.array:
        """ Convert all elements in every target column into one type

        :param target: tabular target array
        :param task: task to solve
        """
        target_with_mixed_types = find_mixed_types_columns(self.target_columns_info)

        if not target_with_mixed_types:
            return target

        if target_with_mixed_types:
            # There are mixed-types columns in features table - convert them
            for mixed_column_id in target_with_mixed_types:
                column_info = self.target_columns_info[mixed_column_id]

                if column_info.get('str_number') > 0:
                    # There are string elements in the array
                    mixed_column = target[:, mixed_column_id]
                    updated_column, new_type_name = self._convert_target_into_one_type(mixed_column, column_info,
                                                                                       mixed_column_id, task)
                    # Store information about converted columns
                    self.target_converted_columns.update({mixed_column_id: new_type_name})

                    if updated_column is not None:
                        target[:, mixed_column_id] = updated_column

        return target

    def prepare_column_types_info(self, predictors: np.array, target: np.array = None,
                                  task: Task = None) -> dict:
        """ Prepare information about columns in a form of dictionary
        Dictionary has two keys: 'target' and 'features'
        """
        if not self.features_columns_info:
            # Information about column types is empty - there is a need to launch algorithm to collect info
            self.features_columns_info = define_column_types(predictors)
            predictors = self.features_types_converting(features=predictors)
        if not self.target_columns_info and task.task_type != TaskTypesEnum.ts_forecasting:
            self.target_columns_info = define_column_types(target)
            target = self.target_types_converting(target=target, task=task)

        features_types = _generate_list_with_types(self.features_columns_info, self.features_converted_columns)
        self._check_columns_vs_types_number(predictors, features_types)

        if target is None or task.task_type == TaskTypesEnum.ts_forecasting:
            return {'features': features_types}
        else:
            target_types = _generate_list_with_types(self.target_columns_info, self.target_converted_columns)
            self._check_columns_vs_types_number(target, target_types)
            return {'features': features_types, 'target': target_types}

    def _check_columns_vs_types_number(self, table: np.array, column_types: list):
        # Check if columns number correct
        n_rows, n_cols = table.shape
        if n_cols != len(column_types):
            # There is an incorrect types calculation
            self.log.warn('Columns number and types numbers do not match.')

    def _convert_feature_into_one_type(self, mixed_column: np.array, column_info: dict, mixed_column_id: int):
        """ Determine new type for current feature column based on the string ratio. And then convert column into it.

        :param mixed_column: one-dimensional array with several data types
        :param column_info: dictionary with information about types in the column
        :param mixed_column_id: index of column in dataset
        """
        if len(column_info['types']) == 2 and NAME_CLASS_NONE in column_info['types']:
            # Column contain only one data type and nans
            filtered_types = list(filter(lambda x: x != NAME_CLASS_NONE, column_info['types']))
            return mixed_column, filtered_types[0]

        string_objects_number = column_info['str_number']
        all_elements_number = string_objects_number + column_info['int_number'] + column_info['float_number']
        string_ratio = string_objects_number / all_elements_number

        if string_ratio > 0.5:
            suggested_type = str
        else:
            suggested_type = _obtain_new_column_type(column_info)

        try:
            mixed_column = mixed_column.astype(suggested_type)
            # If there were nans in the column - paste nan
            if column_info['nan_number'] > 0:
                mixed_column = mixed_column.astype(object)
                mixed_column[column_info['nan_ids']] = np.nan
                del column_info['nan_ids']
            return mixed_column, str(suggested_type)
        except ValueError:
            # Cannot convert string objects into int or float (for example 'a' into int)
            prefix = f'Feature column with index {mixed_column_id} contains ' \
                     f'following data types: {column_info["types"]}.'
            self.log.warn(f'{prefix} String cannot be converted into {suggested_type}. Drop column.')
            return None, 'removed'

    def _convert_target_into_one_type(self, mixed_column: np.array, column_info: dict, mixed_column_id: int,
                                      task: Task) -> [np.array, str]:
        """ Convert target columns into one type based on column proportions of object and task """
        if task.task_type is TaskTypesEnum.classification:
            # For classification labels are string if at least one element is a string
            suggested_type = str
        else:
            suggested_type = _obtain_new_column_type(column_info)

        try:
            mixed_column = mixed_column.astype(suggested_type)
            return mixed_column, str(suggested_type)
        except ValueError:
            # Cannot convert string objects into int or float (for example 'a' into int)
            target_column = pd.Series(mixed_column)
            converted_column = pd.to_numeric(target_column, errors='coerce')

            prefix = f'Target column with index {mixed_column_id} contains ' \
                     f'following data types: {column_info["types"]}.'
            log_message = f'{prefix} String cannot be converted into {suggested_type}. Ignore non converted values.'
            self.log.debug(log_message)
            self.target_converting_has_errors = True
            return converted_column.values, str(suggested_type)

    def _into_categorical_features_transformation_for_fit(self, data: 'InputData'):
        """
        Perform automated categorical features determination. If feature column
        contains int or float values with few unique values (less than 13)
        """
        n_rows, n_cols = data.features.shape
        for column_id in range(n_cols):
            # For every int/float column perform check
            column_type = data.supplementary_data.column_types['features'][column_id]
            if 'int' in column_type or 'float' in column_type:
                numerical_column = pd.Series(data.features[:, column_id])

                # Calculate number of unique values except nans
                unique_numbers = len(numerical_column.dropna().unique())

                if 2 < unique_numbers < self.numerical_min_uniques:
                    # Column need to be transformed into categorical (string) one
                    self.numerical_into_str.append(column_id)

                    # Convert into string
                    converted_array = convert_num_column_into_string_array(numerical_column)

                    # Store converted column into features table
                    data.features[:, column_id] = converted_array

                    # Update information about column types (in-place)
                    features_types = data.supplementary_data.column_types['features']
                    features_types[column_id] = NAME_CLASS_STR

    def _into_categorical_features_transformation_for_predict(self, data: 'InputData'):
        """ Apply conversion into categorical string column for every signed column """
        if not self.numerical_into_str:
            # There is no transformation for current table
            return data

        n_rows, n_cols = data.features.shape
        for column_id in range(n_cols):
            if column_id in self.numerical_into_str:
                numerical_column = pd.Series(data.features[:, column_id])
                # Column must be converted into categorical
                converted_array = convert_num_column_into_string_array(numerical_column)
                data.features[:, column_id] = converted_array

                # Update information about column types (in-place)
                features_types = data.supplementary_data.column_types['features']
                features_types[column_id] = NAME_CLASS_STR

    def _into_numeric_features_transformation_for_fit(self, data: 'InputData'):
        """
        Automatically determine categorical features which should be converted into float
        """
        n_rows, n_cols = data.features.shape
        for column_id in range(n_cols):
            # For every string column perform converting if necessary
            column_type = data.supplementary_data.column_types['features'][column_id]
            if 'str' in column_type:
                string_column = pd.Series(data.features[:, column_id])
                unique_numbers = len(string_column.dropna().unique())

                if unique_numbers > self.categorical_max_classes_th:
                    # Number of nans in the column
                    nans_number = string_column.isna().sum()

                    # Column probably not an "actually categorical" but a column with an incorrectly defined type
                    converted_column = pd.to_numeric(string_column, errors='coerce')
                    # Calculate applied nans
                    result_nans_number = converted_column.isna().sum()
                    failed_objects_number = result_nans_number - nans_number
                    non_nan_all_objects_number = n_rows - nans_number
                    failed_ratio = failed_objects_number / non_nan_all_objects_number

                    if failed_ratio < 0.5:
                        # The majority of objects can be converted into numerical
                        data.features[:, column_id] = converted_column.values

                        # Update information about column types (in-place)
                        self.categorical_into_float.append(column_id)
                        features_types = data.supplementary_data.column_types['features']
                        features_types[column_id] = NAME_CLASS_FLOAT

    def _into_numeric_features_transformation_for_predict(self, data: 'InputData'):
        """ Apply conversion into float string column for every signed column """
        if not self.categorical_into_float:
            # There is no transformation for current table
            return data

        n_rows, n_cols = data.features.shape
        for column_id in range(n_cols):
            if column_id in self.categorical_into_float:
                string_column = pd.Series(data.features[:, column_id])

                # Column must be converted into float from categorical
                converted_column = pd.to_numeric(string_column, errors='coerce')
                data.features[:, column_id] = converted_column.values

                # Update information about column types (in-place)
                features_types = data.supplementary_data.column_types['features']
                features_types[column_id] = NAME_CLASS_FLOAT


def define_column_types(table: np.array):
    """ Prepare information about types per columns. For each column store unique
    types, which column contains. If column with mixed type contain str object
    additional field 'str_ids' with indices of string objects is prepared
    """
    # TODO: current processing is relatively computationally expensive - probably refactor needed

    def type_ignoring_nans(item):
        """ Return type of element in the array. If item is np.nan - return NoneType """
        current_type = type(item)
        if current_type is float and np.isnan(item):
            # Check is current element is nan or not (np.nan is a float type)
            return type(None)
        return current_type

    if table is None:
        return {}

    n_rows, n_columns = table.shape
    columns_info = {}
    for column_id in range(n_columns):
        current_column = table[:, column_id]

        # Check every element in numpy array - it can take a long time!
        column_types = list(map(type_ignoring_nans, current_column))

        # Store only unique values
        set_column_types = set(column_types)
        # Convert types into string names
        column_types_names = list(map(lambda x: str(x), set_column_types))

        if len(column_types_names) > 1:
            # There are several types in one column
            types_names = np.array(column_types, dtype=str)
            # Calculate number of string objects in the dataset
            str_number = len(np.argwhere(types_names == NAME_CLASS_STR))
            int_number = len(np.argwhere(types_names == NAME_CLASS_INT))
            float_number = len(np.argwhere(types_names == NAME_CLASS_FLOAT))

            # Store information about nans in the target
            nan_ids = np.ravel(np.argwhere(types_names == NAME_CLASS_NONE))
            nan_number = len(nan_ids)
            columns_info.update({column_id: {'types': column_types_names,
                                             'str_number': str_number,
                                             'int_number': int_number,
                                             'float_number': float_number,
                                             'nan_number': nan_number,
                                             'nan_ids': nan_ids}})
        else:
            # There is only one type, or several types such as int and float
            columns_info.update({column_id: {'types': column_types_names}})
    return columns_info


def find_mixed_types_columns(columns_info: dict):
    """ Search for columns with several types in them """
    columns_with_mixed_types = []
    for column_id, information in columns_info.items():
        column_types = information['types']
        if len(column_types) > 1:
            columns_with_mixed_types.append(column_id)

    return columns_with_mixed_types


def apply_type_transformation(table: np.array, column_types: list, log: Log):
    """
    Apply transformation for columns in dataset into desired type. Perform
    transformation on predict stage when column types were already determined
    during fit
    """

    def type_by_name(current_type_name: str):
        """ Return type by it's name """
        if 'int' in current_type_name:
            return int
        elif 'str' in current_type_name:
            return str
        else:
            return float

    if table is None:
        # Occurs if for predict stage there is no target info
        return None

    n_rows, n_cols = table.shape
    for column_id in range(n_cols):
        current_column = table[:, column_id]
        current_type = type_by_name(column_types[column_id])
        try:
            table[:, column_id] = current_column.astype(current_type)
        except ValueError as ex:
            log.debug(f'Cannot convert column with id {column_id} into type {current_type} due to {ex}')

            message = str(ex)
            if 'NaN' not in message:
                # Try to convert column from string into float
                unseen_label = message.split("\'")[1]
                if ',' in unseen_label:
                    # Most likely case: '20,000' must be converted into '20.000'
                    err = f'Column {column_id} contains both "." and ",". Standardize it.'
                    raise ValueError(err)
                else:
                    # Most likely case: ['a', '1.5'] -> [np.nan, 1.5]
                    label_ids = np.ravel(np.argwhere(current_column == unseen_label))
                    current_column[label_ids] = np.nan
                    table[:, column_id] = current_column.astype(float)

    return table


def convert_num_column_into_string_array(numerical_column: pd.Series) -> np.array:
    """ Convert pandas column into numpy one-dimensional array """
    # Convert into string
    converted_column = numerical_column.astype(str)
    converted_array = converted_column.values

    # If there are nans - insert them
    nan_ids = np.ravel(np.argwhere(converted_array == 'nan'))
    if len(nan_ids) > 0:
        converted_array = converted_array.astype(object)
        converted_array[nan_ids] = np.nan

    return converted_array


def _obtain_new_column_type(column_info):
    """ Suggest in or float type based on the presence of nan and float values """
    if column_info['float_number'] > 0 or column_info['nan_number'] > 0:
        # Even if one of types are float - all elements should be converted into float
        return float
    else:
        # It is available to convert numerical into integer type
        return int


def _generate_list_with_types(columns_types_info: dict, converted_columns: dict) -> list:
    """ Create list with types for all remained columns

    :param columns_types_info: dictionary with initial column types
    :param converted_columns: dictionary with transformed column types
    """
    updated_column_types = []
    for column_id, column_info in columns_types_info.items():
        column_types = column_info['types']

        if len(column_types) == 1:
            # Column initially contain only one type
            updated_column_types.append(column_types[0])
        elif len(column_types) == 2 and NAME_CLASS_NONE in column_types:
            # Column with one type and nans
            filtered_types = list(filter(lambda x: x != NAME_CLASS_NONE, column_types))
            updated_column_types.append(filtered_types[0])
        else:
            if any('str' in column_type_name for column_type_name in column_types):
                # Mixed-types column with string
                new_column_type = converted_columns[column_id]
                if new_column_type != 'removed':
                    updated_column_types.append(new_column_type)
            else:
                # Mixed-types with float and integer
                updated_column_types.append(NAME_CLASS_FLOAT)

    return updated_column_types
