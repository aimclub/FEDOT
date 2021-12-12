import numpy as np

from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import Task, TaskTypesEnum

NAME_CLASS_STR = "<class 'str'>"
NAME_CLASS_INT = "<class 'int'>"
NAME_CLASS_FLOAT = "<class 'float'>"
NAME_CLASS_NONE = "<class 'NoneType'>"


class TableTypesCorrector:
    """
    Class for checking types in input data. Also perform conversion for columns with types conflicts
    """

    def __init__(self, log: Log = None):
        self.features_columns_info = {}
        self.target_columns_info = {}

        # Dictionary with information about converted during fitting columns
        self.features_converted_columns = {}
        self.target_converted_columns = {}

        # Columns to delete due to types conflicts
        self.columns_to_del = []
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def convert_data_for_fit(self, data: 'InputData'):
        """ If column contain several data types - perform correction procedure """
        # Determine types for each column in features and target if it is necessary
        if not self.features_columns_info:
            self.features_columns_info = define_column_types(data.features)
        if not self.target_columns_info:
            self.target_columns_info = define_column_types(data.target)

        # Correct types in features table
        table = self.features_types_converting(features=data.features)
        # Remain only correct columns
        data.features = self.remove_incorrect_features(table, self.features_converted_columns)

        # And in target(s)
        data.target = self.target_types_converting(target=data.target, task=data.task)
        data.supplementary_data.column_types = self.store_column_types_info(predictors=data.features,
                                                                            target=data.target,
                                                                            task=data.task)
        return data

    def convert_data_for_predict(self, data: 'InputData'):
        """ Prepare data for predict stage. Include only column types transformation """
        # Ordering is important because after removing incorrect features - indices are obsolete
        table = apply_type_transformation(data.features, self.features_converted_columns)
        data.features = self.remove_incorrect_features(table, self.features_converted_columns)
        data.target = apply_type_transformation(data.target, self.target_converted_columns)
        data.supplementary_data.column_types = self.store_column_types_info(predictors=data.features,
                                                                            target=data.target,
                                                                            task=data.task)
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
                column_info = self.features_columns_info[mixed_column_id]

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

    def store_column_types_info(self, predictors: np.array, target: np.array = None,
                                task: Task = None) -> dict:
        """ Prepare information about columns in a form of dictionary
        Dictionary has two keys: 'target' and 'features'
        """
        if not self.features_columns_info:
            # Information about column types is empty - there is a need to launch algorithm to collect info
            self.features_columns_info = define_column_types(predictors)
            predictors = self.features_types_converting(features=predictors)
        if not self.target_columns_info:
            self.target_columns_info = define_column_types(target)
            target = self.target_types_converting(target=target, task=task)

        features_types = _generate_list_with_types(self.features_columns_info, self.features_converted_columns)
        self._check_columns_vs_types_number(predictors, features_types)

        if target is None:
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
            self.log.warn(f'Columns number and types numbers do not match.')

    def _convert_feature_into_one_type(self, mixed_column: np.array, column_info: dict, mixed_column_id: int):
        """ Determine new type for current feature column based on the string ratio. And then convert column into it.

        :param mixed_column: one-dimensional array with several data types
        :param column_info: dictionary with information about types in the column
        :param mixed_column_id: index of column in dataset
        """
        string_objects_number = column_info['str_number']
        all_elements_number = string_objects_number + column_info['int_number'] + column_info['float_number']
        string_ratio = string_objects_number / all_elements_number

        if string_ratio > 0.5:
            suggested_type = str
        else:
            suggested_type = _define_new_int_float_type(column_info)

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
                                      task: Task):
        """ Convert target columns into one type based on column proportions of object and task """
        if task.task_type is TaskTypesEnum.classification:
            # For classification labels are string if at least one element is a string
            suggested_type = str
        else:
            suggested_type = _define_new_int_float_type(column_info)

        try:
            mixed_column = mixed_column.astype(suggested_type)
            return mixed_column, str(suggested_type)
        except ValueError:
            # Cannot convert string objects into int or float (for example 'a' into int)
            prefix = f'Target column with index {mixed_column_id} contains ' \
                     f'following data types: {column_info["types"]}.'
            error_message = f'{prefix} String cannot be converted into {suggested_type}.'
            self.log.error(error_message)
            raise ValueError(error_message)


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


def apply_type_transformation(table: np.array, converted_columns: dict):
    """ Apply transformation for columns in dataset with several data types """
    type_by_name = {"<class 'int'>": int, "<class 'str'>": str, "<class 'float'>": float}

    if table is None:
        # Occurs if for predict stage there is no target info
        return None

    if len(converted_columns) == 0:
        # There are no columns for converting
        return table

    for column_id, type_name in converted_columns.items():
        if type_name != 'removed':
            current_type = type_by_name[type_name]
            table[:, column_id] = table[:, column_id].astype(current_type)
    return table


def _define_new_int_float_type(column_info):
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
        elif len(column_types) == 1 and 'NoneType' in column_types:
            # Column with one type and nans
            filtered_types = list(filter(lambda x: x != NAME_CLASS_NONE, column_types))
            updated_column_types.append(filtered_types[0])
        else:
            if 'str' in column_types:
                # Mixed-types column with string
                new_column_type = converted_columns[column_id]
                if new_column_type != 'removed':
                    updated_column_types.append(new_column_type)
            else:
                # Mixed-types with float and integer
                updated_column_types.append(NAME_CLASS_FLOAT)

    return updated_column_types
