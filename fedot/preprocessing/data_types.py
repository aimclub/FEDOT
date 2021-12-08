import numpy as np

from fedot.core.log import Log, default_log
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum


class TypesCorrector:
    """
    Class for checking types in input data.
    Correction also can be performed for such datasets.
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

    def check_data_types(self, data: InputData):
        """
        Find every column in features and target with mixed types.
        """
        self.features_columns_info = define_column_types(data.features)
        self.target_columns_info = define_column_types(data.target)

    def convert_data_for_fit(self, data: InputData):
        """ If column contain several data types - perform correction procedure """
        # Correct types in features table
        table = self.features_types_filtering(data=data)
        # Remain only correct columns
        data.features = self.remove_incorrect_features(table, self.features_converted_columns)

        # And in target(s)
        data.target = self.target_types_filtering(data=data)
        return data

    def convert_data_for_predict(self, data: InputData):
        """ Prepare data for predict stage. Include only column types transformation """
        # Ordering is important because after removing incorrect features - indices are obsolete
        table = apply_type_transformation(data.features, self.features_converted_columns)
        data.features = self.remove_incorrect_features(table, self.features_converted_columns)
        data.target = apply_type_transformation(data.target, self.target_converted_columns)
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

    def features_types_filtering(self, data: InputData):
        """ Convert all elements in the data in every feature column into one type

        :param data: Input data with tabular features array
        """
        table = data.features
        features_with_mixed_types = find_mixed_types_columns(self.features_columns_info)
        if features_with_mixed_types:
            # There are mixed-types columns in features table - convert them
            for mixed_column_id in features_with_mixed_types:
                column_info = self.features_columns_info[mixed_column_id]

                if column_info.get('str_number') > 0:
                    # There are string elements in the array
                    mixed_column = table[:, mixed_column_id]
                    updated_column, new_type_name = self._convert_feature_into_one_type(mixed_column, column_info,
                                                                                        mixed_column_id)
                    # Store information about converted columns
                    self.features_converted_columns.update({mixed_column_id: new_type_name})

                    if updated_column is not None:
                        table[:, mixed_column_id] = updated_column

        return table

    def target_types_filtering(self, data: InputData):
        """ Convert all elements in every target column into one type

        :param data: Input data with tabular target array
        """
        task = data.task
        target_table = data.target
        target_with_mixed_types = find_mixed_types_columns(self.target_columns_info)
        if target_with_mixed_types:
            # There are mixed-types columns in features table - convert them
            for mixed_column_id in target_with_mixed_types:
                column_info = self.features_columns_info[mixed_column_id]

                if column_info.get('str_number') > 0:
                    # There are string elements in the array
                    mixed_column = target_table[:, mixed_column_id]
                    updated_column, new_type_name = self._convert_target_into_one_type(mixed_column, column_info,
                                                                                       mixed_column_id, task)
                    # Store information about converted columns
                    self.target_columns_info.update({mixed_column_id: new_type_name})

                    if updated_column is not None:
                        target_table[:, mixed_column_id] = updated_column

        return target_table

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

    n_rows, n_columns = table.shape
    columns_info = {}
    for column_id in range(n_columns):
        current_column = table[:, column_id]

        # Check every element in numpy array - it can take a long time!
        column_types = list(map(type_ignoring_nans, current_column))

        # Store only unique values
        set_column_types = set(column_types)

        if len(set_column_types) > 1:
            # There are several types in one column
            types_names = np.array(column_types, dtype=str)
            # Calculate number of string objects in the dataset
            str_number = len(np.argwhere(types_names == "<class 'str'>"))
            int_number = len(np.argwhere(types_names == "<class 'int'>"))
            float_number = len(np.argwhere(types_names == "<class 'float'>"))
            nan_number = len(np.argwhere(types_names == "<class 'NoneType'>"))
            columns_info.update({column_id: {'types': set_column_types,
                                             'str_number': str_number,
                                             'int_number': int_number,
                                             'float_number': float_number,
                                             'nan_number': nan_number}})
        else:
            # There is only one type, or several types such as int and float
            columns_info.update({column_id: {'types': set_column_types}})
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

    if len(converted_columns) == 0:
        # There are np columns for converting
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
