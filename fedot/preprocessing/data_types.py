import numpy as np
from fedot.core.data.data import InputData


class TypesCorrector:
    """
    Class for checking types in input data.
    Correction also can be performed for such datasets.
    """

    def __init__(self):
        self.features_columns_info = {}
        self.target_columns_info = {}

        # Dictionary with information about converted during fitting columns
        self.converted_columns = {}

    def check_data_types(self, data: InputData):
        """
        Find every column in features and target with mixed types.
        """
        self.features_columns_info = define_column_types(data.features)
        self.target_columns_info = define_column_types(data.target)

    def convert_data_for_fit(self, data: InputData):
        """ If column contain several data types - perform correction procedure """
        # Correct types in features table
        data.features = self._reduce_to_one_type(columns_info=self.features_columns_info, table=data.features)

        # And in target(s)
        data.target = self._reduce_to_one_type(columns_info=self.target_columns_info, table=data.target)
        return data

    def _reduce_to_one_type(self, columns_info: dict, table: np.array):
        """ Convert all elements in the data in every column into one type

        :param columns_info: dictionary with information about column types
        :param table: tabular array
        """
        columns_with_mixed_types = find_mixed_types_columns(columns_info)
        if columns_with_mixed_types:
            # There are mixed-types columns in features table - convert them
            for mixed_column_id in columns_with_mixed_types:
                column_info = columns_info[mixed_column_id]
                str_ids = column_info.get('str_ids')

                if str_ids is not None:
                    # There are string elements in the array
                    mixed_column = table[:, mixed_column_id]

                    column_types = column_info['types']
                    new_column_type = define_new_type(mixed_column, str_ids, column_types)

                    # Store information about converted columns
                    self.converted_columns.update({mixed_column_id: new_column_type})

        return table


def define_column_types(table: np.array):
    """ Prepare information about types per columns. For each column store unique
    types, which column contains. If column with mixed type contain str object
    additional field 'str_ids' with indices of string objects is prepared
    """
    # TODO: current processing is relatively computationally expensive - probably refactor needed
    n_rows, n_columns = table.shape
    columns_info = {}
    for column_id in range(n_columns):
        current_column = table[:, column_id]

        # Check every element in numpy array
        column_types = list(map(lambda element: type(element), current_column))

        # Store only unique values
        set_column_types = set(column_types)

        if len(set_column_types) > 1 and str in set_column_types:
            # There are several types in one column and one of it is string
            types_names = np.array(column_types, dtype=str)
            str_ids = np.ravel(np.argwhere(types_names == "<class 'str'>"))
            columns_info.update({column_id: {'types': set_column_types,
                                             'str_ids': str_ids}})
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


def define_new_type(mixed_column: np.array, str_ids: np.array, column_types: set):
    """ Determine new type for current column based on the string ratio

    :param mixed_column: one-dimensional array with several data types
    :param str_ids: array with indices of string objects
    :param column_types: set with types in the column
    """
    # Calculate string objects ratio
    all_elements_number = len(mixed_column)
    string_objects_number = len(str_ids)
    # TODO np.nan is 'float' so here may be some imbalance
    string_ratio = string_objects_number / all_elements_number

    if string_ratio > 0.5:
        new_type = str
    else:
        if float in column_types:
            # Even if one of types are float - all elements should be converted into float
            new_type = float
        else:
            # It is available to convert numerical into integer type
            new_type = int

    try:
        mixed_column = mixed_column.astype(new_type)
        return mixed_column
    except Exception:
        return mixed_column
