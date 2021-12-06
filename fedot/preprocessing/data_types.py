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

    def check_data_types(self, data: InputData):
        """
        Find every column in features and target with mixed types.
        """
        # TODO: current processing is relatively computationally expensive - probably refactor needed
        self.features_columns_info = define_column_types(data.features)
        self.target_columns_info = define_column_types(data.target)

        a = 0

    def convert_data(self, data: InputData):
        pass

    def has_column_mixed_types(self):
        pass

    def find_columns_with_mixed_types(self):
        pass


def define_column_types(table: np.array):
    """ Prepare information about types per columns """
    n_rows, n_columns = table.shape
    columns_info = {}
    for column_id in range(n_columns):
        current_column = table[:, column_id]

        # Check every element in numpy array
        column_types = set(map(lambda element: type(element), current_column))
        columns_info.update({column_id: list(column_types)})

    return columns_info
