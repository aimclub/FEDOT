import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.preprocessing.data_types import TableTypesCorrector
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from test.unit.preprocessing.test_pipeline_preprocessing import data_with_mixed_types_in_each_column


def generate_linear_pipeline():
    """
    Generate linear pipeline where the data changes from node to node.
    The number of columns increases (poly_features), the number of columns
    decreases (rfe_lin_class), the types change while keeping the number of
    columns ('label_encoding')
    """
    encoding_label = PrimaryNode('label_encoding')
    poly_node = SecondaryNode('poly_features', nodes_from=[encoding_label])
    rfe_node = SecondaryNode('rfe_lin_class', nodes_from=[poly_node])
    final_node = SecondaryNode('dt', nodes_from=[rfe_node])
    pipeline = Pipeline(final_node)

    return pipeline


def get_types_incorrect_data():
    """
    Generate InputData with mixed column types.
    Train part - idx [0 ... 15], test part - idx [16, 17]
    Column description by indices:
        * 0) int column with single np.nan value - nans must be filled in
        * 1) int column with nans more than 90% - column must be removed
        * 2) int column with categorical values (number of unique values = 12) -
        categorical indices must be converted into integers, then due to number
        of unique values less than 13 - perform converting column into str type
        * 3) int column the same as 2) column but with additional 13th label in the test part
        * 4) int column (number of unique values = 4) - must be converted into string
        * 5) str column with unique categories 'a', 'b', 'c' and spaces in labels.
        New category 'd' arise in the test part. Categories will be converted into float
        * 6) str binary column - must be converted into integer
        * 7) int binary column and nans - nan cells must be filled in
        * 8) str column with truly int values as strings - must be converted into float column
    """

    task = Task(TaskTypesEnum.classification)
    features = np.array([[0, np.nan, 1, 1, 1, '  a    ', 'true', 1, '0'],
                         [np.nan, 5, 2, 2, 0, '   b   ', np.nan, 0, '1'],
                         [2, np.nan, 3, 3, np.nan, 'c', 'false', 1, '2'],
                         [3, np.nan, 4, 4, 3.0, '  a  ', 'true', 0, '3'],
                         [4, np.nan, 5, 5.0, 0, '   b ', np.nan, 0, '4'],
                         [5, np.nan, 6, 6, 0, '   c  ', 'false', 0, '5'],
                         [6, np.inf, 7, 7, 0, '    a  ', 'true', 1, '6'],
                         [7, np.inf, 8, 8, 1.0, ' b   ', np.nan, 0, '7'],
                         [np.inf, np.inf, '9', '9', 2, np.nan, 'true', 1, '8'],
                         [9, np.inf, '10', '10', 2, ' c  ', 'false', 0, '9'],
                         [10, np.nan, 11.0, 11.0, 0, 'c ', 'false', 0, '10'],
                         [11, np.nan, 12, 12, 2.0, np.nan, 'false', 1, '11'],
                         [12, np.nan, 1, 1.0, 1.0, ' b  ', 'false', 0, '12'],
                         [13, np.nan, 2, 2, 1, ' c  ', 'true', np.nan, '13'],
                         [14, np.nan, 3, 3, 2.0, 'a', 'false', np.nan, '14'],
                         [15, np.nan, 4, 4, 1, 'a  ', 'false', np.nan, '15'],
                         [16, 2, 5, 12, 0, '   d       ', 'true', 1, '16'],
                         [17, 3, 6, 13, 0, '  d      ', 'false', 0, '17']],
                        dtype=object)
    target = np.array(['no', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no',
                       'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no'])
    input_data = InputData(idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                           features=features, target=target, task=task, data_type=DataTypesEnum.table)

    return input_data


def test_column_types_converting_correctly():
    """ Generate dataset with mixed data types per columns and then apply correction procedure.
    After converting into new types, field "column_types" for target and features is checked
    """
    data = data_with_mixed_types_in_each_column(multi_output=True)
    # Remove column full of nans (column with id 0)
    data.features = data.features[:, 1:]

    types_corr = TableTypesCorrector()
    data = types_corr.convert_data_for_fit(data)

    features_types = data.supplementary_data.column_types['features']
    target_types = data.supplementary_data.column_types['target']

    assert len(features_types) == len(target_types) == 2
    assert features_types[0] == "<class 'str'>"
    assert features_types[1] == "<class 'float'>"
    assert target_types[0] == target_types[0] == "<class 'str'>"


def test_column_types_process_correctly():
    """ Generate table with different types of data per columns.
    After transferring via pipeline, they must be processed correctly
    (added and removed)
    """

    data = data_with_mixed_types_in_each_column()
    train_data, test_data = train_test_data_setup(data, split_ratio=0.9)

    # Remove target from test sample
    test_data.target = None
    pipeline = generate_linear_pipeline()

    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data)

    features_columns = predicted.supplementary_data.column_types['features']
    assert len(features_columns) == predicted.predict.shape[1]
    # All output values are float
    assert all('float' in str(feature_type) for feature_type in features_columns)
