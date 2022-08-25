import os

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.preprocessing.data_types import TableTypesCorrector
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME
from test.unit.api.test_api_cli_params import project_root_path
from test.unit.preprocessing.test_pipeline_preprocessing import data_with_mixed_types_in_each_column, \
    correct_preprocessing_params


def get_data_with_string_columns():
    file_path = os.path.join(project_root_path, 'test/data/data_with_mixed_column.csv')
    df = pd.read_csv(file_path)

    task = Task(TaskTypesEnum.classification)
    input_data = InputData(idx=np.arange(len(df)),
                           features=np.array(df[['mixed_column', 'numerical_column']]),
                           target=np.array(df['target']).reshape(-1, 1),
                           task=task,
                           data_type=DataTypesEnum.table)

    return input_data


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


def data_with_complicated_types():
    """
    Generate InputData with mixed column types.
    Train part - idx [0 ... 15], test part - idx [16, 17]
    Column description by indices:
        0) int column with single np.nan value - nans must be filled in
        1) int column with nans more than 90% - column must be removed
        column must be removed due to the fact that inf will be replaced with nans
        2) int-float column with categorical values (number of unique values = 12) -
        categorical indices must be converted into integers, then due to number
        of unique values less than 13 - perform converting column into str type
        3) int column the same as 2) column but with additional 13th label in the test part
        4) int column (number of unique values = 4) - must be converted into string
        5) str-int column with words and numerical cells - must be removed because can not
        be converted into integers
        6) str column with unique categories 'a', 'b', 'c' and spaces in labels.
        New category 'd' arise in the test part. Categories will be converted into float
        7) str binary column - must be converted into integer
        8) int binary column and nans - nan cells must be filled in
        9) str column with truly int values as strings - must be converted into float column
        10) str column with truly categorical values - must stay remained
    """

    task = Task(TaskTypesEnum.classification)
    features = np.array([[0, np.nan, 1, 1, 1, 'monday', 'a ', 'true', 1, '0', 'a'],
                         [np.nan, 5, 2, 2, 0, 'tuesday', 'b', np.nan, 0, '1', np.inf],
                         [2, np.nan, 3, 3, np.nan, 3, 'c', 'false', 1, '?', 'c'],
                         [3, np.nan, 4, 4, 3.0, 4, '  a  ', 'true', 0, '2', 'd'],
                         [4, np.nan, 5, 5.0, 0, 5, '   b ', np.nan, 0, '3', 'e'],
                         [5, np.nan, 6, 6, 0, 6, '   c  ', 'false', 0, '4', 'f'],
                         [6, np.inf, 7, 7, 0, 7, '    a  ', 'true', 1, '5', 'g'],
                         [7, np.inf, 8, 8, 1.0, 1, ' b   ', np.nan, 0, '6', 'h'],
                         [np.inf, np.inf, '9', '9', 2, 2, np.nan, 'true', 1, '7', 'i'],
                         [9, np.inf, '10', '10', 2, 3, ' c  ', 'false', 0, '8', 'j'],
                         [10, np.nan, 11.0, 11.0, 0, 4, 'c ', 'false', 0, '9', 'k'],
                         [11, np.nan, 12, 12, 2.0, 5, np.nan, 'false', 1, '10', 'l'],
                         [12, np.nan, 1, 1.0, 1.0, 6, ' b  ', 'false', 0, '11', 'm'],
                         [13, np.nan, 2, 2, 1, 7, ' c  ', 'true', np.nan, '12', 'n'],
                         [14, np.nan, 3, 3, 2.0, 1, 'a', 'false', np.nan, 'error', 'o'],
                         [15, np.nan, 4, 4, 1, 2, 'a  ', 'false', np.nan, '13', 'p'],
                         [16, 2, 5, 12, 0, 3, '   d       ', 'true', 1, '16', 'r'],
                         [17, 3, 6, 13, 0, 4, '  d      ', 'false', 0, '17', 's']],
                        dtype=object)
    target = np.array([['no'], ['yes'], ['yes'], ['yes'], ['no'], ['no'], ['no'], ['no'], ['no'],
                       ['yes'], ['yes'], ['yes'], ['yes'], ['yes'], ['no'], ['no'], ['yes'], ['no']])
    input_data = InputData(idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                           features=features, target=target, task=task, data_type=DataTypesEnum.table)

    return train_test_data_setup(input_data, split_ratio=0.9)


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
    assert features_types[1] == "<class 'str'>"
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


def test_complicated_table_types_processed_correctly():
    """ Checking correctness of table type detection and type conversions """
    train_data, test_data = data_with_complicated_types()

    pipeline = Pipeline(PrimaryNode('dt'))
    pipeline = correct_preprocessing_params(pipeline, categorical_max_classes_th=13)
    train_predicted = pipeline.fit(train_data)
    pipeline.predict(test_data)

    # Table types corrector after fitting
    types_correctors = pipeline.preprocessor.types_correctors
    assert train_predicted.features.shape[1] == 52
    # Column with id 2 was removed be data preprocessor and column with source id 5 became 4th
    assert types_correctors[DEFAULT_SOURCE_NAME].columns_to_del[0] == 4
    # Source id 9 became 7th - column must be converted into float
    assert types_correctors[DEFAULT_SOURCE_NAME].categorical_into_float[0] == 7
    # Three columns in the table must be converted into string
    assert len(types_correctors[DEFAULT_SOURCE_NAME].numerical_into_str) == 3


def test_numerical_column_with_string_nans():
    """
    A table is generated in which one column contains numeric data along with the
    characters '?' and 'x'. These cells replace the nan. This column must be removed
    from the table.
    """
    input_data = get_data_with_string_columns()

    types_corr = TableTypesCorrector()
    # Set maximum allowed unique classes in categorical column
    types_corr.categorical_max_classes_th = 5
    data = types_corr.convert_data_for_fit(input_data)

    n_rows, n_cols = data.features.shape
    assert n_cols == 1
