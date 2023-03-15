import numpy as np
import pandas as pd
from golem.core.log import default_log

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root
from fedot.preprocessing.data_types import TableTypesCorrector, apply_type_transformation
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME
from test.unit.preprocessing.test_pipeline_preprocessing import data_with_mixed_types_in_each_column, \
    correct_preprocessing_params


def get_mixed_data_with_str_and_float_values(idx: int = None):
    task = Task(TaskTypesEnum.classification)
    features = np.array([['exal', 'exal', 'exal'],
                         ['greka', 0, 'greka'],
                         ['cherez', 1, 'cherez'],
                         ['reku', 0, 0],
                         ['vidit', 1, 1],
                         [1, 0, 1],
                         [1, 1, 0],
                         [0, 0, 0]], dtype=object)
    target = np.array([['no'], ['yes'], ['yes'], ['yes'], ['no'], ['no'], ['no'], ['no']])
    if isinstance(idx, int):
        input_data = InputData(idx=np.arange(8),
                               features=features[:, idx], target=target, task=task, data_type=DataTypesEnum.table)
    else:
        input_data = InputData(idx=np.arange(8),
                               features=features, target=target, task=task, data_type=DataTypesEnum.table)
    return input_data


def get_data_with_string_columns():
    file_path = fedot_project_root().joinpath('test/data/data_with_mixed_column.csv')
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
    encoding_label = PipelineNode('label_encoding')
    poly_node = PipelineNode('poly_features', nodes_from=[encoding_label])
    rfe_node = PipelineNode('rfe_lin_class', nodes_from=[poly_node])
    final_node = PipelineNode('dt', nodes_from=[rfe_node])
    pipeline = Pipeline(final_node)

    return pipeline


def data_with_complicated_types():
    """
    Generate InputData with mixed column types.
    Train part - idx [0 ... 15], test part - idx [16, 17]
    Column description by indices:
        0) int column with single np.nan value - nans must be filled in
        1) int column with nans more than 90% - column must be removed
        due to the fact that inf will be replaced with nans
        2) int-float column with categorical values (number of unique values = 12) -
        categorical indices must be converted into float, then due to number
        of unique values less than 13 - perform converting column into str type
        3) int column the same as 2) column but with additional 13th label in the test part
        4) int column (number of unique values = 4) - must be converted into string
        5) str-int column with words and numerical cells - must be converted to int,
        true str values replaced with nans and filled
        6) str column with unique categories 'a', 'b', 'c' and spaces in labels.
        New category 'd' arise in the test part. Categories will be encoded using OHE
        7) str binary column - must be converted into integer, nan cells must be filled in
        8) int binary column and nans - nan cells must be filled in
        9) str column with truly int values as strings - must be converted into float column,
        '?' and 'error' must be replaced with nans and then filled in
        10) str column with truly categorical values - must stay remained and encoded using OHE
    """

    task = Task(TaskTypesEnum.classification)
    features = np.array([[0, np.nan, 1, 1, 1, 'monday', 'a ', 'true', 1, '0', 'a'],
                         [np.nan, 5, 2, 2, 0, 'tuesday', 'b', np.nan, 0, '1', np.inf],
                         [2, np.nan, 3, 3, np.nan, 3, 'c', 'false', 1, '?', 'c'],
                         [3, np.nan, 4, 4, 3.0, 4, '  a  ', 'true', 0, 'error', 'd'],
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
                         [16, 2, 5, 12, 0, 3, '   d       ', 'true', 1, '?', 'r'],
                         [17, 3, 6, 13, 0, 4, '  d      ', 'false', 0, '17', 's']],
                        dtype=object)
    target = np.array([['no'], ['yes'], ['yes'], ['yes'], ['no'], ['no'], ['no'], ['no'], ['no'],
                       ['yes'], ['yes'], ['yes'], ['yes'], ['yes'], ['no'], ['no'], ['yes'], ['no']])
    input_data = InputData(idx=np.arange(18),
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

    assert len(features_types) == 4
    assert len(target_types) == 2
    assert features_types[0] == "<class 'str'>"
    assert features_types[1] == "<class 'str'>"
    assert features_types[2] == "<class 'str'>"
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

    pipeline = Pipeline(PipelineNode('dt'))
    pipeline = correct_preprocessing_params(pipeline, categorical_max_uniques_th=13)
    train_predicted = pipeline.fit(train_data)
    pipeline.predict(test_data)

    # Table types corrector after fitting
    types_correctors = pipeline.preprocessor.types_correctors
    assert train_predicted.features.shape[1] == 57
    # Source id 9 became 7th - column must be converted into float
    assert types_correctors[DEFAULT_SOURCE_NAME].categorical_into_float[0] == 1
    # Three columns in the table must be converted into string
    assert len(types_correctors[DEFAULT_SOURCE_NAME].numerical_into_str) == 4


def test_numerical_column_with_string_nans():
    """
    A table is generated in which one column contains numeric data along with the
    characters '?' and 'x'. These cells replace the nan. This column must be removed
    from the table.
    """
    input_data = get_data_with_string_columns()

    types_corr = TableTypesCorrector()
    # Set maximum allowed unique classes in categorical column
    types_corr.categorical_max_uniques_th = 5
    data = types_corr.convert_data_for_fit(input_data)

    n_rows, n_cols = data.features.shape
    assert n_cols == 1


def test_binary_pseudo_string_column_process_correctly():
    """ Checks if pseudo strings with int/float values in it process correctly with binary classification """
    task = Task(TaskTypesEnum.classification)
    features = np.array([['1'],
                         ['1.0'],
                         ['0.0'],
                         ['1'],
                         ['1.0'],
                         ['0'],
                         ['0.0'],
                         ['1']], dtype=object)
    target = np.array([['no'], ['yes'], ['yes'], ['yes'], ['no'], ['no'], ['no'], ['no']])
    input_data = InputData(idx=np.arange(8),
                           features=features, target=target, task=task, data_type=DataTypesEnum.table)

    train_data, test_data = train_test_data_setup(input_data, split_ratio=0.9)

    pipeline = Pipeline(PipelineNode('dt'))
    pipeline = correct_preprocessing_params(pipeline)
    train_predicted = pipeline.fit(train_data)

    assert train_predicted.features.shape[1] == 1
    assert all(isinstance(el[0], float) for el in train_predicted.features)


def fit_predict_cycle_for_testing(idx: int):
    input_data = get_mixed_data_with_str_and_float_values(idx=idx)
    train_data, test_data = train_test_data_setup(input_data, split_ratio=0.9)

    pipeline = Pipeline(PipelineNode('dt'))
    pipeline = correct_preprocessing_params(pipeline)
    train_predicted = pipeline.fit(train_data)
    return train_predicted


def test_mixed_column_with_str_and_float_values():
    """ Checks if columns with different data type ratio process correctly """

    # column with index 0 must be converted to string and encoded with OHE
    train_predicted = fit_predict_cycle_for_testing(idx=0)
    assert train_predicted.features.shape[1] == 5
    assert all(isinstance(el, np.ndarray) for el in train_predicted.features)

    # column with index 1 must be converted to float and the gaps must be filled
    train_predicted = fit_predict_cycle_for_testing(idx=1)
    assert train_predicted.features.shape[1] == 1
    assert all(isinstance(el[0], float) for el in train_predicted.features)

    # column with index 2 must be removed due to unclear type of data
    try:
        _ = fit_predict_cycle_for_testing(idx=2)
    except ValueError:
        pass


def test_str_numbers_with_dots_and_commas_in_predict():
    """ Checks that if training part type was defined as int than predict part will be correctly
    converted to ints even if it contains str with dots/commas"""
    task = Task(TaskTypesEnum.classification)
    features = np.array([['8,5'],
                         ['4.9'],
                         ['3,2'],
                         ['6.1']], dtype=object)
    target = np.array([['no'], ['yes'], ['yes'], ['yes']])
    input_data = InputData(idx=np.arange(4),
                           features=features, target=target, task=task, data_type=DataTypesEnum.table)

    transformed_predict = apply_type_transformation(table=input_data.features, column_types=['int'],
                                                    log=default_log('test_str_numbers_with_dots_and_commas_in_predict'))

    assert all(transformed_predict == np.array([[8], [4], [3], [6]]))
