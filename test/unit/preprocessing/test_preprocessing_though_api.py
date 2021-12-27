import numpy as np

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.data_types import NAME_CLASS_STR


def data_with_only_categorical_features():
    """ Generate tabular data with only categorical features. All of them are binary. """
    supp_data = SupplementaryData(column_types={'features': [NAME_CLASS_STR] * 3})
    task = Task(TaskTypesEnum.regression)
    features = np.array([['a', '0', '1'],
                         ['b', '1', '0'],
                         ['c', '1', '0']], dtype=object)
    input_data = InputData(idx=[0, 1, 2], features=features,
                           target=np.array([0, 1, 2]),
                           task=task,  data_type=DataTypesEnum.table,
                           supplementary_data=supp_data)

    return input_data


def data_with_too_much_nans():
    """ Generate tabular data with too much nan's in numpy array (inf values also must be signed as nan).
    Columns with ids 1 and 2 have nans more than 90% in their structure.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, np.inf, np.nan],
                         [np.nan, np.inf, np.nan],
                         [3, np.inf, np.nan],
                         [7, np.inf, np.nan],
                         [8, np.nan, np.nan],
                         [np.nan, np.nan, 23],
                         [9, np.inf, np.nan],
                         [9, np.inf, np.nan],
                         [9, np.inf, np.nan],
                         [9, '1', np.inf],
                         [8, np.nan, np.inf]], dtype=object)
    target = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_spaces_and_nans_in_features():
    """
    Generate InputData with categorical features with leading and
    trailing spaces. Dataset contains np.nan also.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([['1 ', '1 '],
                         [np.nan, ' 0'],
                         [' 1 ', np.nan],
                         ['1 ', '0  '],
                         ['0  ', '  1'],
                         ['1 ', '  0']], dtype=object)
    target = np.array([[0], [1], [2], [3], [4], [5]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_nans_in_target_column():
    """ Generate InputData with np.nan values in target column """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, 2],
                         [2, 2],
                         [0, 3],
                         [2, 3],
                         [3, 4],
                         [1, 3]])
    target = np.array([[0], [1], [np.nan], [np.nan], [4], [5]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_nans_in_multi_target():
    """
    Generate InputData with np.nan values in target columns.
    So the multi-output regression task is solved.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, 2],
                         [2, 2],
                         [0, 3],
                         [2, 3],
                         [3, 4],
                         [1, 3]])
    target = np.array([[0, 2], [1, 3], [np.nan, np.nan], [3, np.nan], [4, 4], [5, 6]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_categorical_target(with_nan: bool = False):
    """
    Generate dataset for classification task where target column is defined as
    string categories (e.g. 'red', 'green'). Dataset is generated so that when
    split into training and test in the test sample in the target will always
    be a new category.

    :param with_nan: is there a need to generate target column with np.nan
    """
    task = Task(TaskTypesEnum.classification)
    features = np.array([[0, 0],
                         [0, 1],
                         [8, 8],
                         [8, 9]])
    if with_nan:
        target = np.array(['blue', np.nan, np.nan, 'di'], dtype=object)
    else:
        target = np.array(['blue', 'da', 'ba', 'di'], dtype=str)
    train_input = InputData(idx=[0, 1, 2, 3], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def test_correct_api_dataset_preprocessing():
    """ Check if dataset preprocessing was performed correctly when API launch using. """
    funcs = [data_with_only_categorical_features, data_with_too_much_nans,
             data_with_spaces_and_nans_in_features, data_with_nans_in_target_column,
             data_with_nans_in_multi_target]

    # Check for all datasets
    for data_generator in funcs:
        input_data = data_generator()
        fedot_model = Fedot(problem='regression')
        pipeline = fedot_model.fit(input_data, predefined_model='auto')
        assert pipeline is not None


def test_categorical_target_processed_correctly():
    """ Check if categorical target for classification task first converted
    into integer values and then perform inverse operation. API tested in this
    test.
    """
    classification_data = data_with_categorical_target()
    train_data, test_data = train_test_data_setup(classification_data)

    fedot_model = Fedot(problem='classification')
    fedot_model.fit(train_data, predefined_model='auto')
    predicted = fedot_model.predict(test_data)

    # Predicted label must be close to 'di' label (so, right prediction is 'ba')
    assert predicted[0] == 'ba'
