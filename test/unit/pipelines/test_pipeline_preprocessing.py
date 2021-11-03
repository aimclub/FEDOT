import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


def data_with_only_categorical_features():
    """ Generate tabular data with only categorical features """
    task = Task(TaskTypesEnum.regression)
    features = np.array([['1', '0', '1'],
                         ['0', '1', '0'],
                         ['1', '1', '0'],
                         ['1', '1', '1']], dtype=object)
    input = InputData(idx=[0, 1, 2, 3], features=features,
                      target=np.array([[0], [1], [2], [3]]),
                      task=task,  data_type=DataTypesEnum.table)

    return input


def data_with_too_mach_nans():
    """ Generate tabular data with too much nan's in numpy array.
    Columns with ids 1 and 2 have nans more than 30% in their structure.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, np.nan, np.nan],
                         [np.nan, np.nan, np.nan],
                         [3, np.nan, np.nan],
                         [7, np.nan, np.nan],
                         [8, '1', np.nan],
                         [np.nan, '0', 23],
                         [9, '0', 22],
                         [9, '0', 2],
                         [9, np.nan, 14],
                         [9, '1', np.nan]], dtype=object)
    target = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table)

    return train_input


def test_only_categorical_data_process_correct():
    """ Check if data with only categorical features processed correctly """
    pipeline = Pipeline(PrimaryNode('ridge'))
    categorical_data = data_with_only_categorical_features()

    pipeline.fit(categorical_data)


def test_nans_columns_processed_correct():
    """ Check if data with nans processed correctly """
    pipeline = Pipeline(PrimaryNode('ridge'))
    data_with_nans = data_with_too_mach_nans()

    pipeline.fit(data_with_nans)

    # Ridge should use only one feature to make prediction
    fitted_ridge = pipeline.nodes[0]
    coefficients = fitted_ridge.operation.fitted_operation.coef_
    coefficients_shape = coefficients.shape

    assert 1 == coefficients_shape[1]
