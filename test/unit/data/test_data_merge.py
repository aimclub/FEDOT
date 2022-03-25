import numpy as np
import pytest

from examples.simple.regression.regression_with_tuning import get_regression_dataset
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.merge import DataMerger
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

np.random.seed(2021)


def generate_outputs():
    """ Function for simple case with non-equal outputs in list """
    idx_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    idx_2 = [2, 3, 4, 5, 6, 7, 8, 9]

    task = Task(TaskTypesEnum.regression)
    generated_target = np.random.sample((len(idx_1), 1))
    generated_features = np.random.sample((len(idx_1), 2))

    list_with_outputs = []
    for idx, data_flow_len in zip([idx_1, idx_2], [1, 0]):
        data_info = SupplementaryData(data_flow_length=data_flow_len)
        output_data = OutputData(idx=idx,
                                 features=generated_features[idx, :],
                                 predict=generated_target[idx, :],
                                 task=task,
                                 target=generated_target[idx, :],
                                 data_type=DataTypesEnum.table,
                                 supplementary_data=data_info)
        list_with_outputs.append(output_data)

    return list_with_outputs, idx_1, idx_2


def generate_outputs_with_different_types():
    """ Create datasets with different types of columns in predictions """
    task = Task(TaskTypesEnum.regression)
    idx = [0, 1, 2]
    target = [1, 2, 10]
    data_info_first = SupplementaryData(column_types={'features': ["<class 'str'>", "<class 'float'>"],
                                                      'target': ["<class 'int'>"]})
    output_first = OutputData(idx=idx, features=None,
                              predict=np.array([['a', 1.1], ['b', 2], ['c', 3]], dtype=object),
                              task=task, target=target, data_type=DataTypesEnum.table,
                              supplementary_data=data_info_first)

    data_info_second = SupplementaryData(column_types={'features': ["<class 'float'>"],
                                                       'target': ["<class 'int'>"]})
    output_second = OutputData(idx=idx, features=None,
                               predict=np.array([[2.5], [2.1], [9.3]], dtype=float),
                               task=task, target=target, data_type=DataTypesEnum.table,
                               supplementary_data=data_info_second)

    return [output_first, output_second]


def test_data_merge_in_pipeline():
    """ Test check is the pipeline can correctly work with dynamic changes in
    tables during the fit process
    """

    #   ridge
    #  /     \ (merge operation)
    # |   ransac_lin_reg (remove several lines in table)
    #  \     /
    #  scaling

    node_scaling = PrimaryNode('scaling')

    node_lin_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_final = SecondaryNode('ridge', nodes_from=[node_lin_ransac, node_scaling])
    pipeline = Pipeline(node_final)

    features_options = {'informative': 2, 'bias': 2.0}
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options=features_options,
                                                              samples_amount=100,
                                                              features_amount=5)
    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    # Fit and predict
    pipeline.fit_from_scratch(train_input)
    prediction = pipeline.predict(train_input)

    assert prediction is not None


def test_data_merge_function():
    """ Test check is the merge function can find appropriate intersections of
    indices. Set {idx_2} ∈ set {idx_1}, so intersection must be = idx_2
    """

    list_with_outputs, idx_1, idx_2 = generate_outputs()

    merged_data = DataMerger.get(list_with_outputs).merge()

    assert tuple(merged_data.idx) == tuple(idx_2)


@pytest.fixture()
def output_dataset():
    task = Task(TaskTypesEnum.classification)

    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    threshold = 0.5
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = OutputData(idx=np.arange(0, 100), features=x, predict=classes,
                      task=task, data_type=DataTypesEnum.table)

    return data


def test_data_merge_into_table(output_dataset):
    data_1 = output_dataset
    data_2 = output_dataset
    data_3 = output_dataset
    new_input_data = DataMerger.get(outputs=[data_1, data_2, data_3]).merge()
    assert new_input_data.features.all() == np.array(
        [data_1.predict, data_2.predict, data_3.predict]).all()


def test_data_merge_tables_of_unequal_lengths_with_nonunique_indices():
    num_features = 5
    input_lengths = [30,25,20]
    task = Task(TaskTypesEnum.regression)
    data_type = DataTypesEnum.table

    outputs = []
    for input_len in input_lengths:
        features = np.random.randint(0, 10, (input_len, num_features))
        idx = features[:, 0].astype(str)
        target = features ** 2
        # input_data = InputData(idx, features, task, data_type, target=target)
        output_data = OutputData(idx, features, task, data_type, predict=features, target=target)
        outputs.append(output_data)

    next_input = DataMerger.get(outputs).merge()
    assert len(next_input.idx) <= min(input_lengths)


def test_define_parents_with_equal_lengths():
    """
    Check the processing of the case when the decompose operation receives
    data whose flow_lens is not different. In this case, the data that came
    from the data_operation node is used as the "Data parent".

    Such case is common for time series forecasting pipelines. So we imitate
    merged output from ARIMA and lagged operations
    """
    sd = SupplementaryData(is_main_target=True,
                           data_flow_length=1,
                           features_mask={'input_ids': [0, 0, 0, 1, 1, 1],
                                          'flow_lens': [0, 0, 0, 0, 0, 0]},
                           previous_operations=['arima', 'lagged'])
    features_mask = np.array(sd.compound_mask)
    unique_features_masks = np.unique(features_mask)

    model_parent, data_parent = sd.define_parents(unique_features_masks, task=TaskTypesEnum.ts_forecasting)

    assert model_parent == '00'
    assert data_parent == '10'


def test_define_types_after_merging():
    """ Check if column types for features table perform correctly """
    outputs = generate_outputs_with_different_types()
    # new_idx, features, target, task, d_type, updated_info = DataMerger(outputs).merge()
    merged_data = DataMerger.get(outputs).merge()
    updated_info = merged_data.supplementary_data

    features_types = updated_info.column_types['features']
    target_types = updated_info.column_types['target']

    # Target type must stay the same
    ancestor_target_type = outputs[0].supplementary_data.column_types['target'][0]
    assert target_types[0] == ancestor_target_type
    assert len(features_types) == 3
    assert tuple(features_types) == ("<class 'str'>", "<class 'float'>", "<class 'float'>")
