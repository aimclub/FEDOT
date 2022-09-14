import operator
from itertools import product, repeat
from typing import List

import numpy as np
import pytest

from examples.simple.regression.regression_with_tuning import get_regression_dataset
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

np.random.seed(2021)


@pytest.fixture()
def output_table_1d():
    task = Task(TaskTypesEnum.classification)

    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    threshold = 0.5
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)

    data = OutputData(idx=np.arange(0, samples), features=x, predict=classes,
                      task=task, data_type=DataTypesEnum.table)
    return data


@pytest.fixture(params=[(1,), (2,), (1, 1, 1), (3, 4), (1, 2, 3, 1)],
                ids=lambda nfeatures: f'{len(nfeatures)} table(s) with {nfeatures} columns')
def output_tables(request):
    """ Generates number of tables with specified dimensions of features and predicts. """
    num_features_all = request.param
    samples = 20

    generated_target = np.random.sample((samples, 1))
    task = Task(TaskTypesEnum.regression)
    data_type = DataTypesEnum.table

    list_with_outputs = []
    for num_features in num_features_all:
        idx = np.arange(0, samples)
        generated_features = np.random.sample((samples, num_features))
        # add some random noise to predictions
        generated_predict = 0.1 * np.random.sample((samples, num_features)) * generated_target
        data_info = SupplementaryData()
        output_data = OutputData(idx=idx,
                                 features=generated_features,
                                 predict=generated_predict,
                                 target=generated_target,
                                 task=task,
                                 data_type=data_type,
                                 supplementary_data=data_info)
        list_with_outputs.append(output_data)
    return list_with_outputs


@pytest.fixture()
def unequal_outputs_table():
    """ Function for simple case with non-equal outputs in list """
    idx_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    idx_2 = np.array([2, 3, 4, 5, 6, 7, 8, 9])

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

    return list_with_outputs


def generate_output_tables(input_lengths: List[int],
                           num_features=5,
                           overlapping=False, unique=True, main_targets=None):
    if not main_targets:
        main_targets = repeat(True)
    task = Task(TaskTypesEnum.regression)
    data_type = DataTypesEnum.table
    outputs = []

    # With this index step almost for sure all indices must overlap all others
    #  but yet have distinct elements that are unique for them
    idx_start = 0
    index_range_step = min(input_lengths) // (len(input_lengths) + 1) if overlapping else 0
    for input_len, main_target in zip(input_lengths, main_targets):
        if unique:
            idx = np.arange(idx_start, idx_start + input_len)
        else:
            # Ensure there will be repetitions in index by constraining idx_range < input_len
            idx_range = input_len // 2
            idx = np.random.randint(idx_start, idx_start + idx_range, input_len)
        idx_start += index_range_step

        features = np.random.randint(0, input_len, (input_len, num_features))
        target = (features[:, -1] ** 2).reshape(-1, 1)
        metadata = SupplementaryData(is_main_target=main_target)

        output_data = OutputData(idx, task, data_type, features=features,
                                 predict=features, target=target,
                                 supplementary_data=metadata)
        outputs.append(output_data)
    return outputs


def test_data_merge_into_table(output_table_1d):
    data_1 = output_table_1d
    data_2 = output_table_1d
    data_3 = output_table_1d
    new_input_data = DataMerger.get(outputs=[data_1, data_2, data_3]).merge()
    assert np.equal(new_input_data.features,
                    np.array([data_1.predict, data_2.predict, data_3.predict])).all()


def test_data_merge_tables(output_tables):
    """ Test merge of tables of various number of predict columns. """
    merged_data = DataMerger.get(output_tables).merge()

    first_table = output_tables[0]
    assert np.equal(merged_data.idx, first_table.idx).all()
    assert merged_data.target.shape == first_table.target.shape
    expected_shape = (len(first_table.predict),
                      sum(table.predict.shape[-1] for table in output_tables))
    assert merged_data.features.shape == expected_shape
    assert np.allclose(merged_data.features,
                       np.hstack([table.predict for table in output_tables]))


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


def test_data_merge_common_index(unequal_outputs_table):
    """ Test check is the merge function can find appropriate intersections of
    indices. Set {idx_2} ∈ set {idx_1}, so intersection must be = idx_2
    """
    idx_2 = unequal_outputs_table[1].idx
    merged_data = DataMerger.get(unequal_outputs_table).merge()

    assert tuple(merged_data.idx) == tuple(idx_2)


def test_data_merge_common_index_empty(unequal_outputs_table):
    output1, output2 = unequal_outputs_table
    output1.idx *= 1000

    # ensure index is completely different
    assert len(np.intersect1d(output1.idx, output2.idx)) == 0

    with pytest.raises(ValueError, match='no common ind'):
        DataMerger.get(unequal_outputs_table).merge()


def test_data_merge_tables_with_equal_length_but_different_indices():
    input_len = 30
    outputs = generate_output_tables(input_lengths=[input_len] * 3, overlapping=True, unique=True)

    merged_data = DataMerger.get(outputs).merge()

    assert 0 < len(merged_data.idx) < input_len
    assert all(np.isin(merged_data.idx, output.idx).all() for output in outputs)


def test_data_merge_tables_with_unequal_nonunique_indices():
    outputs = generate_output_tables(input_lengths=[20, 25, 30], unique=False)
    with pytest.raises(ValueError, match='not equal and not unique'):
        DataMerger.get(outputs).merge()


def test_data_merge_datatypes_compatibility():
    available_types = [*DataTypesEnum]
    for type_pair in product(available_types, available_types):
        merged = DataMerger.get_datatype_for_merge(type_pair)
        # Able to merge only same data type
        expected = type_pair[0] if operator.eq(*type_pair) else None
        assert merged == expected


def test_data_merge_no_main_targets():
    """ Test that without main targets the 'nearest' auxiliary is selected. """
    num_outputs = 3
    outputs = generate_output_tables([30] * num_outputs, main_targets=[False] * num_outputs)

    outputs[0].supplementary_data.data_flow_length = 3
    outputs[1].supplementary_data.data_flow_length = 1  # priority output
    outputs[2].supplementary_data.data_flow_length = 2

    merged_data = DataMerger.get(outputs).merge()

    assert np.equal(merged_data.target, outputs[1].target).all()
