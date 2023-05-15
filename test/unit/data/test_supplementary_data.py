import numpy as np
import pytest

from fedot.core.data.data import OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.data.merge.supplementary_data_merger import SupplementaryDataMerger
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID
from test.unit.tasks.test_regression import get_synthetic_regression_data

from test.unit.data.test_data_merge import unequal_outputs_table  # noqa, fixture


@pytest.fixture()
def outputs_table_with_different_types():
    """ Create datasets with different types of columns in predictions """
    task = Task(TaskTypesEnum.regression)
    idx = [0, 1, 2]
    target = [1, 2, 10]
    data_info_first = SupplementaryData(column_types={'features': [TYPE_TO_ID[str], TYPE_TO_ID[float]],
                                                      'target': [TYPE_TO_ID[int]]})
    output_first = OutputData(idx=idx, features=None,
                              predict=np.array([['a', 1.1], ['b', 2], ['c', 3]], dtype=object),
                              task=task, target=target, data_type=DataTypesEnum.table,
                              supplementary_data=data_info_first)

    data_info_second = SupplementaryData(column_types={'features': [TYPE_TO_ID[float]],
                                                       'target': [TYPE_TO_ID[int]]})
    output_second = OutputData(idx=idx, features=None,
                               predict=np.array([[2.5], [2.1], [9.3]], dtype=float),
                               task=task, target=target, data_type=DataTypesEnum.table,
                               supplementary_data=data_info_second)

    return [output_first, output_second]


def generate_straight_pipeline():
    """ Simple linear pipeline """
    node_scaling = PipelineNode('scaling')
    node_ridge = PipelineNode('ridge', nodes_from=[node_scaling])
    node_linear = PipelineNode('linear', nodes_from=[node_ridge])
    pipeline = Pipeline(node_linear)
    return pipeline


def test_parent_mask_correct(unequal_outputs_table):  # noqa, fixture
    """ Test correctness of function for tables mask generation """
    correct_parent_mask = {'input_ids': [0, 1], 'flow_lens': [1, 0]}

    # Calculate parent mask from outputs
    main_output = DataMerger.find_main_output(unequal_outputs_table)
    p_mask = SupplementaryDataMerger(unequal_outputs_table, main_output).prepare_parent_mask()

    assert tuple(p_mask['input_ids']) == tuple(correct_parent_mask['input_ids'])
    assert tuple(p_mask['flow_lens']) == tuple(correct_parent_mask['flow_lens'])


def test_calculate_data_flow_len_correct():
    """ Function checks whether the number of nodes visited by the data block
     is calculated correctly """

    # Pipeline consists of 3 nodes
    simple_pipeline = generate_straight_pipeline()
    data = get_synthetic_regression_data(n_samples=100, n_features=2)

    simple_pipeline.fit(data)
    predict_output = simple_pipeline.predict(data)

    assert predict_output.supplementary_data.data_flow_length == 2


def test_get_compound_mask_correct():
    """ Checking whether the procedure for combining lists with keys is
    performed correctly for features_mask """

    synthetic_mask = {'input_ids': [0, 0, 1, 1],
                      'flow_lens': [1, 1, 0, 0]}
    output_example = OutputData(idx=[0, 0], features=[0, 0], predict=[0, 0],
                                task=Task(TaskTypesEnum.regression),
                                target=[0, 0], data_type=DataTypesEnum.table,
                                supplementary_data=SupplementaryData(features_mask=synthetic_mask))

    mask = output_example.supplementary_data.compound_mask

    assert ('01', '01', '10', '10') == tuple(mask)


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


def test_define_types_after_merging(outputs_table_with_different_types):
    """ Check if column types for features table perform correctly """
    outputs = outputs_table_with_different_types
    # new_idx, features, target, task, d_type, updated_info = DataMerger(outputs).merge()
    merged_data = DataMerger.get(outputs).merge()
    updated_info = merged_data.supplementary_data

    features_types = updated_info.column_types['features']
    target_types = updated_info.column_types['target']

    # Target type must stay the same
    ancestor_target_type = outputs[0].supplementary_data.column_types['target'][0]
    assert target_types[0] == ancestor_target_type
    assert len(features_types) == 3
    assert tuple(features_types) == (TYPE_TO_ID[str], TYPE_TO_ID[float], TYPE_TO_ID[float])
