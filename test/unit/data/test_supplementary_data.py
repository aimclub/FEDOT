from fedot.core.data.data import OutputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.data.test_data_merge import generate_outputs
from test.unit.tasks.test_regression import get_synthetic_regression_data


def generate_straight_pipeline():
    """ Simple linear pipeline """
    node_scaling = PrimaryNode('scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_linear = SecondaryNode('linear', nodes_from=[node_ridge])
    pipeline = Pipeline(node_linear)
    return pipeline


def test_parent_mask_correct():
    """ Test correctness of function for tables mask generation """
    correct_parent_mask = {'input_ids': [0, 1], 'flow_lens': [1, 0]}

    # Generates outputs with 1 column in prediction
    list_with_outputs, idx_1, idx_2 = generate_outputs()

    # Calculate parent mask from outputs
    data_info = SupplementaryData()
    data_info.prepare_parent_mask(list_with_outputs)
    p_mask = data_info.features_mask
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

    mask = output_example.supplementary_data.get_compound_mask()

    assert ('01', '01', '10', '10') == tuple(mask)
