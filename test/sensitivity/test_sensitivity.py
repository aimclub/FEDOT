import os
from unittest.mock import patch

from examples.real_cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.structural_analysis.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze


def scoring_dataset():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    return train_data, test_data


def get_pipeline():
    knn_node = PipelineNode('knn')
    lda_node = PipelineNode('qda')
    rf_node = PipelineNode('rf')

    final = PipelineNode('rf', nodes_from=[knn_node, lda_node, rf_node])

    pipeline = Pipeline(final)

    return pipeline


def given_data():
    pipeline = get_pipeline()
    train_data, test_data = scoring_dataset()
    node_index = 2
    result_path = 'sa_test_result_path'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    return pipeline, train_data, test_data, pipeline.nodes[node_index], result_path


# ------------------------------------------------------------------------------
# MultiOperationAnalyze

@patch('fedot.structural_analysis.operations_hp_sensitivity.multi_operations_sensitivity.'
       'MultiOperationsHPAnalyze.analyze',
       return_value=[{'key': 'value'}])
def test_multi_operations_analyze_analyze(analyze_method):
    # given
    pipeline, train_data, test_data, node_index, result_dir = given_data()

    # when
    result = MultiOperationsHPAnalyze(pipeline=pipeline,
                                      train_data=train_data,
                                      test_data=test_data, path_to_save=result_dir).analyze(sample_size=1)

    # then
    assert isinstance(result, list)
    assert analyze_method.called
