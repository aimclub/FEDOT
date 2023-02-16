import os
from unittest.mock import patch

import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.sensitivity.deletion_methods.multi_times_analysis import MultiTimesAnalyze
from fedot.sensitivity.node_sa_approaches import NodeAnalysis, NodeDeletionAnalyze, NodeReplaceOperationAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis
from fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.sensitivity.operations_hp_sensitivity.one_operation_sensitivity import OneOperationHPAnalyze
from fedot.sensitivity.pipeline_sensitivity import PipelineAnalysis
from fedot.sensitivity.pipeline_sensitivity_facade import PipelineSensitivityAnalysis
from fedot.sensitivity.sa_requirements import SensitivityAnalysisRequirements
from test.unit.utilities.test_pipeline_import_export import create_func_delete_files


@pytest.fixture(scope='session', autouse=True)
def delete_files(request):
    paths = ['sa_test_result_path']
    delete_files = create_func_delete_files(paths)
    delete_files()
    request.addfinalizer(delete_files)


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
# PipelineStructureAnalysis

def test_pipeline_structure_analyze_init_log_defined():
    # given
    pipeline, train_data, test_data, nodes_to_analyze, _ = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    pipeline_analyzer = NodesAnalysis(pipeline=pipeline,
                                      train_data=train_data,
                                      test_data=test_data,
                                      approaches=approaches,
                                      nodes_to_analyze=[nodes_to_analyze])

    assert isinstance(pipeline_analyzer, NodesAnalysis)


def test_pipeline_structure_analyze_analyze():
    # given
    pipeline, train_data, test_data, _, result_dir = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    result = NodesAnalysis(pipeline=pipeline,
                           train_data=train_data,
                           test_data=test_data,
                           approaches=approaches,
                           path_to_save=result_dir).analyze()
    assert isinstance(result, dict)


# ------------------------------------------------------------------------------
# NodeAnalysis


def test_node_analysis_init_default():
    # given

    # when
    node_analyzer = NodeAnalysis()

    # then
    assert isinstance(node_analyzer, NodeAnalysis)
    assert len(node_analyzer.approaches) == 2


def test_node_analysis_init_defined_approaches():
    # given
    approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze]

    node_analyzer = NodeAnalysis(approaches=approaches)

    # then
    assert isinstance(node_analyzer, NodeAnalysis)
    assert len(node_analyzer.approaches) == 2


# @patch('fedot.sensitivity.sensitivity_facade.NodeAnalysis.analyze', return_value={'key': 'value'})
# @pytest.mark.skip('Works for more than 10 minutes - TODO improve it')
def test_node_analysis_analyze():
    # given
    pipeline, train_data, test_data, node_to_analyze, result_dir = given_data()

    # when
    node_analysis_result: dict = NodeAnalysis(path_to_save=result_dir). \
        analyze(pipeline=pipeline,
                node=node_to_analyze,
                train_data=train_data,
                test_data=test_data)

    assert isinstance(node_analysis_result, dict)


# ------------------------------------------------------------------------------
# NodeAnalyzeApproach

def test_node_deletion_analyze():
    # given
    pipeline, train_data, test_data, node_to_analyze, result_dir = given_data()

    # when
    node_analysis_result = NodeDeletionAnalyze(pipeline=pipeline,
                                               train_data=train_data,
                                               test_data=test_data,
                                               path_to_save=result_dir).analyze(node=node_to_analyze)

    # then
    assert isinstance(node_analysis_result, list)


def test_node_deletion_sample_method():
    # given
    _, train_data, test_data, _, result_dir = given_data()
    primary_first = PipelineNode('knn')
    primary_second = PipelineNode('knn')
    central = PipelineNode('rf', nodes_from=[primary_first, primary_second])
    secondary_first = PipelineNode('lda', nodes_from=[central])
    secondary_second = PipelineNode('lda', nodes_from=[central])
    root = PipelineNode('logit', nodes_from=[secondary_first, secondary_second])
    pipeline_with_multiple_children = Pipeline(nodes=root)

    # when
    result = NodeDeletionAnalyze(pipeline=pipeline_with_multiple_children,
                                 train_data=train_data,
                                 test_data=test_data,
                                 path_to_save=result_dir).sample(pipeline_with_multiple_children.nodes[2])

    # then
    assert result is None


def test_node_deletion_analyze_zero_node_id():
    # given
    pipeline, train_data, test_data, _, result_dir = given_data()

    # when
    node_analysis_result = NodeDeletionAnalyze(pipeline=pipeline,
                                               train_data=train_data,
                                               test_data=test_data,
                                               path_to_save=result_dir).analyze(node=pipeline.root_node)

    # then
    assert isinstance(node_analysis_result, list)
    assert node_analysis_result == [1.0]


def test_node_replacement_analyze_defined_nodes():
    # given
    pipeline, train_data, test_data, node_to_analyze, result_dir = given_data()

    replacing_node = PipelineNode('lda')

    # when
    node_analysis_result = \
        NodeReplaceOperationAnalyze(pipeline=pipeline,
                                    train_data=train_data,
                                    test_data=test_data,
                                    path_to_save=result_dir).analyze(node=node_to_analyze,
                                                                     nodes_to_replace_to=[replacing_node])

    # then
    assert isinstance(node_analysis_result, list)


def test_node_replacement_analyze_random_nodes_default_number():
    # given
    pipeline, train_data, test_data, node_to_analyze, result_dir = given_data()

    # when
    node_analysis_result = \
        (NodeReplaceOperationAnalyze(pipeline=pipeline,
                                     train_data=train_data,
                                     test_data=test_data,
                                     path_to_save=result_dir).
         analyze(node=node_to_analyze))

    # then
    assert isinstance(node_analysis_result, list)


# ------------------------------------------------------------------------------
# OneOperationAnalyze


def test_one_operation_analyze_analyze():
    # given
    pipeline, train_data, test_data, node_to_analyze, result_dir = given_data()

    requirements = SensitivityAnalysisRequirements(hyperparams_analysis_samples_size=1)

    # when
    result = OneOperationHPAnalyze(pipeline=pipeline, train_data=train_data, test_data=test_data,
                                   requirements=requirements, path_to_save=result_dir). \
        analyze(node=node_to_analyze)

    assert type(result) is dict


# ------------------------------------------------------------------------------
# MultiOperationAnalyze

@patch('fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity.MultiOperationsHPAnalyze.analyze',
       return_value=[{'key': 'value'}])
def test_multi_operations_analyze_analyze(analyze_method):
    # given
    pipeline, train_data, test_data, node_index, result_dir = given_data()

    # when
    result = MultiOperationsHPAnalyze(pipeline=pipeline,
                                      train_data=train_data,
                                      test_data=test_data, path_to_save=result_dir).analyze(sample_size=1)

    # then
    assert type(result) is list
    assert analyze_method.called


# ------------------------------------------------------------------------------
# SA Facade

def test_pipeline_sensitivity_facade_init():
    # given
    pipeline, train_data, test_data, node_to_analyze, result_dir = given_data()

    # when
    sensitivity_facade = PipelineSensitivityAnalysis(pipeline=pipeline,
                                                     train_data=train_data,
                                                     test_data=test_data,
                                                     nodes_to_analyze=[node_to_analyze],
                                                     path_to_save=result_dir)
    # then
    assert type(sensitivity_facade) is PipelineSensitivityAnalysis


@patch('fedot.sensitivity.pipeline_sensitivity_facade.PipelineSensitivityAnalysis.analyze', return_value=None)
def test_pipeline_sensitivity_facade_analyze(analyze_method):
    # given
    pipeline, train_data, test_data, node_to_analyze, result_dir = given_data()

    # when
    sensitivity_analyze_result = PipelineSensitivityAnalysis(pipeline=pipeline,
                                                             train_data=train_data,
                                                             test_data=test_data,
                                                             nodes_to_analyze=[node_to_analyze],
                                                             path_to_save=result_dir).analyze()

    # then
    assert sensitivity_analyze_result is None
    assert analyze_method.called


# ------------------------------------------------------------------------------
# SA Non-structural analysis

def test_pipeline_non_structure_analyze_init():
    # given
    pipeline, train_data, test_data, node_index, result_dir = given_data()
    approaches = [MultiOperationsHPAnalyze]

    # when
    non_structure_analyzer = PipelineAnalysis(pipeline=pipeline,
                                              train_data=train_data,
                                              test_data=test_data,
                                              approaches=approaches,
                                              path_to_save=result_dir)

    # then
    assert type(non_structure_analyzer) is PipelineAnalysis


@patch('fedot.sensitivity.pipeline_sensitivity.PipelineAnalysis.analyze',
       return_value=[{'key': 'value'}])
def test_pipeline_analysis_analyze(analyze_method):
    # given
    pipeline, train_data, test_data, node_index, result_dir = given_data()

    requirements = SensitivityAnalysisRequirements(hyperparams_analysis_samples_size=1)

    # when
    non_structure_analyze_result = PipelineAnalysis(pipeline=pipeline,
                                                    train_data=train_data,
                                                    test_data=test_data,
                                                    requirements=requirements,
                                                    path_to_save=result_dir).analyze()

    # then
    assert type(non_structure_analyze_result) is list
    assert analyze_method.called


# ------------------------------------------------------------------------------
# Multi-Times-Analysis for Pipeline size decrease

def test_multi_times_analyze_init_defined_approaches():
    # given
    pipeline, train_data, test_data, node_index, result_dir = given_data()
    approaches = [NodeDeletionAnalyze]
    test_data, valid_data = train_test_data_setup(test_data, split_ratio=0.5)

    # when
    analyzer = MultiTimesAnalyze(pipeline=pipeline,
                                 train_data=train_data,
                                 test_data=test_data,
                                 valid_data=valid_data,
                                 case_name='test_case_name',
                                 path_to_save=result_dir,
                                 approaches=approaches)

    # then
    assert type(analyzer) is MultiTimesAnalyze


@patch('fedot.sensitivity.deletion_methods.multi_times_analysis.MultiTimesAnalyze.analyze',
       return_value=1.0)
def test_multi_times_analyze_analyze(analyze_method):
    # given
    pipeline, train_data, test_data, node_index, result_dir = given_data()
    test_data, valid_data = train_test_data_setup(test_data, split_ratio=0.5)

    # when
    analyze_result = MultiTimesAnalyze(pipeline=pipeline,
                                       train_data=train_data,
                                       test_data=test_data,
                                       valid_data=valid_data,
                                       case_name='test_case_name',
                                       path_to_save=result_dir).analyze()

    # then
    assert type(analyze_result) is float
    assert analyze_method.called
