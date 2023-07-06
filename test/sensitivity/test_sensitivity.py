import os
from unittest.mock import patch

import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.structural_analysis.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.structural_analysis.sa_requirements import SensitivityAnalysisRequirements
from test.integration.utilities.test_pipeline_import_export import create_func_delete_files


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
# MultiOperationAnalyze

@patch('fedot.structural_analysis.operations_hp_sensitivity.multi_operations_sensitivity.MultiOperationsHPAnalyze.analyze',
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


@patch('fedot.structural_analysis.pipeline_sensitivity.PipelineAnalysis.analyze',
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


@patch('fedot.structural_analysis.deletion_methods.multi_times_analysis.MultiTimesAnalyze.analyze',
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
