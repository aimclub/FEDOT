import json
import os
import shutil

import numpy as np
import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate, extract_subtree_root
from test.unit.tasks.test_forecasting import get_multiscale_pipeline, get_ts_data


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    paths = ['test_fitted_pipeline_convert_to_json', 'test_import_json_to_pipeline_correctly_1',
             'test_empty_pipeline_convert_to_json', 'test_pipeline_convert_to_json',
             'test_export_pipeline_to_json_correctly', 'test_import_json_to_pipeline_correctly_2',
             'test_fitted_pipeline_cache_correctness_after_export_and_import',
             'test_import_json_to_fitted_pipeline_correctly', 'test_export_import_for_one_pipeline_object_correctly_1',
             'test_export_import_for_one_pipeline_object_correctly_2', 'data_model_forecasting',
             'test_export_import_for_one_pipeline_object_correctly_3', 'data_model_classification',
             'test_absolute_relative_paths_correctly_no_exception',
             'test_import_custom_json_object_to_pipeline_and_fit_correctly_no_exception']

    delete_files = create_func_delete_files(paths)
    delete_files()
    create_json_models_files()
    request.addfinalizer(delete_files)


def create_func_delete_files(paths):
    """
    Create function to delete files that created after tests.
    """

    def wrapper():
        for path in paths:
            path = create_correct_path(path, True)
            if path is not None and os.path.isdir(path):
                shutil.rmtree(path)

    return wrapper


def create_correct_path(path: str, dirname_flag: bool = False):
    """
    Create path with time which was created during the testing process.
    """

    for dirname in next(os.walk(os.path.curdir))[1]:
        if dirname.endswith(path):
            if dirname_flag:
                return dirname
            else:
                file = os.path.join(dirname, path + '.json')
                return file
    return None


def create_json_models_files():
    """
    Creating JSON's files for test before tests.
    """
    pipeline = create_pipeline()
    pipeline.save('test_pipeline_convert_to_json')

    pipeline_fitted = create_fitted_pipeline()
    pipeline_fitted.save('test_fitted_pipeline_convert_to_json')

    pipeline_empty = Pipeline()
    pipeline_empty.save('test_empty_pipeline_convert_to_json')


def create_pipeline() -> Pipeline:
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = PrimaryNode('xgboost')

    node_knn = PrimaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_lda, node_knn]

    node_logit_second = SecondaryNode('logit')
    node_logit_second.nodes_from = [node_xgboost, node_lda]

    node_lda_second = SecondaryNode('lda')
    node_lda_second.custom_params = {'n_components': 1}
    node_lda_second.nodes_from = [node_logit_second, node_knn_second, node_logit]

    node_xgboost_second = SecondaryNode('xgboost')
    node_xgboost_second.nodes_from = [node_logit, node_logit_second, node_knn]

    node_knn_third = SecondaryNode('knn')
    node_knn_third.custom_params = {'n_neighbors': 8}
    node_knn_third.nodes_from = [node_lda_second, node_xgboost_second]

    pipeline = Pipeline(node_knn_third)

    return pipeline


def create_fitted_pipeline() -> Pipeline:
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    pipeline = create_pipeline()
    pipeline.fit(train_data)

    return pipeline


def create_classification_pipeline_with_preprocessing():
    node_scaling = PrimaryNode('scaling')
    node_rfe = PrimaryNode('rfe_lin_class')

    xgb_node = SecondaryNode('xgboost', nodes_from=[node_scaling])
    logit_node = SecondaryNode('logit', nodes_from=[node_rfe])

    knn_root = SecondaryNode('knn', nodes_from=[xgb_node, logit_node])

    pipeline = Pipeline(knn_root)

    return pipeline


def create_four_depth_pipeline():
    knn_node = PrimaryNode('knn')
    lda_node = PrimaryNode('lda')
    xgb_node = PrimaryNode('xgboost')
    logit_node = PrimaryNode('logit')

    logit_node_second = SecondaryNode('logit', nodes_from=[knn_node, lda_node])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[logit_node])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_second, xgb_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def test_export_pipeline_to_json_correctly():
    pipeline = create_pipeline()
    json_actual, fitted_models_dict = pipeline.save('test_export_pipeline_to_json_correctly')

    json_path_load = create_correct_path('test_pipeline_convert_to_json')
    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected, indent=4)
    assert fitted_models_dict is None


def test_pipeline_template_to_json_correctly():
    json_path_load = create_correct_path('test_pipeline_convert_to_json')

    pipeline = create_pipeline()
    pipeline_template = PipelineTemplate(pipeline)
    json_actual = pipeline_template.convert_to_dict(root_node=pipeline.root_node)

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_fitted_pipeline_cache_correctness_after_export_and_import():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    pipeline = create_classification_pipeline_with_preprocessing()
    pipeline.fit(train_data)
    pipeline.save('test_fitted_pipeline_cache_correctness_after_export_and_import')
    prediction = pipeline.predict(test_data)

    new_pipeline = Pipeline()
    new_pipeline.load(create_correct_path('test_fitted_pipeline_cache_correctness_after_export_and_import'))

    new_prediction = new_pipeline.predict(test_data)

    assert np.array_equal(prediction.predict, new_prediction.predict)
    assert new_pipeline.is_fitted


def test_import_json_to_pipeline_correctly():
    json_path_load = create_correct_path('test_pipeline_convert_to_json')

    pipeline = Pipeline()
    pipeline.load(json_path_load)
    json_actual, _ = pipeline.save('test_import_json_to_pipeline_correctly_1')

    pipeline_expected = create_pipeline()
    json_expected, _ = pipeline_expected.save('test_import_json_to_pipeline_correctly_2')

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_template_to_pipeline_correctly():
    json_path_load = create_correct_path('test_pipeline_convert_to_json')

    pipeline = Pipeline()
    pipeline_template = PipelineTemplate(pipeline)
    pipeline_template.import_pipeline(json_path_load)
    json_actual = pipeline_template.convert_to_dict()

    pipeline_expected = create_pipeline()
    pipeline_expected_template = PipelineTemplate(pipeline_expected)
    json_expected = pipeline_expected_template.convert_to_dict()

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_to_fitted_pipeline_correctly():
    json_path_load = create_correct_path('test_fitted_pipeline_convert_to_json')

    pipeline = Pipeline()
    pipeline.load(json_path_load)
    json_actual, _ = pipeline.save('test_import_json_to_fitted_pipeline_correctly')

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected, indent=4)


def test_import_json_to_fitted_pipeline_template_correctly():
    json_path_load = create_correct_path('test_fitted_pipeline_convert_to_json')

    pipeline = Pipeline()
    pipeline_template = PipelineTemplate(pipeline)
    pipeline_template.import_pipeline(json_path_load)
    json_actual = pipeline_template.convert_to_dict(pipeline.root_node)

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_empty_pipeline_to_json_correctly():
    json_path_load = create_correct_path('test_empty_pipeline_convert_to_json')

    pipeline = Pipeline()
    pipeline_template = PipelineTemplate(pipeline)
    json_actual = pipeline_template.convert_to_dict()

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_local_save_for_pipeline_correctly():
    pipeline_fitted = create_fitted_pipeline()
    json, dict_fitted = pipeline_fitted.save()
    assert json is not None
    assert len(dict_fitted) == 9
    assert dict_fitted['operation_3'] is not None


def test_export_import_for_one_pipeline_object_correctly():
    """
    This test checks whether it is possible to upload a new model to the same object. In other words,
     apply a sequence of commands to the pipeline object:
    - load_pipeline(path_first)
    - load_pipeline(path_second)
    - load_pipeline(path_third)
    and the last command will rewrite the pipeline object correctly.
    """
    pipeline_fitted = create_fitted_pipeline()
    json_first, _ = pipeline_fitted.save('test_export_import_for_one_pipeline_object_correctly_2')

    pipeline_fitted_after = create_pipeline()
    pipeline_fitted_after.save('test_export_import_for_one_pipeline_object_correctly_1')

    json_path_load_2 = create_correct_path('test_export_import_for_one_pipeline_object_correctly_2')
    pipeline_fitted_after.load(json_path_load_2)

    json_second, dict_fitted = pipeline_fitted_after.save('test_export_import_for_one_pipeline_object_correctly_3')

    assert json_first == json_second
    assert len(dict_fitted) == 9
    assert dict_fitted['operation_3'] is not None


def test_absolute_relative_paths_correctly_no_exception():
    pipeline = create_pipeline()
    pipeline.save('test_absolute_relative_paths_correctly_no_exception')
    pipeline.save(os.path.abspath('test_absolute_relative_paths_correctly_no_exception'))

    json_path_load = create_correct_path('test_absolute_relative_paths_correctly_no_exception')
    json_path_load_abs = os.path.abspath(json_path_load)
    pipeline.load(json_path_load)
    pipeline.load(json_path_load_abs)


def test_import_custom_json_object_to_pipeline_and_fit_correctly_no_exception():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/test_custom_json_template.json'
    json_path_load = os.path.join(test_file_path, file)

    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    pipeline = Pipeline()
    pipeline.load(json_path_load)
    pipeline.fit(train_data)

    pipeline.save('test_import_custom_json_object_to_pipeline_and_fit_correctly_no_exception')


def test_data_model_types_forecasting_pipeline_fit():
    train_data, test_data = get_ts_data(forecast_length=10)

    pipeline = get_multiscale_pipeline()
    pipeline.fit(train_data)
    pipeline.save('data_model_forecasting')

    expected_len_nodes = len(pipeline.nodes)
    actual_len_nodes = len(PipelineTemplate(pipeline).operation_templates)

    assert actual_len_nodes == expected_len_nodes


def test_data_model_type_classification_pipeline_fit():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    pipeline = create_classification_pipeline_with_preprocessing()
    pipeline.fit(train_data)
    pipeline.save('data_model_classification')

    expected_len_nodes = len(pipeline.nodes)
    actual_len_nodes = len(PipelineTemplate(pipeline).operation_templates)

    assert actual_len_nodes == expected_len_nodes


def test_extract_subtree_root():
    pipeline = create_four_depth_pipeline()
    pipeline_template = PipelineTemplate(pipeline)

    expected_types = ['knn', 'logit', 'knn', 'lda', 'xgboost']
    new_root_node_id = 4

    root_node = extract_subtree_root(root_operation_id=new_root_node_id,
                                     pipeline_template=pipeline_template)

    sub_pipeline = Pipeline(root_node)
    actual_types = [node.operation.operation_type for node in sub_pipeline.nodes]

    assertion_list = [True if expected_types[index] == actual_types[index] else False
                      for index in range(len(expected_types))]
    assert all(assertion_list)
