import json
import os
import shutil

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from fedot.api.main import Fedot
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate, extract_subtree_root
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from test.unit.api.test_main_api import get_dataset
from test.unit.data_operations.test_data_operations_implementations import get_mixed_data
from test.unit.multimodal.data_generators import get_single_task_multimodal_tabular_data
from test.unit.pipelines.test_decompose_pipelines import get_classification_data
from test.unit.preprocessing.test_preprocessing_though_api import \
    data_with_spaces_and_nans_in_features
from test.unit.tasks.test_forecasting import get_multiscale_pipeline, get_simple_ts_pipeline, get_ts_data


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    paths = ['test_fitted_pipeline_convert_to_json', 'test_import_json_to_pipeline_correctly_1',
             'test_empty_pipeline_convert_to_json', 'test_pipeline_convert_to_json',
             'test_export_pipeline_to_json_correctly', 'test_import_json_to_pipeline_correctly_2',
             'test_fitted_pipeline_cache_correctness_after_export_and_import',
             'test_import_json_to_fitted_pipeline_correctly', 'test_export_import_for_one_pipeline_object_correctly_1',
             'test_export_import_for_one_pipeline_object_correctly_2', 'data_model_forecasting',
             'test_export_import_for_one_pipeline_object_correctly_3', 'data_model_classification',
             'test_absolute_relative_paths_correctly_no_exception', 'test_export_one_hot_encoding_operation',
             'test_import_custom_json_object_to_pipeline_and_fit_correctly_no_exception',
             'test_save_pipeline_with_np_int_type', 'test_pipeline_with_preprocessing_serialized_correctly',
             'test_multimodal_pipeline_serialized_correctly', 'test_load_though_api_perform_correctly']

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
                file = os.path.abspath(os.path.join(dirname, path + '.json'))
                return file
    return None


def create_json_models_files():
    """
    Creating JSON's files for test before tests.
    """
    pipeline = create_pipeline()
    pipeline.save('test_pipeline_convert_to_json', is_final_path=True)

    pipeline_fitted = create_fitted_pipeline()
    pipeline_fitted.save('test_fitted_pipeline_convert_to_json', is_final_path=True)

    pipeline_empty = Pipeline()
    pipeline_empty.save('test_empty_pipeline_convert_to_json', is_final_path=True)


def create_pipeline() -> Pipeline:
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.parameters = {'n_components': 1}

    node_rf = PrimaryNode('rf')

    node_knn = PrimaryNode('knn')
    node_knn.parameters = {'n_neighbors': 9}

    node_knn_second = SecondaryNode('knn')
    node_knn_second.parameters = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_lda, node_knn]

    node_logit_second = SecondaryNode('logit')
    node_logit_second.nodes_from = [node_rf, node_lda]

    node_lda_second = SecondaryNode('lda')
    node_lda_second.parameters = {'n_components': 1}
    node_lda_second.nodes_from = [node_logit_second, node_knn_second, node_logit]

    node_rf_second = SecondaryNode('rf')
    node_rf_second.nodes_from = [node_logit, node_logit_second, node_knn]

    node_knn_third = SecondaryNode('knn')
    node_knn_third.parameters = {'n_neighbors': 8}
    node_knn_third.nodes_from = [node_lda_second, node_rf_second]

    pipeline = Pipeline(node_knn_third)

    return pipeline


def create_fitted_pipeline() -> Pipeline:
    train_data, _ = get_classification_data()

    pipeline = create_pipeline()
    pipeline.fit(train_data)

    return pipeline


def create_classification_pipeline_with_preprocessing():
    node_scaling = PrimaryNode('scaling')
    node_rfe = PrimaryNode('rfe_lin_class')

    rf_node = SecondaryNode('rf', nodes_from=[node_scaling])
    logit_node = SecondaryNode('logit', nodes_from=[node_rfe])

    knn_root = SecondaryNode('knn', nodes_from=[rf_node, logit_node])

    pipeline = Pipeline(knn_root)

    return pipeline


def create_four_depth_pipeline():
    knn_node = PrimaryNode('knn')
    lda_node = PrimaryNode('lda')
    rf_node = PrimaryNode('rf')
    logit_node = PrimaryNode('logit')

    logit_node_second = SecondaryNode('logit', nodes_from=[knn_node, lda_node])
    rf_node_second = SecondaryNode('rf', nodes_from=[logit_node])

    qda_node_third = SecondaryNode('qda', nodes_from=[rf_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_second, rf_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def test_export_pipeline_to_json_correctly():
    pipeline = create_pipeline()
    json_actual, fitted_models_dict = pipeline.save('test_export_pipeline_to_json_correctly',
                                                    is_final_path=True)

    json_path_load = create_correct_path('test_export_pipeline_to_json_correctly')

    with open(json_path_load) as json_file:
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
    train_data, test_data = get_classification_data()

    pipeline = create_classification_pipeline_with_preprocessing()
    pipeline.fit(train_data)
    pipeline.save('test_fitted_pipeline_cache_correctness_after_export_and_import', is_final_path=True)
    prediction = pipeline.predict(test_data)

    json_load_path = create_correct_path('test_fitted_pipeline_cache_correctness_after_export_and_import')
    new_pipeline = Pipeline.from_serialized(json_load_path)

    new_prediction = new_pipeline.predict(test_data)

    assert np.array_equal(prediction.predict, new_prediction.predict)
    assert new_pipeline.is_fitted


def test_import_json_to_pipeline_correctly():
    json_path_load = create_correct_path('test_pipeline_convert_to_json')

    pipeline = Pipeline.from_serialized(json_path_load)
    json_actual, _ = pipeline.save('test_import_json_to_pipeline_correctly_1', is_final_path=True)

    pipeline_expected = create_pipeline()
    json_expected, _ = pipeline_expected.save('test_import_json_to_pipeline_correctly_2', is_final_path=True)

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

    pipeline = Pipeline.from_serialized(json_path_load)
    json_actual, _ = pipeline.save('test_import_json_to_fitted_pipeline_correctly', is_final_path=True)

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
    json, dict_fitted = pipeline_fitted.save(is_final_path=True)
    assert json is not None
    assert len(dict_fitted) == 10
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
    json_first, _ = pipeline_fitted.save('test_export_import_for_one_pipeline_object_correctly_2', is_final_path=True)

    pipeline_fitted_after = create_pipeline()
    pipeline_fitted_after.save('test_export_import_for_one_pipeline_object_correctly_1', is_final_path=True)

    json_path_load_2 = create_correct_path('test_export_import_for_one_pipeline_object_correctly_2')
    pipeline_fitted_after.load(json_path_load_2)

    json_second, dict_fitted = pipeline_fitted_after.save('test_export_import_for_one_pipeline_object_correctly_3',
                                                          is_final_path=True)

    assert json_first == json_second
    assert len(dict_fitted) == 10
    assert dict_fitted['operation_3'] is not None


def test_absolute_relative_paths_correctly_no_exception():
    pipeline = create_pipeline()
    pipeline.save('test_absolute_relative_paths_correctly_no_exception', is_final_path=True)
    pipeline.save(os.path.abspath('test_absolute_relative_paths_correctly_no_exception'), is_final_path=True)

    json_path_load = create_correct_path('test_absolute_relative_paths_correctly_no_exception')
    json_path_load_abs = os.path.abspath(json_path_load)
    pipeline.load(json_path_load)
    pipeline.load(json_path_load_abs)


def test_import_custom_json_object_to_pipeline_and_fit_correctly_no_exception():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/test_custom_json_template.json'
    json_path_load = os.path.join(test_file_path, file)

    train_data, _ = get_classification_data()

    pipeline = Pipeline.from_serialized(json_path_load)

    pipeline.fit(train_data)

    pipeline.save('test_import_custom_json_object_to_pipeline_and_fit_correctly_no_exception', is_final_path=True)


def test_export_without_path_correctly():
    pipeline = create_pipeline()

    save_not_fitted_without_path, not_fitted_dict = pipeline.save(is_final_path=True)
    assert len(save_not_fitted_without_path) > 0
    assert not_fitted_dict is None

    fitted_pipeline = create_fitted_pipeline()

    save_fitted_without_path, fitted_dict = fitted_pipeline.save(is_final_path=True)
    assert len(save_fitted_without_path) > 0
    assert fitted_dict is not None


def test_data_model_types_forecasting_pipeline_fit():
    train_data, test_data = get_ts_data(forecast_length=10)

    pipeline = get_multiscale_pipeline()
    pipeline.fit(train_data)
    pipeline.save('data_model_forecasting', is_final_path=True)

    expected_len_nodes = pipeline.length
    actual_len_nodes = len(PipelineTemplate(pipeline).operation_templates)

    assert actual_len_nodes == expected_len_nodes


def test_data_model_type_classification_pipeline_fit():
    train_data, _ = get_classification_data()

    pipeline = create_classification_pipeline_with_preprocessing()
    pipeline.fit(train_data)
    pipeline.save('data_model_classification', is_final_path=True)

    expected_len_nodes = pipeline.length
    actual_len_nodes = len(PipelineTemplate(pipeline).operation_templates)

    assert actual_len_nodes == expected_len_nodes


def test_extract_subtree_root():
    pipeline = create_four_depth_pipeline()
    pipeline_template = PipelineTemplate(pipeline)

    expected_types = ['knn', 'logit', 'knn', 'lda', 'rf']
    new_root_node_id = 4

    root_node = extract_subtree_root(root_operation_id=new_root_node_id,
                                     pipeline_template=pipeline_template)

    sub_pipeline = Pipeline(root_node)
    actual_types = [node.operation.operation_type for node in sub_pipeline.nodes]

    assertion_list = [True if expected_types[index] == actual_types[index] else False
                      for index in range(len(expected_types))]
    assert all(assertion_list)


def test_one_hot_encoder_serialization():
    train_data, test_data, threshold = get_dataset('classification')

    pipeline = Pipeline()
    one_hot_node = PrimaryNode('one_hot_encoding')
    final_node = SecondaryNode('dt', nodes_from=[one_hot_node])
    pipeline.add_node(final_node)

    pipeline.fit(train_data)
    prediction_before_export = pipeline.predict(test_data)

    pipeline.save('test_export_one_hot_encoding_operation', is_final_path=True)

    pipeline_after = Pipeline.from_serialized(create_correct_path('test_export_one_hot_encoding_operation'))
    prediction_after_export = pipeline_after.predict(test_data)

    assert np.array_equal(prediction_before_export.features, prediction_after_export.features)


def test_save_pipeline_with_np_int_type():
    pipeline = get_simple_ts_pipeline()
    pipeline.nodes[1].parameters["test"] = np.int32(42)
    pipeline.save(path='test_save_pipeline_with_np_int_type', is_final_path=True)


def test_pipeline_with_preprocessing_serialized_correctly():
    """
    Pipeline with preprocessing blocks must be serializable as well as any other pipeline.
    Pipeline doesn't contain any preprocessing operation in its structure. So, imputation and gap-filling
    (imputation) should be performed as preprocessing
    """
    save_path = 'test_pipeline_with_preprocessing_serialized_correctly'

    scaling_node = PrimaryNode('scaling')
    single_node_pipeline = Pipeline(SecondaryNode('ridge', nodes_from=[scaling_node]))

    mixed_input = get_mixed_data(task=Task(TaskTypesEnum.regression),
                                 extended=True)

    # Calculate metric before serialization
    single_node_pipeline.fit(mixed_input)
    before_output = single_node_pipeline.predict(mixed_input)
    mae_before = mean_absolute_error(mixed_input.target, before_output.predict)

    single_node_pipeline.save(path=save_path, is_final_path=True)

    pipeline_after = Pipeline.from_serialized(create_correct_path(save_path))

    after_output = pipeline_after.predict(mixed_input)
    mae_after = mean_absolute_error(mixed_input.target, after_output.predict)

    assert np.isclose(mae_before, mae_after)


def test_multimodal_pipeline_serialized_correctly():
    """
    Checks that MultiModal pipelining together with complex preprocessing
    (gap filling and categorical encoding) is serialized correctly
    """
    save_path = 'test_multimodal_pipeline_serialized_correctly'
    mm_data, pipeline = get_single_task_multimodal_tabular_data()

    pipeline.fit(mm_data)
    before_save_predicted_labels = pipeline.predict(mm_data, output_mode='labels')
    pipeline.save(path=save_path, is_final_path=True)

    pipeline_loaded = Pipeline.from_serialized(create_correct_path(save_path))
    after_load_predicted_labels = pipeline_loaded.predict(mm_data, output_mode='labels')

    assert np.array_equal(before_save_predicted_labels.predict, after_load_predicted_labels.predict)


def test_old_serialized_paths_load_correctly():
    """
    In older versions of FEDOT, pipelines were loaded using paths written to a json file.
    The paths were represented as strings, not as lists. This test checks if the old version
    pipelines can be loaded successfully using the new logic.
    """
    path = os.path.join(fedot_project_root(), 'test', 'data', 'pipeline_with_old_paths', 'pipeline_with_old_paths.json')

    pipeline_loaded = Pipeline.from_serialized(path)

    assert pipeline_loaded.nodes is not None


def test_load_though_api_perform_correctly():
    """ Test API wrapper for pipeline loading """
    input_data = data_with_spaces_and_nans_in_features()

    model = Fedot(problem='regression')
    obtained_pipeline = model.fit(input_data, predefined_model='ridge')
    predictions = model.predict(input_data)

    # Save pipeline
    obtained_pipeline.save('test_load_though_api_perform_correctly', is_final_path=True)

    loaded_model = Fedot(problem='regression')
    loaded_model.load(create_correct_path('test_load_though_api_perform_correctly'))
    loaded_predictions = loaded_model.predict(input_data)

    assert np.array_equal(predictions, loaded_predictions)
