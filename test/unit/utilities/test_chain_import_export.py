import json
import os
import shutil

import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_template import ChainTemplate, extract_subtree_root
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from test.unit.tasks.test_forecasting import get_multiscale_chain, get_synthetic_ts_data_period


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    paths = ['test_fitted_chain_convert_to_json', 'test_import_json_to_chain_correctly_1',
             'test_empty_chain_convert_to_json', 'test_chain_convert_to_json',
             'test_export_chain_to_json_correctly', 'test_import_json_to_chain_correctly_2',
             'test_fitted_chain_cache_correctness_after_export_and_import',
             'test_import_json_to_fitted_chain_correctly', 'test_export_import_for_one_chain_object_correctly_1',
             'test_export_import_for_one_chain_object_correctly_2', 'data_model_forecasting',
             'test_export_import_for_one_chain_object_correctly_3', 'data_model_classification',
             'test_absolute_relative_paths_correctly_no_exception',
             'test_import_custom_json_object_to_chain_and_fit_correctly_no_exception']

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
    chain = create_chain()
    chain.save('test_chain_convert_to_json')

    chain_fitted = create_fitted_chain()
    chain_fitted.save('test_fitted_chain_convert_to_json')

    chain_empty = Chain()
    chain_empty.save('test_empty_chain_convert_to_json')


def create_chain() -> Chain:
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

    chain = Chain(node_knn_third)

    return chain


def create_fitted_chain() -> Chain:
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = create_chain()
    chain.fit(train_data)

    return chain


def create_classification_chain_with_preprocessing():
    node_scaling = PrimaryNode('scaling')
    node_rfe = PrimaryNode('rfe_lin_class')

    xgb_node = SecondaryNode('xgboost', nodes_from=[node_scaling])
    logit_node = SecondaryNode('logit', nodes_from=[node_rfe])

    knn_root = SecondaryNode('knn', nodes_from=[xgb_node, logit_node])

    chain = Chain(knn_root)

    return chain


def create_four_depth_chain():
    knn_node = PrimaryNode('knn')
    lda_node = PrimaryNode('lda')
    xgb_node = PrimaryNode('xgboost')
    logit_node = PrimaryNode('logit')

    logit_node_second = SecondaryNode('logit', nodes_from=[knn_node, lda_node])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[logit_node])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_second, xgb_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    chain = Chain(knn_root)

    return chain


def test_export_chain_to_json_correctly():
    chain = create_chain()
    json_actual = chain.save('test_export_chain_to_json_correctly')

    json_path_load = create_correct_path('test_chain_convert_to_json')
    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected)


def test_chain_template_to_json_correctly():
    json_path_load = create_correct_path('test_chain_convert_to_json')

    chain = create_chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.convert_to_dict()

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_fitted_chain_cache_correctness_after_export_and_import():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = Chain(PrimaryNode('logit'))
    chain.fit(train_data)

    chain.save('test_fitted_chain_cache_correctness_after_export_and_import')

    json_path_load = create_correct_path('test_fitted_chain_cache_correctness_after_export_and_import')
    new_chain = Chain()
    new_chain.load(json_path_load)

    results = new_chain.fit(train_data)

    assert results is not None


def test_import_json_to_chain_correctly():
    json_path_load = create_correct_path('test_chain_convert_to_json')

    chain = Chain()
    chain.load(json_path_load)
    json_actual = chain.save('test_import_json_to_chain_correctly_1')

    chain_expected = create_chain()
    json_expected = chain_expected.save('test_import_json_to_chain_correctly_2')

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_template_to_chain_correctly():
    json_path_load = create_correct_path('test_chain_convert_to_json')

    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_chain(json_path_load)
    json_actual = chain_template.convert_to_dict()

    chain_expected = create_chain()
    chain_expected_template = ChainTemplate(chain_expected)
    json_expected = chain_expected_template.convert_to_dict()

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_to_fitted_chain_correctly():
    json_path_load = create_correct_path('test_fitted_chain_convert_to_json')

    chain = Chain()
    chain.load(json_path_load)
    json_actual = chain.save('test_import_json_to_fitted_chain_correctly')

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected)


def test_import_json_to_fitted_chain_template_correctly():
    json_path_load = create_correct_path('test_fitted_chain_convert_to_json')

    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_chain(json_path_load)
    json_actual = chain_template.convert_to_dict()

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_empty_chain_to_json_correctly():
    json_path_load = create_correct_path('test_empty_chain_convert_to_json')

    chain = Chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.convert_to_dict()

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_export_import_for_one_chain_object_correctly():
    """
    This test checks whether it is possible to upload a new model to the same object. In other words,
     apply a sequence of commands to the chain object:
    - load_chain(path_first)
    - load_chain(path_second)
    - load_chain(path_third)
    and the last command will rewrite the chain object correctly.
    """
    chain_fitted = create_fitted_chain()
    json_first = chain_fitted.save('test_export_import_for_one_chain_object_correctly_2')

    chain_fitted_after = create_chain()
    chain_fitted_after.save('test_export_import_for_one_chain_object_correctly_1')

    json_path_load_2 = create_correct_path('test_export_import_for_one_chain_object_correctly_2')
    chain_fitted_after.load(json_path_load_2)

    json_second = chain_fitted_after.save('test_export_import_for_one_chain_object_correctly_3')

    assert json_first == json_second


def test_absolute_relative_paths_correctly_no_exception():
    chain = create_chain()
    chain.save('test_absolute_relative_paths_correctly_no_exception')
    chain.save(os.path.abspath('test_absolute_relative_paths_correctly_no_exception'))

    json_path_load = create_correct_path('test_absolute_relative_paths_correctly_no_exception')
    json_path_load_abs = os.path.abspath(json_path_load)
    chain.load(json_path_load)
    chain.load(json_path_load_abs)


def test_import_custom_json_object_to_chain_and_fit_correctly_no_exception():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/test_custom_json_template.json'
    json_path_load = os.path.join(test_file_path, file)

    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = Chain()
    chain.load(json_path_load)
    chain.fit(train_data)

    chain.save('test_import_custom_json_object_to_chain_and_fit_correctly_no_exception')


def test_data_model_types_forecasting_chain_fit():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=10)

    chain = get_multiscale_chain()
    chain.fit(train_data)
    chain.save('data_model_forecasting')

    expected_len_nodes = len(chain.nodes)
    actual_len_nodes = len(ChainTemplate(chain).operation_templates)

    assert actual_len_nodes == expected_len_nodes


def test_data_model_type_classification_chain_fit():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = create_classification_chain_with_preprocessing()
    chain.fit(train_data)
    chain.save('data_model_classification')

    expected_len_nodes = len(chain.nodes)
    actual_len_nodes = len(ChainTemplate(chain).operation_templates)

    assert actual_len_nodes == expected_len_nodes


def test_extract_subtree_root():
    chain = create_four_depth_chain()
    chain_template = ChainTemplate(chain)

    expected_types = ['knn', 'logit', 'knn', 'lda', 'xgboost']
    new_root_node_id = 4

    root_node = extract_subtree_root(root_operation_id=new_root_node_id,
                                     chain_template=chain_template)

    sub_chain = Chain(root_node)
    actual_types = [node.operation.operation_type for node in sub_chain.nodes]

    assertion_list = [True if expected_types[index] == actual_types[index] else False
                      for index in range(len(expected_types))]
    assert all(assertion_list)
