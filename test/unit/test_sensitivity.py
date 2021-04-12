import os

import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.sensitivity.chain_sensitivity import ChainStructureAnalyze
from fedot.sensitivity.operation_sensitivity import OperationAnalyze
from fedot.sensitivity.node_sensitivity import NodeAnalysis, NodeDeletionAnalyze, NodeReplaceOperationAnalyze, \
    NodeTuneAnalyze
from test.unit.utilities.test_chain_import_export import create_func_delete_files


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


def get_chain():
    knn_node = PrimaryNode('knn')
    lda_node = PrimaryNode('qda')
    xgb_node = PrimaryNode('xgboost')

    final = SecondaryNode('xgboost', nodes_from=[knn_node, lda_node, xgb_node])

    chain = Chain(final)

    return chain


def given_data():
    chain = get_chain()
    train_data, test_data = scoring_dataset()
    node_index = 2
    result_path = 'sa_test_result_path'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    return chain, train_data, test_data, node_index, result_path


# ------------------------------------------------------------------------------
# ChainStructureAnalysis

def test_chain_structure_analyze_init_log_defined():
    # given
    chain, train_data, test_data, node_ids, _ = given_data()
    approaches = [NodeDeletionAnalyze]
    test_log_object = default_log('test_log_chain_sa')

    # when
    chain_analyzer = ChainStructureAnalyze(chain=chain,
                                           train_data=train_data,
                                           test_data=test_data,
                                           approaches=approaches,
                                           nodes_ids_to_analyze=[node_ids],
                                           log=test_log_object)

    assert isinstance(chain_analyzer, ChainStructureAnalyze)


def test_chain_structure_analyze_init_all_and_ids_raise_exception():
    # given
    chain, train_data, test_data, _, _ = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    with pytest.raises(ValueError) as exc:
        assert ChainStructureAnalyze(chain=chain,
                                     train_data=train_data,
                                     test_data=test_data,
                                     approaches=approaches,
                                     all_nodes=True,
                                     nodes_ids_to_analyze=[2])

    assert str(exc.value) == "Choose only one parameter between " \
                             "all_nodes and nodes_ids_to_analyze"


def test_chain_structure_analyze_init_no_all_no_ids_raise_exception():
    # given
    chain, train_data, test_data, _, _ = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    with pytest.raises(ValueError) as exc:
        assert ChainStructureAnalyze(chain=chain,
                                     train_data=train_data,
                                     test_data=test_data,
                                     approaches=approaches)

    assert str(exc.value) == "Define nodes to analyze: " \
                             "all_nodes or nodes_ids_to_analyze"


def test_chain_structure_analyze_analyze():
    # given
    chain, train_data, test_data, _, result_dir = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    result = ChainStructureAnalyze(chain=chain,
                                   train_data=train_data,
                                   test_data=test_data,
                                   approaches=approaches,
                                   all_nodes=True,
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
    assert len(node_analyzer.approaches) == 3


def test_node_analysis_init_defined_approaches_and_log():
    # given
    approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze]
    test_log_object = default_log('test_log_node_sa')

    node_analyzer = NodeAnalysis(approaches=approaches,
                                 log=test_log_object)

    # then
    assert isinstance(node_analyzer, NodeAnalysis)
    assert len(node_analyzer.approaches) == 2
    assert node_analyzer.log is test_log_object


# @patch('fedot.sensitivity.sensitivity_facade.NodeAnalysis.analyze', return_value={'key': 'value'})
# @pytest.mark.skip('Works for more than 10 minutes - TODO improve it')
def test_node_analysis_analyze():
    # given
    chain, train_data, test_data, node_index, result_dir = given_data()

    # when
    node_analysis_result: dict = NodeAnalysis(path_to_save=result_dir). \
        analyze(chain=chain,
                node_id=node_index,
                train_data=train_data,
                test_data=test_data)

    assert isinstance(node_analysis_result, dict)


# ------------------------------------------------------------------------------
# NodeAnalyzeApproach

def test_node_deletion_analyze():
    # given
    chain, train_data, test_data, node_index, result_dir = given_data()

    # when
    node_analysis_result = NodeDeletionAnalyze(chain=chain,
                                               train_data=train_data,
                                               test_data=test_data,
                                               path_to_save=result_dir).analyze(node_id=node_index)

    # then
    assert isinstance(node_analysis_result, float)


def test_node_deletion_analyze_zero_node_id():
    # given
    chain, train_data, test_data, _, result_dir = given_data()

    # when
    node_analysis_result = NodeDeletionAnalyze(chain=chain,
                                               train_data=train_data,
                                               test_data=test_data,
                                               path_to_save=result_dir).analyze(node_id=0)

    # then
    assert isinstance(node_analysis_result, float)
    assert node_analysis_result == 1.0


def test_node_tune_analyze():
    # given
    chain, train_data, test_data, node_index, result_dir = given_data()

    # when
    node_analysis_result = NodeTuneAnalyze(chain=chain,
                                           train_data=train_data,
                                           test_data=test_data,
                                           path_to_save=result_dir).analyze(node_id=node_index)
    # then
    assert isinstance(node_analysis_result, float)


def test_node_replacement_analyze_defined_nodes():
    # given
    chain, train_data, test_data, node_index, result_dir = given_data()

    replacing_node = PrimaryNode('lda')

    # when
    node_analysis_result = \
        NodeReplaceOperationAnalyze(chain=chain,
                                    train_data=train_data,
                                    test_data=test_data,
                                    path_to_save=result_dir).analyze(node_id=node_index,
                                                                     nodes_to_replace_to=[replacing_node])

    # then
    assert isinstance(node_analysis_result, float)


# @pytest.mark.skip('Works for more than 10 minutes - TODO improve it')
def test_node_replacement_analyze_random_nodes_default_number():
    # given
    chain, train_data, test_data, node_index, result_dir = given_data()

    # when
    node_analysis_result = \
        (NodeReplaceOperationAnalyze(chain=chain,
                                     train_data=train_data,
                                     test_data=test_data,
                                     path_to_save=result_dir).
         analyze(node_id=node_index))

    # then
    assert isinstance(node_analysis_result, float)


# ------------------------------------------------------------------------------
# ModelAnalyze


def test_model_analyze_analyze():
    # given
    chain, train_data, test_data, node_index, result_dir = given_data()

    # when
    result = OperationAnalyze(chain=chain, train_data=train_data,
                              test_data=test_data, path_to_save=result_dir). \
        analyze(node_id=node_index, sample_size=1)

    assert type(result) is list
