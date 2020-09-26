import json
import os

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from utilities.synthetic.chain_template_new import ChainTemplate
from core.models.data import InputData
from benchmark.benchmark_utils import get_scoring_case_data_paths

CURRENT_PATH = str(os.path.dirname(__file__))


def create_static_fitted_chain() -> Chain:
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = Chain()
    node_lda = PrimaryNode('lda')

    node_lda_second = PrimaryNode('lda')
    node_rf = SecondaryNode('rf')

    node_rf.nodes_from = [node_lda, node_lda_second]

    chain.add_node(node_rf)

    chain.fit(train_data)

    return chain


def create_static_chain() -> Chain:
    chain = Chain()
    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'max_run_time_sec': 60}

    node_lda_second = PrimaryNode('lda')
    node_lda_second.custom_params = {'n_components': 100}

    node_lda_third = PrimaryNode('lda')

    node_dt = PrimaryNode('dt')
    node_dt.custom_params = {'n_estimators': 99}

    node_dt_second = SecondaryNode('dt')
    node_dt_second.custom_params = {'n_estimators': 102}
    node_dt_second.nodes_from = [node_lda, node_lda_second]

    node_rf = SecondaryNode('rf')
    node_rf.custom_params = {'n_estimators': 103}
    node_rf.nodes_from = [node_lda_third, node_lda_second]

    node_rf_second = SecondaryNode('rf')
    node_rf_second.custom_params = {'n_estimators': 104}
    node_rf_second.nodes_from = [node_dt_second, node_rf, node_dt]

    chain.add_node(node_rf_second)

    return chain


def test_static_chain_convert_to_json_correctly():
    chain = create_static_chain()
    chain_template = ChainTemplate(chain)
    json_object_actual = chain_template.export_to_json()

    with open(CURRENT_PATH + "/data/chain_to_json_test.json", 'r') as json_file:
        json_object_expected = json.load(json_file)

    assert json_object_actual == json.dumps(json_object_expected)


def test_static_fitted_chain_convert_to_json_correctly():
    chain = create_static_fitted_chain()
    chain_template = ChainTemplate(chain)
    json_object_actual = chain_template.export_to_json()

    with open(CURRENT_PATH + "/data/fitted_chain_to_json_test.json", 'r') as json_file:
        json_object_expected = json.load(json_file)

    assert json_object_actual == json.dumps(json_object_expected)
