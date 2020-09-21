import json

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from utilities.synthetic.chain_template_new import ChainTemplate


def create_static_chain_1() -> Chain:
    chain = Chain()
    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'max_run_time_sec': 60}

    node_lda_second = PrimaryNode('lda')
    node_lda_second.custom_params = {'n_components': 60, 'max_iter': 30, 'n_jobs': 6, 'learning_method': 'batch', 'verbose': 1}

    node_rf = SecondaryNode('rf')
    node_rf.custom_params = {'n_estimators': 100}
    node_rf.nodes_from = [node_lda, node_lda_second]

    node_dt = SecondaryNode('dt')
    node_dt.nodes_from = [node_rf]

    node_dt_second = SecondaryNode('dt')
    node_dt_second.custom_params = {'n_estimators': 100}
    node_dt_second.nodes_from = [node_dt]

    chain.add_node(node_dt_second)

    return chain


def create_static_chain_2() -> Chain:
    chain = Chain()
    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'max_run_time_sec': 60}

    node_lda_second = PrimaryNode('lda')
    node_lda_second.custom_params = {'n_components': 100}

    node_lda_third = PrimaryNode('lda')
    node_lda_third.custom_params = {'n_components': 101}

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


def test_static_chain_convert_to_json_correctly_1():
    chain = create_static_chain_1()
    chain_template = ChainTemplate(chain)
    json_object_actual = chain_template.export_to_json()



    with open('test/data/chain_test_1.json', 'r') as json_file:
        json_object_expected = json.load(json_file)

    assert json_object_actual == json.dumps(json_object_expected)


def test_static_chain_convert_to_json_correctly_2():
    chain = create_static_chain_2()
    chain_template = ChainTemplate(chain)
    json_object_actual = chain_template.export_to_json()

    print(json_object_actual)

    with open('test/data/chain_test_2.json', 'r') as json_file:
        json_object_expected = json.load(json_file)

    print()
    print(json_object_expected)

    assert json_object_actual == json.dumps(json_object_expected)


# def test_static_chain_convert_to_json_correctly():
#     chain = create_static_chain_1()
#
#     with open('data/chain_test_1.json', 'r') as json_file:
#         json_object_expected = json.load(json_file)
#
#     json_object_actual = serializing_json_from_chain(chain)
#
#     assert json_object_expected == json_object_actual
#
#
# def test_static_chain_convert_from_json_correctly():
#     chain = create_static_chain_1()
#
#     with open('data/chain_test_1.json', 'r') as json_file:
#         json_object = json.load(json_file)
#
#     chain_expected = deserializing_chain_from_json(json_object)
#
#     assert chain_expected == chain
