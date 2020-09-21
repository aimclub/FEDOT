import json
from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from utilities.synthetic.chain_template_new import ChainTemplate


def create_static_chain_1() -> Chain:
    chain = Chain()
    node_tpot = PrimaryNode('lda')
    node_tpot.custom_params = {'max_run_time_sec': 60}

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 60, 'max_iter': 30, 'n_jobs': 6, 'learning_method': 'batch', 'verbose': 1}

    node_rf = SecondaryNode('rf')
    node_rf.custom_params = {'n_estimators': 100}
    node_rf.nodes_from = [node_tpot, node_lda]

    node_dt = SecondaryNode('dt')
    node_dt.nodes_from = [node_rf]

    node_dt_second = SecondaryNode('dt')
    node_dt_second.custom_params = {'n_estimators': 100}
    node_dt_second.nodes_from = [node_dt]

    chain.add_node(node_dt_second)

    return chain


def test_chain_convert_to_chain_template():
    chain = create_static_chain_1()
    chain_template = ChainTemplate(chain)
    json_object_actual = chain_template.export_to_json()

    with open('test/data/chain_test_1.json', 'r') as json_file:
        json_object_expected = json.load(json_file)

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
