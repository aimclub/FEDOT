import json
from utilities.synthetic.model_json_vice_versa import serializing_json_from_chain, deserializing_chain_from_json
from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode


def test_json_from_chain():
    chain = Chain()
    node_tpot = PrimaryNode('tpot')
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

    json_object_expected = serializing_json_from_chain(chain)

    json_object = json.dumps({
        "root_node": {
            "model_id": None,
            "model_type": "dt",
            "params": {
                "n_estimators": 100
            },
            "nodes_from": [
                {
                    "model_id": None,
                    "model_type": "dt",
                    "params": None,
                    "nodes_from": [
                        {
                            "model_id": None,
                            "model_type": "rf",
                            "params": {
                                "n_estimators": 100
                            },
                            "nodes_from": [
                                {
                                    "model_id": None,
                                    "model_type": "tpot",
                                    "params": {
                                        "max_run_time_sec": 60
                                    },
                                    "nodes_from": []
                                },
                                {
                                    "model_id": None,
                                    "model_type": "lda",
                                    "params": {
                                        "n_components": 60,
                                        "max_iter": 30,
                                        "n_jobs": 6,
                                        "learning_method": "batch",
                                        "verbose": 1
                                    },
                                    "nodes_from": []
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "depth": 4
    })

    assert json_object == json_object_expected


def test_chain_from_json():
    json_object = '''{
        "root_node": {
            "model_id": null,
            "model_type": "dt",
            "params": {
                "n_estimators": 100
            },
            "nodes_from": [
                {
                    "model_id": null,
                    "model_type": "dt",
                    "params": null,
                    "nodes_from": [
                        {
                            "model_id": null,
                            "model_type": "rf",
                            "params": {
                                "n_estimators": 100
                            },
                            "nodes_from": [
                                {
                                    "model_id": null,
                                    "model_type": "tpot",
                                    "params": {
                                        "max_run_time_sec": 60
                                    },
                                    "nodes_from": []
                                },
                                {
                                    "model_id": null,
                                    "model_type": "lda",
                                    "params": {
                                        "n_components": 60,
                                        "max_iter": 30,
                                        "n_jobs": 6,
                                        "learning_method": "batch",
                                        "verbose": 1
                                    },
                                    "nodes_from": []
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "depth": 4
    }'''

    chain = Chain()
    node_tpot = PrimaryNode('tpot')
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

    assert deserializing_chain_from_json(json_object) == chain
