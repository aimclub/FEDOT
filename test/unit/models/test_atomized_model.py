from datetime import timedelta

import pytest
from sklearn.metrics import mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.models.atomized_model import AtomizedModel
from fedot.core.data.data import InputData
from cases.data.data_utils import get_scoring_case_data_paths


def create_atomized_model() -> AtomizedModel:
    """
    Example, how to create Atomized model.
    """
    chain = Chain()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = SecondaryNode('xgboost')
    node_xgboost.custom_params = {'n_components': 1}
    node_xgboost.nodes_from = [node_logit, node_lda]

    chain.add_node(node_xgboost)

    atomized_model = AtomizedModel(chain)

    return atomized_model


def create_atomized_model_with_several_atomized_models() -> AtomizedModel:
    chain = Chain()
    node_atomized_model_primary = PrimaryNode(model_type='atomized_model',
                                              atomized_model=create_atomized_model())
    node_atomized_model_secondary = SecondaryNode(model_type='atomized_model',
                                                  atomized_model=create_atomized_model())
    node_atomized_model_secondary_second = SecondaryNode(model_type='atomized_model',
                                                         atomized_model=create_atomized_model())
    node_atomized_model_secondary_third = SecondaryNode(model_type='atomized_model',
                                                        atomized_model=create_atomized_model())

    node_atomized_model_secondary.nodes_from = [node_atomized_model_primary]
    node_atomized_model_secondary_second.nodes_from = [node_atomized_model_primary]
    node_atomized_model_secondary_third.nodes_from = [node_atomized_model_secondary,
                                                      node_atomized_model_secondary_second]

    chain.add_node(node_atomized_model_secondary_third)
    atomized_model = AtomizedModel(chain)

    return atomized_model


def create_chain_with_atomized_model_last() -> Chain:
    chain = Chain()

    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = SecondaryNode('xgboost')
    node_xgboost.custom_params = {'n_components': 1}
    node_xgboost.nodes_from = [node_logit, node_lda]

    node_atomized_model = SecondaryNode(model_type='atomized_model',
                                        atomized_model=create_atomized_model())
    node_atomized_model.nodes_from = [node_xgboost, node_lda]

    chain.add_node(node_atomized_model)

    return chain


def create_chain_with_atomized_model_first() -> Chain:
    chain = Chain()
    node_atomized_model = PrimaryNode(model_type='atomized_model',
                                      atomized_model=create_atomized_model())

    node_knn = SecondaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}
    node_knn.nodes_from = [node_atomized_model]

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_atomized_model, node_knn]

    chain.add_node(node_knn_second)

    return chain


def create_chain_with_several_nested_atomized_model() -> Chain:
    chain = Chain()
    node_atomized_model = PrimaryNode(model_type='atomized_model',
                                      atomized_model=create_atomized_model_with_several_atomized_models())

    node_atomized_model_secondary = SecondaryNode(model_type='atomized_model',
                                                  atomized_model=create_atomized_model())
    node_atomized_model_secondary.nodes_from = [node_atomized_model]

    node_knn = SecondaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}
    node_knn.nodes_from = [node_atomized_model]

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_atomized_model, node_atomized_model_secondary, node_knn]

    node_atomized_model_secondary_second = \
        SecondaryNode(model_type='atomized_model',
                      atomized_model=create_atomized_model_with_several_atomized_models())

    node_atomized_model_secondary_second.nodes_from = [node_knn_second]

    chain.add_node(node_atomized_model_secondary_second)

    return chain


def create_chain_with_empty_atomized_model() -> Chain:
    chain = Chain()
    empty_chain = Chain()
    atomized_model = AtomizedModel(empty_chain)
    node_primary_atomized_model = SecondaryNode(model_type='atomized_model',
                                                atomized_model=atomized_model)

    node_logit = PrimaryNode('logit')

    node_lda = SecondaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_primary_atomized_model_second = PrimaryNode(model_type='atomized_model',
                                                     atomized_model=create_atomized_model())

    node_primary_atomized_model.nodes_from = [node_logit, node_primary_atomized_model_second]
    node_lda.nodes_from = [node_primary_atomized_model]

    chain.add_node(node_lda)

    return chain


def _create_data_for_train():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    return train_data, test_data


def _fit_predict_atomized_model(chain: Chain) -> [int, int]:
    """
    Fit and predict atomized_model and return sizes of test and predicted target.
    :params chain: chain to fit and predict
    :return: size of test target and size of predicted target
    """
    train_data, test_data = _create_data_for_train()

    chain.fit(train_data)
    predicted_value = chain.predict(test_data)

    return [len(test_data.target), len(predicted_value.predict)]


def test_fit_predict_atomized_model_correctly():
    train_data, test_data = _create_data_for_train()

    atomized_model = create_atomized_model()
    atomized_model.fit(train_data)
    predicted_value = atomized_model.predict(atomized_model.chain, test_data)

    assert len(predicted_value) == len(test_data.target)


def test_fit_predict_chain_with_atomized_model_last_correctly():
    chain = create_chain_with_atomized_model_last()
    sizes = _fit_predict_atomized_model(chain)

    assert sizes[0] == sizes[1]


def test_fit_predict_chain_with_atomized_model_first_correctly():
    chain = create_chain_with_atomized_model_first()
    sizes = _fit_predict_atomized_model(chain)

    assert sizes[0] == sizes[1]


def test_fit_predict_chain_with_several_nested_atomized_model_correctly():
    chain = create_chain_with_several_nested_atomized_model()
    sizes = _fit_predict_atomized_model(chain)

    assert sizes[0] == sizes[1]


def test_create_empty_atomized_model_raised():
    with pytest.raises(Exception):
        create_chain_with_empty_atomized_model()


def test_fine_tune_atomized_model_correct():
    train_data, test_data = _create_data_for_train()

    fine_tuned_model = create_atomized_model()
    dummy_model = create_atomized_model()

    dummy_model.fit(train_data)
    fine_tuned_model.fine_tune(train_data, iterations=5, max_lead_time=timedelta(minutes=0.1))

    after_tuning_predicted = fine_tuned_model.predict(None, test_data)
    before_tuning_predicted = dummy_model.predict(None, test_data)

    bfr_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=before_tuning_predicted)
    aft_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=after_tuning_predicted)

    assert aft_tun_mse <= bfr_tun_mse


def test_atomized_model():
    train_data, test_data = _create_data_for_train()

    chain = create_chain_with_atomized_model_first()
    atomized_model = AtomizedModel(chain)

    chain.fit(train_data)
    predicted_values = chain.predict(test_data)

    atomized_model.fit(train_data)
    predicted_atomized_values = atomized_model.predict(None, test_data)

    bfr_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=predicted_values.predict)
    aft_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=predicted_atomized_values)

    assert aft_tun_mse == bfr_tun_mse
