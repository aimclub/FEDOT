import pytest

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.chain_model import ChainModel
from core.models.data import InputData
from cases.data.data_utils import get_scoring_case_data_paths


def create_chain_model() -> ChainModel:
    model_chain_template = Chain()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = SecondaryNode('xgboost')
    node_xgboost.custom_params = {'n_components': 1}
    node_xgboost.nodes_from = [node_logit, node_lda]

    model_chain_template.add_node(node_xgboost)

    chain_model = ChainModel(model_chain_template)

    return chain_model


def create_chain_model_in_chain_model() -> ChainModel:
    chain = Chain()
    node_chain_model_primary = PrimaryNode(model_type='chain_model', model=create_chain_model())
    node_chain_model_secondary = SecondaryNode(model_type='chain_model', model=create_chain_model())
    node_chain_model_secondary_second = SecondaryNode(model_type='chain_model', model=create_chain_model())
    node_chain_model_secondary_third = SecondaryNode(model_type='chain_model', model=create_chain_model())

    node_chain_model_secondary.nodes_from = [node_chain_model_primary]
    node_chain_model_secondary_second.nodes_from = [node_chain_model_primary]
    node_chain_model_secondary_third.nodes_from = [node_chain_model_secondary, node_chain_model_secondary_second]

    chain.add_node(node_chain_model_secondary_third)
    chain_model = ChainModel(chain)

    return chain_model


def create_chain_with_empty_chain_model() -> Chain:
    chain = Chain()
    chain_ = Chain()
    chain_model = ChainModel(chain_)
    node_chain_model_primary = SecondaryNode(model_type='chain_model', model=chain_model)

    node_logit = PrimaryNode('logit')

    node_lda = SecondaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_chain_model_primary_second = PrimaryNode(model_type='chain_model', model=create_chain_model())

    node_chain_model_primary.nodes_from = [node_logit, node_chain_model_primary_second]
    node_lda.nodes_from = [node_chain_model_primary]

    chain.add_node(node_lda)

    return chain


def create_chain_with_chain_model_last() -> Chain:
    chain = Chain()

    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = SecondaryNode('xgboost')
    node_xgboost.custom_params = {'n_components': 1}
    node_xgboost.nodes_from = [node_logit, node_lda]

    node_chain_model = SecondaryNode(model_type='chain_model', model=create_chain_model())
    node_chain_model.nodes_from = [node_xgboost, node_lda]

    chain.add_node(node_chain_model)

    return chain


def create_chain_with_chain_model_first() -> Chain:
    chain = Chain()
    node_chain_model = PrimaryNode(model_type='chain_model', model=create_chain_model())

    node_knn = SecondaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}
    node_knn.nodes_from = [node_chain_model]

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_chain_model, node_knn]

    chain.add_node(node_knn_second)

    return chain


def create_chain_with_several_chain_models_nested() -> Chain:
    chain = Chain()
    node_chain_model = PrimaryNode(model_type='chain_model', model=create_chain_model_in_chain_model())

    node_chain_model_secondary = SecondaryNode(model_type='chain_model', model=create_chain_model())
    node_chain_model_secondary.nodes_from = [node_chain_model]

    node_knn = SecondaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}
    node_knn.nodes_from = [node_chain_model]

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_chain_model, node_chain_model_secondary, node_knn]

    node_chain_model_secondary_second = SecondaryNode(model_type='chain_model', model=create_chain_model_in_chain_model())
    node_chain_model_secondary_second.nodes_from = [node_knn_second]

    chain.add_node(node_chain_model_secondary_second)

    return chain


def _fit_predict_chain_model_correct(chain: Chain):
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    chain.fit(train_data)
    chain.predict(test_data)


def test_fit_predict_chain_model_last():
    chain = create_chain_with_chain_model_last()
    _fit_predict_chain_model_correct(chain)


def test_fit_predict_chain_model_first():
    chain = create_chain_with_chain_model_first()
    _fit_predict_chain_model_correct(chain)


def test_fit_predict_several_chain_models_nesting():
    chain = create_chain_with_several_chain_models_nested()
    _fit_predict_chain_model_correct(chain)


def test_fit_predict_empty_chain_model():
    with pytest.raises(Exception) as e:
        create_chain_with_empty_chain_model()
