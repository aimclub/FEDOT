import os
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.composer import ComposerRequirements
from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.node import NodeGenerator
from core.composer.node import PrimaryNode, SecondaryNode
from core.composer.random_composer import RandomSearchComposer
from core.models.data import InputData
from core.models.model import LogRegression
from core.models.model import XGBoost, KNN
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/test_dataset2.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


def test_composer_hierarchical_chain():
    composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    empty_data = InputData(np.zeros(1), np.zeros(1), np.zeros(1))
    composer_requirements = ComposerRequirements(primary=[LogRegression(), XGBoost()],
                                                 secondary=[LogRegression()])
    new_chain = composer.compose_chain(data=empty_data,
                                       initial_chain=None,
                                       composer_requirements=composer_requirements,
                                       metrics=None)

    assert len(new_chain.nodes) == 3
    assert isinstance(new_chain.nodes[0], PrimaryNode)
    assert isinstance(new_chain.nodes[1], PrimaryNode)
    assert isinstance(new_chain.nodes[2], SecondaryNode)
    assert new_chain.nodes[2].nodes_from[0] is new_chain.nodes[0]
    assert new_chain.nodes[2].nodes_from[1] is new_chain.nodes[1]
    assert new_chain.nodes[1].nodes_from is None


def test_composer_flat_chain():
    composer = DummyComposer(DummyChainTypeEnum.flat)
    empty_data = InputData(np.zeros(1), np.zeros(1), np.zeros(1))
    composer_requirements = ComposerRequirements(primary=[LogRegression()],
                                                 secondary=[LogRegression(), XGBoost()])
    new_chain = composer.compose_chain(data=empty_data,
                                       initial_chain=None,
                                       composer_requirements=composer_requirements,
                                       metrics=None)

    assert len(new_chain.nodes) == 3
    assert isinstance(new_chain.nodes[0], PrimaryNode)
    assert isinstance(new_chain.nodes[1], SecondaryNode)
    assert isinstance(new_chain.nodes[2], SecondaryNode)
    assert new_chain.nodes[1].nodes_from[0] is new_chain.nodes[0]
    assert new_chain.nodes[2].nodes_from[0] is new_chain.nodes[1]
    assert new_chain.nodes[0].nodes_from is None


def custom_chain(data):
    chain_created_by_hand = Chain()
    last_node = NodeGenerator.secondary_node(XGBoost())
    for requirement_model in (KNN(), KNN()):
        new_node = NodeGenerator.primary_node(requirement_model, data)
        chain_created_by_hand.add_node(new_node)
        last_node.nodes_from.append(new_node)
    chain_created_by_hand.add_node(last_node)
    return chain_created_by_hand


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_random_composer(data_fixture, request):
    random.seed(1)
    assert 2 > 1
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    models_repo = ModelTypesRepository()
    available_model_names = models_repo.search_model_types_by_attributes(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))

    models_impl = [models_repo.model_by_id(model_name) for model_name in available_model_names]

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    random_composer = RandomSearchComposer(iter_num=1)
    req = ComposerRequirements(primary=models_impl, secondary=models_impl)
    chain_random_composed = random_composer.compose_chain(data=dataset_to_compose,
                                                          initial_chain=None,
                                                          composer_requirements=req,
                                                          metrics=metric_function)

    predicted_random_composed = chain_random_composed.predict(dataset_to_validate)

    roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_random_composed.predict)

    assert roc_on_valid_random_composed > 0.6


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    models_repo = ModelTypesRepository()

    available_model_names = models_repo.search_model_types_by_attributes(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))
    models_impl = [models_repo.model_by_id(model_name) for model_name in available_model_names]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    gp_composer = GPComposer()
    req = GPComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=1, crossover_prob=0.4,
        mutation_prob=0.5)
    chain_gp_composed = gp_composer.compose_chain(data=dataset_to_compose,
                                                  initial_chain=None,
                                                  composer_requirements=req,
                                                  metrics=metric_function)

    predicted_gp_composed = chain_gp_composed.predict(dataset_to_validate)

    roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                       y_score=predicted_gp_composed.predict)

    assert roc_on_valid_gp_composed > 0.6


@pytest.mark.skip("the dataset doesn't provide necessary condition fulfillment (too small)")
@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_quality(data_fixture, request):
    random.seed(1)
    data = request.getfixturevalue(data_fixture)

    models_impl = [XGBoost(), KNN()]

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain_created_by_hand = custom_chain(data)

    predicted_created_by_hand = chain_created_by_hand.predict(data).predict
    data.target = [int(round(i)) for i in predicted_created_by_hand]

    composer_requirements = GPComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=3, pop_size=5, num_of_generations=1,
        crossover_prob=0.8, mutation_prob=0.8)

    # Create GP-based composer
    composer = GPComposer()

    chain_created_by_evo_alg = composer.compose_chain(data=data,
                                                      initial_chain=None,
                                                      composer_requirements=composer_requirements,
                                                      metrics=metric_function, is_visualise=False)

    predicted_created_by_evo_alg = chain_created_by_evo_alg.predict(data).predict

    roc_auc_chain_created_by_hand = roc_auc(y_true=data.target, y_score=predicted_created_by_hand)
    roc_auc_chain_evo_alg = roc_auc(y_true=data.target, y_score=predicted_created_by_evo_alg)
    print("model created by hand prediction:", roc_auc_chain_created_by_hand)
    print("gp composed model prediction:", roc_auc_chain_evo_alg)

    assert chain_created_by_evo_alg == chain_created_by_hand or chain_created_by_evo_alg != chain_created_by_hand and abs(
        roc_auc_chain_created_by_hand - roc_auc_chain_evo_alg) < 0.01
