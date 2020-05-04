import datetime
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
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository,
    ModelTypesIdsEnum
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


def test_dummy_composer_hierarchical_chain_build_correct():
    composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    empty_data = InputData(idx=np.zeros(1), features=np.zeros(1), target=np.zeros(1),
                           task_type=MachineLearningTasksEnum.classification)
    primary = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost]
    secondary = [ModelTypesIdsEnum.logit]
    composer_requirements = ComposerRequirements(primary=primary,
                                                 secondary=secondary)
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


def test_dummy_composer_flat_chain_build_correct():
    composer = DummyComposer(DummyChainTypeEnum.flat)
    empty_data = InputData(idx=np.zeros(1), features=np.zeros(1), target=np.zeros(1),
                           task_type=MachineLearningTasksEnum.classification)
    primary = [ModelTypesIdsEnum.logit]
    secondary = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost]
    composer_requirements = ComposerRequirements(primary=primary,
                                                 secondary=secondary)
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


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_random_composer(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    random_composer = RandomSearchComposer(iter_num=1)
    req = ComposerRequirements(primary=available_model_types, secondary=available_model_types)
    chain_random_composed = random_composer.compose_chain(data=dataset_to_compose,
                                                          initial_chain=None,
                                                          composer_requirements=req,
                                                          metrics=metric_function)
    chain_random_composed.fit_from_scratch(input_data=dataset_to_compose)

    predicted_random_composed = chain_random_composed.predict(dataset_to_validate)

    roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_random_composed.predict)

    assert roc_on_valid_random_composed > 0.6


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_build_chain_correct(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    models_repo = ModelTypesRepository()

    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    gp_composer = GPComposer()
    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 max_arity=2, max_depth=2, pop_size=2, num_of_generations=1,
                                 crossover_prob=0.4, mutation_prob=0.5)
    chain_gp_composed = gp_composer.compose_chain(data=dataset_to_compose,
                                                  initial_chain=None,
                                                  composer_requirements=req,
                                                  metrics=metric_function)

    chain_gp_composed.fit_from_scratch(input_data=dataset_to_compose)
    predicted_gp_composed = chain_gp_composed.predict(dataset_to_validate)

    roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                       y_score=predicted_gp_composed.predict)

    assert roc_on_valid_gp_composed > 0.6


def baseline_chain():
    chain = Chain()
    last_node = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost,
                                             nodes_from=[])
    for requirement_model in [ModelTypesIdsEnum.knn, ModelTypesIdsEnum.logit]:
        new_node = NodeGenerator.primary_node(requirement_model)
        chain.add_node(new_node)
        last_node.nodes_from.append(new_node)
    chain.add_node(last_node)

    return chain


@pytest.mark.skip('Refactor this test && move it to benchmarks?')
@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_quality(data_fixture, request):
    random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    baseline = baseline_chain()
    baseline.fit_from_scratch(input_data=dataset_to_compose)

    predict_baseline = baseline.predict(dataset_to_validate).predict
    dataset_to_compose.target = np.array([int(round(i)) for i in predict_baseline])

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=3, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.8)

    # Create GP-based composer
    composer = GPComposer()
    composed_chain = composer.compose_chain(data=dataset_to_compose,
                                            initial_chain=None,
                                            composer_requirements=composer_requirements,
                                            metrics=metric_function)
    composed_chain.fit_from_scratch(input_data=dataset_to_compose)

    predict_composed = composed_chain.predict(dataset_to_validate).predict

    roc_auc_chain_created_by_hand = roc_auc(y_true=dataset_to_validate.target, y_score=predict_baseline)
    roc_auc_chain_evo_alg = roc_auc(y_true=dataset_to_validate.target, y_score=predict_composed)
    print("model created by hand prediction:", roc_auc_chain_created_by_hand)
    print("gp composed model prediction:", roc_auc_chain_evo_alg)

    assert composed_chain == baseline or composed_chain != baseline and abs(
        roc_auc_chain_created_by_hand - roc_auc_chain_evo_alg) < 0.01


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_composition_time(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)

    models_impl = [ModelTypesIdsEnum.mlp, ModelTypesIdsEnum.knn]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    gp_composer_terminated_evolution = GPComposer()

    req_terminated_evolution = GPComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=5, crossover_prob=0.9,
        mutation_prob=0.9, max_lead_time=datetime.timedelta(minutes=0.01))

    chain_terminated_evolution = gp_composer_terminated_evolution.compose_chain(data=data,
                                                                                initial_chain=None,
                                                                                composer_requirements=req_terminated_evolution,
                                                                                metrics=metric_function)

    gp_composer_completed_evolution = GPComposer()

    req_completed_evolution = GPComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=2, crossover_prob=0.4,
        mutation_prob=0.5)

    chain_completed_evolution = gp_composer_completed_evolution.compose_chain(data=data,
                                                                              initial_chain=None,
                                                                              composer_requirements=req_completed_evolution,
                                                                              metrics=metric_function)

    assert len(gp_composer_terminated_evolution.history) == 4
    assert len(gp_composer_completed_evolution.history) == 4
