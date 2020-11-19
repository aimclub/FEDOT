import datetime
import os
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.chain import Chain
from fedot.core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from fedot.core.composer.gp_composer.fixed_structure_composer import FixedStructureComposer
from fedot.core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from fedot.core.composer.node import PrimaryNode, SecondaryNode
from fedot.core.composer.random_composer import RandomSearchComposer
from fedot.core.models.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.chain.test_chain_tuning import get_class_chain


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = '../data/advanced_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def test_dummy_composer_hierarchical_chain_build_correct():
    composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    empty_data = InputData(idx=np.zeros(1), features=np.zeros(1), target=np.zeros(1),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    primary = ['logit', 'xgboost']
    secondary = ['logit']

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
                           task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)

    primary = ['logit']
    secondary = ['logit', 'xgboost']

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

    available_model_types, _ = ModelTypesRepository().suitable_model(
        task_type=TaskTypesEnum.classification)

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
def test_fixed_structure_composer(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    available_model_types = ['logit', 'lda', 'knn']

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer = FixedStructureComposer()
    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 pop_size=2, num_of_generations=1,
                                 crossover_prob=0.4, mutation_prob=0.5,
                                 add_single_model_chains=False)

    reference_chain = get_class_chain()

    chain_composed = composer.compose_chain(data=dataset_to_compose,
                                            initial_chain=reference_chain,
                                            composer_requirements=req,
                                            metrics=metric_function)
    chain_composed.fit_from_scratch(input_data=dataset_to_compose)

    predicted_random_composed = chain_composed.predict(dataset_to_validate)

    roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_random_composed.predict)

    assert roc_on_valid_random_composed > 0.6
    assert chain_composed.depth == reference_chain.depth
    assert chain_composed.length == reference_chain.length


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_build_chain_correct(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    available_model_types, _ = ModelTypesRepository().suitable_model(
        task_type=TaskTypesEnum.classification)

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
    last_node = SecondaryNode(model_type='xgboost',
                              nodes_from=[])
    for requirement_model in ['knn', 'logit']:
        new_node = PrimaryNode(requirement_model)
        chain.add_node(new_node)
        last_node.nodes_from.append(new_node)
    chain.add_node(last_node)

    return chain


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_composition_time(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)

    models_impl = ['mlp', 'knn']
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    gp_composer_terminated_evolution = GPComposer()

    req_terminated_evolution = GPComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=5, crossover_prob=0.9,
        mutation_prob=0.9, max_lead_time=datetime.timedelta(minutes=0.000001))

    _ = gp_composer_terminated_evolution.compose_chain(data=data,
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

    _ = gp_composer_completed_evolution.compose_chain(data=data,
                                                      initial_chain=None,
                                                      composer_requirements=req_completed_evolution,
                                                      metrics=metric_function)

    assert len(gp_composer_terminated_evolution.history) == len(gp_composer_completed_evolution.history)
