import datetime
import os
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.random_composer import RandomGraphFactory, RandomSearchComposer, RandomSearchOptimizer
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.objective import Objective, DataSourceSplitter, PipelineObjectiveEvaluate
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.pipelines.test_pipeline_comparison import pipeline_first, pipeline_second


def to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.fixture(autouse=True)
def seed():
    random.seed(1)
    np.random.seed(1)


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/advanced_classification.csv'
    input_data = InputData.from_csv(os.path.join(test_file_path, file))
    input_data.idx = to_numerical(categorical_ids=input_data.idx)
    return input_data


def get_unimproveable_data():
    """ Create simple dataset which will not allow to improve metric values """
    features = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 101],
                         [1, 102], [1, 103], [1, 104], [1, 105]])
    target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    input_data = InputData(idx=np.arange(0, 10),
                           features=features,
                           target=target,
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    return input_data


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_random_composer(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    available_model_types = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.classification)

    objective = Objective(ClassificationMetricsEnum.ROCAUC)
    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types)
    optimiser = RandomSearchOptimizer(objective, RandomGraphFactory(req.primary, req.secondary), iter_num=2)
    random_composer = RandomSearchComposer(optimiser, composer_requirements=req)

    pipeline_random_composed = random_composer.compose_pipeline(data=dataset_to_compose)
    pipeline_random_composed.fit_from_scratch(input_data=dataset_to_compose)
    predicted_random_composed = pipeline_random_composed.predict(dataset_to_validate)

    roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_random_composed.predict)

    assert roc_on_valid_random_composed > 0.6


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_build_pipeline_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    task = Task(TaskTypesEnum.classification)
    available_model_types = OperationTypesRepository().suitable_operation(
        task_type=task.task_type)

    metric_function = ClassificationMetricsEnum.ROCAUC

    req = PipelineComposerRequirements(primary=available_model_types,
                                       secondary=available_model_types,
                                       num_of_generations=1)
    params = GPGraphOptimizerParameters(pop_size=2)

    builder = ComposerBuilder(task).with_requirements(req).with_optimizer_params(params).with_metrics(metric_function)
    gp_composer = builder.build()
    pipeline_gp_composed = gp_composer.compose_pipeline(data=dataset_to_compose)

    pipeline_gp_composed.fit_from_scratch(input_data=dataset_to_compose)
    predicted_gp_composed = pipeline_gp_composed.predict(dataset_to_validate)

    roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                       y_score=predicted_gp_composed.predict)

    assert roc_on_valid_gp_composed > 0.6


def baseline_pipeline():
    pipeline = Pipeline()
    last_node = SecondaryNode(operation_type='rf',
                              nodes_from=[])
    for requirement_model in ['knn', 'logit']:
        new_node = PrimaryNode(requirement_model)
        pipeline.add_node(new_node)
        last_node.nodes_from.append(new_node)
    pipeline.add_node(last_node)

    return pipeline


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_composition_time(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    task = Task(TaskTypesEnum.classification)
    models_impl = ['mlp', 'knn']
    metric_function = ClassificationMetricsEnum.ROCAUC

    req_terminated_evolution = PipelineComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        num_of_generations=5,
        timeout=datetime.timedelta(minutes=0.000001))
    params = GPGraphOptimizerParameters(pop_size=2)

    builder = ComposerBuilder(task) \
        .with_history() \
        .with_requirements(req_terminated_evolution) \
        .with_optimizer_params(params) \
        .with_metrics(metric_function)

    gp_composer_terminated_evolution = builder.build()

    _ = gp_composer_terminated_evolution.compose_pipeline(data=data)

    req_completed_evolution = PipelineComposerRequirements(
        primary=models_impl,
        secondary=models_impl,
        num_of_generations=2
    )
    params = GPGraphOptimizerParameters(pop_size=2)

    builder = ComposerBuilder(task) \
        .with_history() \
        .with_requirements(req_completed_evolution) \
        .with_optimizer_params(params) \
        .with_metrics(metric_function)
    gp_composer_completed_evolution = builder.build()

    _ = gp_composer_completed_evolution.compose_pipeline(data=data)

    assert len(gp_composer_terminated_evolution.history.individuals) == 1  # only the initial randomized population
    assert len(gp_composer_completed_evolution.history.individuals) == 3


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_parameter_free_composer_build_pipeline_correct(data_fixture, request):
    """ Checks that when a metric stagnates, the number of individuals in the population increases """
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    available_model_types = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.classification)

    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       max_arity=2, max_depth=2, num_of_generations=3)
    params = GPGraphOptimizerParameters(pop_size=2, genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free)

    gp_composer = ComposerBuilder(task=Task(TaskTypesEnum.classification)) \
        .with_history() \
        .with_optimizer_params(params) \
        .with_requirements(req) \
        .with_metrics(ClassificationMetricsEnum.ROCAUC) \
        .build()
    pipeline_gp_composed = gp_composer.compose_pipeline(data=dataset_to_compose)

    pipeline_gp_composed.fit_from_scratch(input_data=dataset_to_compose)
    predicted_gp_composed = pipeline_gp_composed.predict(dataset_to_validate)

    roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                       y_score=predicted_gp_composed.predict)

    all_individuals = len(gp_composer.history.individuals)
    population_len = sum([len(history) for history in gp_composer.history.individuals]) / all_individuals

    assert population_len != len(gp_composer.history.individuals[0])
    assert roc_on_valid_gp_composed > 0.6


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_multi_objective_composer(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    task_type = TaskTypesEnum.classification
    available_model_types = OperationTypesRepository().suitable_operation(task_type=task_type)
    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       max_arity=2, max_depth=2, num_of_generations=1)
    params = GPGraphOptimizerParameters(pop_size=2, genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                        selection_types=[SelectionTypesEnum.spea2])

    composer = ComposerBuilder(task=Task(task_type))\
        .with_requirements(req)\
        .with_metrics((ClassificationMetricsEnum.ROCAUC, ComplexityMetricsEnum.node_num))\
        .with_optimizer_params(params)\
        .build()
    pipelines_evo_composed = composer.compose_pipeline(data=dataset_to_compose)
    pipelines_roc_auc = []

    assert type(pipelines_evo_composed) is list
    assert len(composer.optimizer.objective.metrics) > 1
    assert composer.optimizer.objective.is_multi_objective

    for pipeline_evo_composed in pipelines_evo_composed:
        pipeline_evo_composed.fit_from_scratch(input_data=dataset_to_compose)
        predicted_gp_composed = pipeline_evo_composed.predict(dataset_to_validate)

        roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_gp_composed.predict)

        pipelines_roc_auc.append(roc_on_valid_gp_composed)

    assert all([roc_auc > 0.6 for roc_auc in pipelines_roc_auc])


def dummy_quality_metric(*args, **kwargs):
    return 1.0  # stagnating


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_with_adaptive_depth(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    available_model_types = ['rf', 'knn']

    quality_metric = dummy_quality_metric
    max_depth = 5
    num_gen = 3
    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       start_depth=2, max_depth=max_depth, num_of_generations=num_gen)
    params = GPGraphOptimizerParameters(adaptive_depth=True,
                                        adaptive_depth_max_stagnation=num_gen - 1,
                                        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state)
    composer = ComposerBuilder(task=Task(TaskTypesEnum.classification)) \
        .with_history() \
        .with_requirements(req) \
        .with_optimizer_params(params) \
        .with_metrics(quality_metric) \
        .build() \

    composer.compose_pipeline(data=dataset_to_compose)

    generations = composer.history.individuals
    current_depth = composer.optimizer.requirements.max_depth
    assert req.start_depth <= current_depth < max_depth, f"max depth couldn't have been reached in {num_gen}"
    assert all(ind.graph.depth < max_depth for ind in generations[-1]), "last generation is too deep"


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_evaluation_saving_info_from_process(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    quality_metric = ClassificationMetricsEnum.ROCAUC

    data_source = DataSourceSplitter().build(data)
    objective_evaluator = PipelineObjectiveEvaluate(Objective(quality_metric), data_source,
                                                    pipelines_cache=OperationsCache())

    objective_evaluator(pipeline_first())
    global_cache_len_before = len(objective_evaluator._pipelines_cache)

    assert global_cache_len_before > 0

    # evaluate additional pipeline to see that cache changes
    new_pipeline = pipeline_second()
    objective_evaluator(new_pipeline)
    global_cache_len_after = len(objective_evaluator._pipelines_cache)

    assert global_cache_len_before < global_cache_len_after
    assert new_pipeline.computation_time is not None


def test_gp_composer_builder_default_params_correct():
    task = Task(TaskTypesEnum.regression)
    builder = ComposerBuilder(task=task)

    # Initialise default parameters
    composer_with_default_params = builder.build()

    # Get default available operations for regression task
    primary_operations = composer_with_default_params.composer_requirements.primary

    # Data operations and models must be in this default primary operations list
    assert 'ridge' in primary_operations
    assert 'scaling' in primary_operations


@pytest.mark.parametrize('max_depth', [1, 3, 5])
def test_gp_composer_random_graph_generation_looping(max_depth):
    """ Test checks random_graph valid generation without freezing in loop of creation. """
    task = Task(TaskTypesEnum.regression)

    operations = get_operations_for_task(task, mode='model')
    primary_operations = operations[:len(operations) // 2]
    secondary_operations = operations[len(operations) // 2:]
    requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations,
        timeout=datetime.timedelta(seconds=300),
        max_pipeline_fit_time=None,
        max_depth=max_depth,
        max_arity=2,
        cv_folds=None,
        num_of_generations=5,
    )

    params = get_pipeline_generation_params(requirements=requirements, task=task)

    graphs = [random_graph(params, requirements, max_depth=None) for _ in range(4)]
    for graph in graphs:
        for node in graph.nodes:
            if node.nodes_from:
                assert node.content['name'] in requirements.secondary
            else:
                assert node.content['name'] in requirements.primary
        assert params.verifier(graph) is True
        assert graph.depth <= requirements.max_depth


def test_gp_composer_early_stopping():
    """ Test checks early stopping criteria """
    train_data = get_unimproveable_data()
    time_limit = datetime.timedelta(minutes=10)
    start = datetime.datetime.now()
    model = Fedot(problem='classification', timeout=1000,
                  early_stopping_generations=1,
                  pop_size=2,
                  with_tuning=False,
                  preset='fast_train')
    model.fit(train_data)
    spent_time = datetime.datetime.now() - start

    assert spent_time < time_limit
