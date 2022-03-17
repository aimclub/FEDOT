import datetime
import os
import random
import shelve

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.composer import ComposerRequirements
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements, \
    sample_split_ratio_for_tasks
from fedot.core.composer.random_composer import RandomSearchComposer
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum, \
    MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.pipelines.test_pipeline_comparison import pipeline_first


def to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


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
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    available_model_types, _ = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.classification)

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    random_composer = RandomSearchComposer(iter_num=1)
    req = ComposerRequirements(primary=available_model_types, secondary=available_model_types)
    pipeline_random_composed = random_composer.compose_pipeline(data=dataset_to_compose,
                                                                composer_requirements=req,
                                                                metrics=metric_function)
    pipeline_random_composed.fit_from_scratch(input_data=dataset_to_compose)

    predicted_random_composed = pipeline_random_composed.predict(dataset_to_validate)

    roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_random_composed.predict)

    assert roc_on_valid_random_composed > 0.6


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_build_pipeline_correct(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    task = Task(TaskTypesEnum.classification)
    available_model_types, _ = OperationTypesRepository().suitable_operation(
        task_type=task.task_type)

    metric_function = ClassificationMetricsEnum.ROCAUC

    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       max_arity=2, max_depth=2, pop_size=2, num_of_generations=1,
                                       crossover_prob=0.4, mutation_prob=0.5)

    builder = ComposerBuilder(task).with_requirements(req).with_metrics(metric_function)
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
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    task = Task(TaskTypesEnum.classification)
    models_impl = ['mlp', 'knn']
    metric_function = ClassificationMetricsEnum.ROCAUC

    req_terminated_evolution = PipelineComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=5, crossover_prob=0.9,
        mutation_prob=0.9, timeout=datetime.timedelta(minutes=0.000001))

    builder = ComposerBuilder(task).with_requirements(req_terminated_evolution).with_metrics(metric_function)

    gp_composer_terminated_evolution = builder.build()

    _ = gp_composer_terminated_evolution.compose_pipeline(data=data)

    req_completed_evolution = PipelineComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=2, crossover_prob=0.4,
        mutation_prob=0.5)

    builder = ComposerBuilder(task).with_requirements(req_completed_evolution).with_metrics(metric_function)
    gp_composer_completed_evolution = builder.build()

    _ = gp_composer_completed_evolution.compose_pipeline(data=data)

    assert len(gp_composer_terminated_evolution.history.individuals) == 1
    assert len(gp_composer_completed_evolution.history.individuals) == 2


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_parameter_free_composer_build_pipeline_correct(data_fixture, request):
    """ Checks that when a metric stagnates, the number of individuals in the population increases """
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    available_model_types, _ = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.classification)

    metric_function = ClassificationMetricsEnum.ROCAUC

    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       max_arity=2, max_depth=2, pop_size=2, num_of_generations=3,
                                       crossover_prob=0.4, mutation_prob=0.5)

    opt_params = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free)
    builder = ComposerBuilder(task=Task(TaskTypesEnum.classification)).with_requirements(req).with_metrics(
        metric_function).with_optimiser(parameters=opt_params)
    gp_composer = builder.build()
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
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    available_model_types, _ = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.classification)
    quality_metric = ClassificationMetricsEnum.ROCAUC
    complexity_metric = ComplexityMetricsEnum.node_num
    metrics = [quality_metric, complexity_metric]
    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       max_arity=2, max_depth=2, pop_size=2, num_of_generations=1,
                                       crossover_prob=0.4, mutation_prob=0.5)
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPGraphOptimiserParameters(genetic_scheme_type=scheme_type,
                                                      selection_types=[SelectionTypesEnum.nsga2])
    builder = ComposerBuilder(task=Task(TaskTypesEnum.classification)).with_requirements(req).with_metrics(
        metrics).with_optimiser(parameters=optimiser_parameters)
    composer = builder.build()
    pipelines_evo_composed = composer.compose_pipeline(data=dataset_to_compose)
    pipelines_roc_auc = []
    for pipeline_evo_composed in pipelines_evo_composed:
        pipeline_evo_composed.fit_from_scratch(input_data=dataset_to_compose)
        predicted_gp_composed = pipeline_evo_composed.predict(dataset_to_validate)

        roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_gp_composed.predict)

        pipelines_roc_auc.append(roc_on_valid_gp_composed)

    assert type(composer.metrics) is list and len(composer.metrics) > 1
    assert type(pipelines_evo_composed) is list
    assert composer.optimiser.parameters.multi_objective
    assert all([roc_auc > 0.6 for roc_auc in pipelines_roc_auc])


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_with_start_depth(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    available_model_types = ['rf', 'knn']
    quality_metric = ClassificationMetricsEnum.ROCAUC
    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       max_arity=2, max_depth=5, pop_size=5, num_of_generations=1,
                                       crossover_prob=0.4, mutation_prob=0.5, start_depth=2)
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPGraphOptimiserParameters(genetic_scheme_type=scheme_type)
    builder = ComposerBuilder(task=Task(TaskTypesEnum.classification)).with_requirements(req).with_metrics(
        quality_metric).with_optimiser(parameters=optimiser_parameters)
    composer = builder.build()
    composer.compose_pipeline(data=dataset_to_compose,
                              is_visualise=True)
    assert all([ind.graph.depth <= 3 for ind in composer.history.individuals[0]])
    assert composer.optimiser.max_depth == 5


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_saving_info_from_process(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    available_model_types = ['rf', 'knn']
    quality_metric = ClassificationMetricsEnum.ROCAUC
    req = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                       max_arity=2, max_depth=4, pop_size=3, num_of_generations=1,
                                       crossover_prob=0.4, mutation_prob=0.5, start_depth=2
                                       )
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPGraphOptimiserParameters(genetic_scheme_type=scheme_type)
    builder = ComposerBuilder(task=Task(TaskTypesEnum.classification)).with_requirements(req).with_metrics(
        quality_metric).with_optimiser(parameters=optimiser_parameters).with_cache()
    composer = builder.build()
    train_data, test_data = train_test_data_setup(data,
                                                  sample_split_ratio_for_tasks[data.task.task_type])
    composer.compose_pipeline(data=dataset_to_compose, is_visualise=True)
    with shelve.open(composer.cache.db_path) as cache:
        global_cache_len_before = len(cache.dict)
    new_pipeline = pipeline_first()
    composer.composer_metric([quality_metric], dataset_to_compose, test_data, new_pipeline)
    with shelve.open(composer.cache.db_path) as cache:
        global_cache_len_after = len(cache.dict)
    assert global_cache_len_before < global_cache_len_after
    assert new_pipeline.computation_time is not None


def test_gp_composer_builder_default_params_correct():
    task = Task(TaskTypesEnum.regression)
    builder = ComposerBuilder(task=task)

    # Initialise default parameters
    builder.set_default_composer_params()
    composer_with_default_params = builder.build()

    # Get default available operations for regression task
    primary_operations = composer_with_default_params.composer_requirements.primary

    # Data operations and models must be in this default primary operations list
    assert 'ridge' in primary_operations
    assert 'scaling' in primary_operations


def test_gp_composer_random_graph_generation_looping():
    """ Test checks random_graph valid generation without freezing in loop of creation.
    """
    task = Task(TaskTypesEnum.regression)

    params = GraphGenerationParams(
        adapter=PipelineAdapter(),
        rules_for_constraint=None,
        advisor=PipelineChangeAdvisor(task=task)
    )

    requirements = PipelineComposerRequirements(
        primary=['simple_imputation'],
        secondary=['ridge', 'dtreg'],
        timeout=datetime.timedelta(seconds=300),
        max_pipeline_fit_time=None,
        max_depth=2,
        max_arity=2,
        cv_folds=None,
        advisor=PipelineChangeAdvisor(task=task),
        pop_size=10,
        num_of_generations=5,
        crossover_prob=0.8,
        mutation_prob=0.8,
        mutation_strength=MutationStrengthEnum.mean
    )

    graph = random_graph(params=params, requirements=requirements, max_depth=None)
    nodes_name = list(map(str, graph.nodes))

    for primary_node in requirements.primary:
        assert primary_node in nodes_name
        assert nodes_name.count(primary_node) == 1
    assert constraint_function(graph, params) is True


def test_gp_composer_early_stopping():
    """ Test checks early stopping criteria """
    train_data = get_unimproveable_data()
    time_limit = datetime.timedelta(minutes=10)
    start = datetime.datetime.now()
    model = Fedot(problem='classification', timeout=1000,
                  composer_params={'stopping_after_n_generation': 1,
                                   'pop_size': 2,
                                   'with_tuning': False},
                  preset='fast_train')
    model.fit(train_data)
    spent_time = datetime.datetime.now() - start

    assert spent_time < time_limit
