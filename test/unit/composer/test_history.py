import os
from functools import partial
from itertools import chain
from pathlib import Path

import numpy as np
import pytest

from fedot.api.main import Fedot
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.dag.graph import Graph
from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.data.data import InputData
from fedot.core.operations.model import Model
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.fitness import SingleObjFitness
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual, ParentOperator
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, Crossover
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, Mutation
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, \
    RegressionMetricsEnum, MetricType
from fedot.core.utils import fedot_project_root
from fedot.core.validation.split import tabular_cv_generator, ts_cv_generator
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.validation.test_table_cv import get_classification_data


def scaling_logit_rf_pipeline():
    node_first = PrimaryNode('scaling')
    node_second = SecondaryNode('logit', nodes_from=[node_first])
    node_third = SecondaryNode('bernb', nodes_from=[node_second])
    return Pipeline(node_third)


def lagged_ridge_rfr_pipeline():
    node_first = PrimaryNode('lagged')
    node_second = SecondaryNode('ridge', nodes_from=[node_first])
    node_third = SecondaryNode('rfr', nodes_from=[node_second])
    return Pipeline(node_third)


def test_parent_operator():
    pipeline = Pipeline(PrimaryNode('linear'))
    adapter = PipelineAdapter()
    ind = Individual(adapter.adapt(pipeline))
    mutation_type = MutationTypesEnum.simple
    operator_for_history = ParentOperator(type='mutation',
                                          operators=(str(mutation_type),),
                                          parent_individuals=(ind,))

    assert operator_for_history.parent_individuals[0] == ind
    assert operator_for_history.type == 'mutation'


def test_ancestor_for_mutation():
    pipeline = Pipeline(PrimaryNode('linear'))
    adapter = PipelineAdapter()
    parent_ind = Individual(adapter.adapt(pipeline))

    available_operations = ['linear']
    composer_requirements = PipelineComposerRequirements(primary=available_operations,
                                                         secondary=available_operations, mutation_prob=1,
                                                         max_depth=2)

    graph_params = get_pipeline_generation_params(requirements=composer_requirements,
                                                  rules_for_constraint=DEFAULT_DAG_RULES)
    mutation = Mutation(mutation_types=[MutationTypesEnum.simple], graph_generation_params=graph_params,
                        requirements=composer_requirements)

    mutation_result = mutation(parent_ind)

    assert mutation_result.parent_operator
    assert mutation_result.parent_operator.type == 'mutation'
    assert len(mutation_result.parent_operator.parent_individuals) == 1
    assert mutation_result.parent_operator.parent_individuals[0].uid == parent_ind.uid


def test_ancestor_for_crossover():
    adapter = PipelineAdapter()
    parent_ind_first = Individual(adapter.adapt(Pipeline(PrimaryNode('linear'))))
    parent_ind_second = Individual(adapter.adapt(Pipeline(PrimaryNode('ridge'))))

    graph_params = get_pipeline_generation_params(rules_for_constraint=DEFAULT_DAG_RULES)
    composer_requirements = PipelineComposerRequirements(max_depth=3, crossover_prob=1)
    crossover = Crossover([CrossoverTypesEnum.subtree], composer_requirements, graph_params)
    crossover_results = crossover([parent_ind_first, parent_ind_second])

    for crossover_result in crossover_results:
        assert crossover_result.parent_operator
        assert crossover_result.parent_operator.type == 'crossover'
        assert len(crossover_result.parent_operator.parent_individuals) == 2
        assert crossover_result.parent_operator.parent_individuals[0].uid == parent_ind_first.uid
        assert crossover_result.parent_operator.parent_individuals[1].uid == parent_ind_second.uid


def test_newly_generated_history():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')

    num_of_gens = 2
    auto_model = Fedot(problem='classification', seed=42,
                       timeout=None,
                       num_of_generations=num_of_gens, pop_size=3,
                       preset='fast_train')
    auto_model.fit(features=file_path_train, target='Y')

    history = auto_model.history

    assert auto_model.history is not None
    assert len(history.individuals) == num_of_gens + 1  # num_of_gens + initial assumption
    assert len(history.archive_history) == num_of_gens + 1  # num_of_gens + initial assumption
    # Test individuals with the same uid are not copied
    individuals = history.individuals
    assert len({id(i): i for i in chain(*individuals)}) == len({i.uid: i for i in chain(*individuals)})
    # Test history dumps
    dumped_history = history.save()
    loaded_history = OptHistory.load(dumped_history).save()
    assert dumped_history is not None
    assert dumped_history == loaded_history, 'The history is not equal to itself after reloading!'


def assert_intermediate_metrics(pipeline: Graph):
    seen_metrics = []
    for node in pipeline.nodes:
        if isinstance(node.operation, Model):
            assert node.metadata.metric is not None
            assert node.metadata.metric not in seen_metrics
            seen_metrics.append(node.metadata.metric)
        else:
            assert node.metadata.metric is None


@pytest.mark.parametrize("pipeline, input_data, metric",
                         [(scaling_logit_rf_pipeline(),
                           get_classification_data(),
                           ClassificationMetricsEnum.ROCAUC),
                          (lagged_ridge_rfr_pipeline(),
                           get_ts_data()[0],
                           RegressionMetricsEnum.RMSE),
                          ])
def test_collect_intermediate_metric(pipeline: Pipeline, input_data: InputData, metric: MetricType):
    graph_gen_params = get_pipeline_generation_params()
    metrics = [metric]

    data_source = DataSourceSplitter().build(input_data)
    objective_eval = PipelineObjectiveEvaluate(Objective(metrics), data_source)
    dispatcher = MultiprocessingDispatcher(graph_gen_params.adapter)
    dispatcher.set_evaluation_callback(objective_eval.evaluate_intermediate_metrics)
    evaluate = dispatcher.dispatch(objective_eval)

    population = [Individual(graph_gen_params.adapter.adapt(pipeline))]
    evaluated_pipeline = evaluate(population)[0].graph
    restored_pipeline = graph_gen_params.adapter.restore(evaluated_pipeline)

    assert_intermediate_metrics(restored_pipeline)


@pytest.mark.parametrize("cv_generator, data",
                         [(partial(tabular_cv_generator, folds=5),
                           get_classification_data()),
                          (partial(ts_cv_generator, folds=3, validation_blocks=2),
                           get_ts_data()[0])])
def test_cv_generator_works_stable(cv_generator, data):
    """ Test if ts cv generator works stable (always return same folds) """
    idx_first = []
    idx_second = []
    for row in cv_generator(data=data):
        idx_first.append(row[1].idx)
    for row in cv_generator(data=data):
        idx_second.append(row[1].idx)

    for i in range(len(idx_first)):
        assert np.all(idx_first[i] == idx_second[i])


def test_history_backward_compatibility():
    test_history_path = Path(fedot_project_root(), 'test', 'data', 'fast_train_classification_history.json')
    history = OptHistory.load(test_history_path)
    # Pre-computing properties
    all_historical_fitness = history.all_historical_fitness
    historical_fitness = history.historical_fitness
    historical_pipelines = history.historical_pipelines
    # Assert that properties are not empty
    assert all_historical_fitness
    assert historical_fitness
    assert historical_pipelines
    # Assert that all history pipelines have fitness
    assert len(historical_pipelines) == len(all_historical_fitness)
    assert np.shape(history.individuals) == np.shape(historical_fitness)
    # Assert that fitness, parent_individuals, and objective are valid
    assert all(isinstance(ind.fitness, SingleObjFitness) for ind in chain(*history.individuals))
    assert all(isinstance(parent_individual, Individual)  # TODO: Check parents recursively until prev gen. Add property
               for ind in chain(*history.individuals) if ind.parent_operator is not None
               for parent_individual in ind.parent_operator.parent_individuals)
    assert isinstance(history._objective, Objective)


def test_history_correct_serialization():
    test_history_path = Path(fedot_project_root(), 'test', 'data', 'fast_train_classification_history.json')

    history = OptHistory.load(test_history_path)
    dumped_history = history.save()
    reloaded_history = OptHistory.load(dumped_history)

    assert history.individuals == reloaded_history.individuals
    assert dumped_history == reloaded_history.save(), 'The history is not equal to itself after reloading!'
