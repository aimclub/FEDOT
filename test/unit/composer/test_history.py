import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from fedot.api.main import Fedot
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.graph import Graph
from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.data.data import InputData
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.fitness import SingleObjFitness
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual, ParentOperator
from fedot.core.optimisers.gp_comp.operators.crossover import crossover, CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.operations.model import Model
from fedot.core.optimisers.objective.data_objective_builder import DataObjectiveBuilder
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
    node_third = SecondaryNode('bernb',  nodes_from=[node_second])
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
    operator_for_history = ParentOperator(operator_type='mutation',
                                          operator_name=str(mutation_type),
                                          parent_individuals=[ind])

    assert operator_for_history.parent_individuals[0] == ind
    assert operator_for_history.operator_type == 'mutation'


def test_ancestor_for_mutation():
    pipeline = Pipeline(PrimaryNode('linear'))
    adapter = PipelineAdapter()
    parent_ind = Individual(adapter.adapt(pipeline))

    available_operations = ['linear']
    composer_requirements = PipelineComposerRequirements(primary=available_operations,
                                                         secondary=available_operations, mutation_prob=1)

    graph_params = get_pipeline_generation_params(requirements=composer_requirements,
                                                  rules_for_constraint=DEFAULT_DAG_RULES)

    mutation_result = mutation(types=[MutationTypesEnum.simple],
                               params=graph_params,
                               ind=parent_ind,
                               requirements=composer_requirements, max_depth=2)
    mutation_result = mutation(types=[MutationTypesEnum.simple], params=graph_params, ind=parent_ind,
                               requirements=composer_requirements, max_depth=2)

    assert len(mutation_result.parent_operators) > 0
    assert mutation_result.parent_operators[-1].operator_type == 'mutation'
    assert len(mutation_result.parent_operators[-1].parent_individuals) == 1
    assert mutation_result.parent_operators[-1].parent_individuals[0].uid == parent_ind.uid


def test_ancestor_for_crossover():
    adapter = PipelineAdapter()
    parent_ind_first = Individual(adapter.adapt(Pipeline(PrimaryNode('linear'))))
    parent_ind_second = Individual(adapter.adapt(Pipeline(PrimaryNode('ridge'))))

    graph_params = get_pipeline_generation_params(rules_for_constraint=DEFAULT_DAG_RULES)

    crossover_results = crossover([CrossoverTypesEnum.subtree], parent_ind_first, parent_ind_second, max_depth=3,
                                  crossover_prob=1, params=graph_params)

    for crossover_result in crossover_results:
        assert len(crossover_result.parent_operators) > 0
        assert crossover_result.parent_operators[-1].operator_type == 'crossover'
        assert len(crossover_result.parent_operators[-1].parent_individuals) == 2
        assert crossover_result.parent_operators[-1].parent_individuals[0].uid == parent_ind_first.uid
        assert crossover_result.parent_operators[-1].parent_individuals[1].uid == parent_ind_second.uid


def test_operators_in_history():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')

    num_of_gens = 2
    auto_model = Fedot(problem='classification', seed=42,
                       timeout=None,
                       num_of_generations=num_of_gens, pop_size=3,
                       preset='fast_train')
    auto_model.fit(features=file_path_train, target='Y')

    assert auto_model.history is not None
    assert len(auto_model.history.individuals) == num_of_gens + 1  # num_of_gens + initial assumption

    # test history dumps
    dumped_history = auto_model.history.save()

    assert dumped_history is not None


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

    objective_builder = DataObjectiveBuilder(Objective(metrics))
    objective_eval = objective_builder.build(input_data)
    dispatcher = MultiprocessingDispatcher(graph_gen_params.adapter)
    dispatcher.set_evaluation_callback(objective_eval.evaluate_intermediate_metrics)
    evaluate = dispatcher.dispatch(objective_eval)

    population = [Individual(graph_gen_params.adapter.adapt(pipeline))]
    evaluated_pipeline = evaluate(population)[0].graph
    restored_pipeline = graph_gen_params.adapter.restore(evaluated_pipeline)

    assert_intermediate_metrics(restored_pipeline)


@pytest.mark.parametrize("cv_generator, data",
                         [(partial(tabular_cv_generator, folds=3),
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
    test_history_path = Path(fedot_project_root(), 'test', 'data', 'test_history.json')
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
    # Assert that fitness and objective are valid
    assert all(isinstance(ind.fitness, SingleObjFitness) for gen in history.individuals for ind in gen)
    assert isinstance(history._objective, Objective)
