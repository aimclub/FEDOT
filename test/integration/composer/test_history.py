from itertools import chain
from pathlib import Path

import numpy as np
import pytest

from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.dag.graph import Graph
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory

from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.operations.model import Model
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, \
    RegressionMetricsEnum, MetricType
from fedot.core.utils import fedot_project_root
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.validation.test_table_cv import get_classification_data


def scaling_logit_rf_pipeline():
    node_first = PipelineNode('scaling')
    node_second = PipelineNode('logit', nodes_from=[node_first])
    node_third = PipelineNode('bernb', nodes_from=[node_second])
    return Pipeline(node_third)


def lagged_ridge_rfr_pipeline():
    node_first = PipelineNode('lagged')
    node_second = PipelineNode('ridge', nodes_from=[node_first])
    node_third = PipelineNode('rfr', nodes_from=[node_second])
    return Pipeline(node_third)


def _test_individuals_in_history(history: OptHistory):
    uids = set()
    ids = set()
    for ind in chain(*history.individuals):
        # All individuals in `history.individuals` must have a native generation.
        assert ind.has_native_generation
        assert ind.fitness
        if ind.native_generation == 0:
            continue
        # All individuals must have parents, except for the initial assumptions.
        assert ind.parents
        assert ind.parents_from_prev_generation
        # The first of `operators_from_prev_generation` must point to `parents_from_prev_generation`.
        assert ind.parents_from_prev_generation == list(ind.operators_from_prev_generation[0].parent_individuals)
        # All parents are from previous generations
        assert all(p.native_generation < ind.native_generation for p in ind.parents_from_prev_generation)

        uids.add(ind.uid)
        ids.add(id(ind))
        for parent_operator in ind.operators_from_prev_generation:
            uids.update({i.uid for i in parent_operator.parent_individuals})
            ids.update({id(i) for i in parent_operator.parent_individuals})

    assert len(uids) == len(ids)


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_newly_generated_history(n_jobs: int):
    file_path_train = fedot_project_root().joinpath('test/data/simple_classification.csv')

    num_of_gens = 2
    auto_model = Fedot(problem='classification', seed=42,
                       timeout=None,
                       num_of_generations=num_of_gens, pop_size=3,
                       preset='fast_train',
                       n_jobs=n_jobs,
                       with_tuning=False)
    auto_model.fit(features=file_path_train, target='Y')

    history = auto_model.history

    assert history is not None
    assert len(history.individuals) == num_of_gens + 2  # initial_assumptions + num_of_gens + final_choices
    assert len(history.archive_history) == num_of_gens + 2  # initial_assumptions + num_of_gens + final_choices
    assert len(history.initial_assumptions) >= 2
    assert len(history.final_choices) == 1
    assert isinstance(history.tuning_result, Graph)
    _test_individuals_in_history(history)
    # Test history dumps
    dumped_history_json = history.save()
    loaded_history = OptHistory.load(dumped_history_json)
    assert dumped_history_json is not None
    assert dumped_history_json == loaded_history.save(), 'The history is not equal to itself after reloading!'
    _test_individuals_in_history(loaded_history)


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

    validation_blocks = 1 if input_data.task.task_type is TaskTypesEnum.ts_forecasting else None
    data_source = DataSourceSplitter(validation_blocks=validation_blocks).build(input_data)
    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metrics),
                                               data_source,
                                               validation_blocks=validation_blocks)
    dispatcher = MultiprocessingDispatcher(graph_gen_params.adapter)
    dispatcher.set_graph_evaluation_callback(objective_eval.evaluate_intermediate_metrics)
    evaluate = dispatcher.dispatch(objective_eval)

    population = [Individual(graph_gen_params.adapter.adapt(pipeline))]
    evaluated_pipeline = evaluate(population)[0].graph
    restored_pipeline = graph_gen_params.adapter.restore(evaluated_pipeline)

    assert_intermediate_metrics(restored_pipeline)


def test_history_backward_compatibility():
    from fedot.core.optimisers.objective import init_backward_serialize_compat
    init_backward_serialize_compat()

    test_history_path = Path(fedot_project_root(), 'test', 'data', 'fast_train_classification_history.json')
    history = OptHistory.load(test_history_path)
    # Pre-computing properties
    all_historical_fitness = history.all_historical_fitness
    historical_fitness = history.historical_fitness
    # Assert presence of necessary fields after deserialization of old history
    assert hasattr(history, 'objective')
    # Assert that properties are not empty
    assert all_historical_fitness
    assert historical_fitness
    # Assert that all fitness properties are valid.
    assert len(history.individuals) == len(historical_fitness)
    assert np.all(len(generation) == len(gen_fitness)
                  for generation, gen_fitness in zip(history.individuals, historical_fitness))
    assert np.all(np.equal([ind.fitness.value for ind in chain(*history.individuals)], all_historical_fitness))
    # Assert that fitness, graph, parent_individuals, and objective are valid
    assert all(isinstance(ind.fitness, SingleObjFitness) for ind in chain(*history.individuals))
    assert all(ind.graph.nodes for ind in chain(*history.individuals))
    assert all(isinstance(parent_ind, Individual)
               for ind in chain(*history.individuals)
               for parent_op in ind.operators_from_prev_generation
               for parent_ind in parent_op.parent_individuals)
    _test_individuals_in_history(history)
