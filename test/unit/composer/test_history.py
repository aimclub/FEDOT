import os
from functools import partial

import numpy as np
import pytest

from fedot.api.main import Fedot
from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import DEFAULT_DAG_RULES
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.operations.model import Model
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.evaluating import collect_intermediate_metric_for_nodes_cv
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.intermediate_metric import collect_intermediate_metric_for_nodes
from fedot.core.optimisers.gp_comp.operators.crossover import crossover, CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum, \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from fedot.core.validation.split import tabular_cv_generator, ts_cv_generator
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.validation.test_table_cv import get_classification_data


def rf_scaling_pipeline():
    node_first = PrimaryNode('scaling')
    node_second = SecondaryNode('rf', nodes_from=[node_first])
    return Pipeline(node_second)


def lagged_ridge_pipeline():
    node_first = PrimaryNode('lagged')
    node_second = SecondaryNode('ridge', nodes_from=[node_first])
    return Pipeline(node_second)


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

    graph_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                         advisor=PipelineChangeAdvisor(task=Task(TaskTypesEnum.regression)),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    available_operations = ['linear']
    composer_requirements = PipelineComposerRequirements(primary=available_operations,
                                                         secondary=available_operations, mutation_prob=1)

    mutation_result = mutation(types=[MutationTypesEnum.simple],
                               params=graph_params,
                               ind=parent_ind,
                               requirements=composer_requirements,
                               log=default_log(__name__), max_depth=2)

    assert len(mutation_result.parent_operators) > 0
    assert mutation_result.parent_operators[-1].operator_type == 'mutation'
    assert len(mutation_result.parent_operators[-1].parent_individuals) == 1
    assert mutation_result.parent_operators[-1].parent_individuals[0].uid == parent_ind.uid


def test_ancestor_for_crossover():
    adapter = PipelineAdapter()
    parent_ind_first = Individual(adapter.adapt(Pipeline(PrimaryNode('linear'))))
    parent_ind_second = Individual(adapter.adapt(Pipeline(PrimaryNode('ridge'))))

    graph_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                         advisor=PipelineChangeAdvisor(task=Task(TaskTypesEnum.regression)),
                                         rules_for_constraint=DEFAULT_DAG_RULES)

    crossover_results = crossover([CrossoverTypesEnum.subtree],
                                  parent_ind_first, parent_ind_second,
                                  params=graph_params, max_depth=3, log=default_log(__name__),
                                  crossover_prob=1)

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
                       composer_params={'num_of_generations': num_of_gens, 'pop_size': 3},
                       preset='fast_train')
    auto_model.fit(features=file_path_train, target='Y')

    assert auto_model.history is not None
    assert len(auto_model.history.individuals) == num_of_gens + 2  # initial assumptions and final model

    # test history dumps
    dumped_history = auto_model.history.save()

    assert dumped_history is not None


@pytest.mark.parametrize("pipeline, data, method",
                         [(rf_scaling_pipeline(),
                           get_classification_data()[0],
                           partial(collect_intermediate_metric_for_nodes_cv,
                                   cv_generator=partial(tabular_cv_generator, data=get_classification_data()[0],
                                                        folds=3),
                                   metric=MetricsRepository().metric_by_id(
                                       ClassificationMetricsEnum.ROCAUC))),
                          (lagged_ridge_pipeline(),
                           get_ts_data()[0],
                           partial(collect_intermediate_metric_for_nodes_cv,
                                   cv_generator=partial(ts_cv_generator, data=get_ts_data()[0], folds=3,
                                                        validation_blocks=2),
                                   validation_blocks=2,
                                   metric=MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE))),
                          (rf_scaling_pipeline(),
                           get_classification_data()[0],
                           partial(collect_intermediate_metric_for_nodes, input_data=get_classification_data()[1],
                                   metric=MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC))),
                          (lagged_ridge_pipeline(),
                           get_ts_data()[0],
                           partial(collect_intermediate_metric_for_nodes, input_data=get_ts_data()[1],
                                   metric=MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)))])
def test_collect_intermediate_metric(pipeline: Pipeline, data: InputData, method):
    """ Test if intermediate metric collected for nodes """
    pipeline.fit(data)
    method(pipeline=pipeline)

    for node in pipeline.nodes:
        print(node.metadata.metric)
        if isinstance(node.operation, Model):
            assert node.metadata.metric is not None and node.metadata.metric != 0.5 and node.metadata.metric < 10000
        else:
            assert node.metadata.metric is None


@pytest.mark.parametrize("cv_generator, data",
                         [(partial(tabular_cv_generator, folds=3),
                           get_classification_data()[0]),
                          (partial(ts_cv_generator, folds=3),
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
