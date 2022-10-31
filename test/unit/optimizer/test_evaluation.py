import datetime

import pytest

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.fitness import Fitness, null_fitness
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher, SimpleDispatcher, \
    ObjectiveEvaluationDispatcher
from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.objective.objective import MetricsObjective
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third, pipeline_fourth
from test.unit.validation.test_table_cv import get_classification_data


def set_up_tests():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]
    return adapter, population


def prepared_objective(pipeline: Pipeline) -> Fitness:
    train_data = get_classification_data()
    pipeline.fit(train_data)
    objective = MetricsObjective(ClassificationMetricsEnum.logloss)
    return objective(pipeline, reference_data=train_data)


def invalid_objective(pipeline: Pipeline) -> Fitness:
    return null_fitness()


@pytest.mark.parametrize(
    'dispatcher',
    [SimpleDispatcher(PipelineAdapter()),
     MultiprocessingDispatcher(PipelineAdapter()),
     MultiprocessingDispatcher(PipelineAdapter(), n_jobs=-1),
     MultiprocessingDispatcher(PipelineAdapter(), n_jobs=1)]
)
def test_dispatchers_with_and_without_multiprocessing(dispatcher):
    _, population = set_up_tests()

    evaluator = dispatcher.dispatch(prepared_objective)
    evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(population) == len(evaluated_population), "Not all pipelines was evaluated"


@pytest.mark.parametrize(
    'objective',
    [invalid_objective]
)
@pytest.mark.parametrize(
    'dispatcher',
    [MultiprocessingDispatcher(PipelineAdapter()),
     SimpleDispatcher(PipelineAdapter())]
)
def test_dispatchers_with_faulty_objectives(objective, dispatcher):
    adapter, population = set_up_tests()

    evaluator = dispatcher.dispatch(objective)
    assert evaluator(population) is None


@pytest.mark.parametrize('dispatcher', [
    MultiprocessingDispatcher(PipelineAdapter()),
    SimpleDispatcher(PipelineAdapter()),
])
def test_dispatcher_with_timeout(dispatcher: ObjectiveEvaluationDispatcher):
    adapter, population = set_up_tests()

    timeout = datetime.timedelta(minutes=0.001)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = dispatcher.dispatch(prepared_objective, timer=t)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(evaluated_population) >= 1, "At least one pipeline is evaluated"
    assert len(evaluated_population) < len(population), "Not all pipelines should be evaluated (not enough time)"

    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = dispatcher.dispatch(prepared_objective, timer=t)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(population) == len(evaluated_population), "Not all pipelines was evaluated"
