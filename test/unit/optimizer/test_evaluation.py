import datetime

import pytest

from fedot.core.adapter import AdaptRegistry, adapt_population
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.fitness import Fitness, null_fitness
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher, SimpleDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third, pipeline_fourth
from test.unit.validation.test_table_cv import get_classification_data


def set_up_tests():
    AdaptRegistry().init_adapter(PipelineAdapter())
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = adapt_population(pipelines)
    return population


def prepared_objective(pipeline: Pipeline) -> Fitness:
    train_data = get_classification_data()
    pipeline.fit(train_data)

    metric = ClassificationMetricsEnum.logloss
    objective = Objective(metric)
    return objective(pipeline, reference_data=train_data)


def invalid_objective(pipeline: Pipeline) -> Fitness:
    return null_fitness()


@pytest.mark.parametrize(
    'dispatcher',
    [SimpleDispatcher(),
     MultiprocessingDispatcher(),
     MultiprocessingDispatcher(n_jobs=-1),
     MultiprocessingDispatcher(n_jobs=1)]
)
def test_dispatchers_with_and_without_multiprocessing(dispatcher):
    population = set_up_tests()

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
    [MultiprocessingDispatcher(),
     SimpleDispatcher()]
)
def test_dispatchers_with_faulty_objectives(objective, dispatcher):
    population = set_up_tests()
    evaluator = dispatcher.dispatch(objective)
    assert evaluator(population) is None


def test_multiprocessing_dispatcher_with_timeout():
    population = set_up_tests()

    timeout = datetime.timedelta(minutes=0.001)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(timer=t).dispatch(prepared_objective)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(evaluated_population) >= 1, "At least one pipeline is evaluated"

    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(timer=t).dispatch(prepared_objective)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(population) == len(evaluated_population), "Not all pipelines was evaluated"


def test_simple_dispatcher_with_timeout():
    population = set_up_tests()

    timeout = datetime.timedelta(milliseconds=400)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = SimpleDispatcher(timer=t).dispatch(prepared_objective)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(evaluated_population) < len(population), "Not all pipelines should be evaluated (not enough time)"

    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = SimpleDispatcher(timer=t).dispatch(prepared_objective)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(population) == len(evaluated_population), "Not all pipelines was evaluated"
