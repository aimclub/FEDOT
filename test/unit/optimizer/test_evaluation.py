import datetime

import pytest

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.fitness import Fitness, null_fitness, SingleObjFitness
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third, pipeline_fourth
from test.unit.validation.test_table_cv import get_classification_data


def prepared_objective(pipeline: Pipeline) -> Fitness:
    train_data = get_classification_data()
    pipeline.fit(train_data)

    metric = ClassificationMetricsEnum.accuracy
    objective = Objective(metric)
    return objective(pipeline, reference_data=train_data)


def invalid_objective(pipeline: Pipeline) -> Fitness:
    return null_fitness()


def throwing_exception_objective(pipeline: Pipeline) -> Fitness:
    x = 1/0
    return SingleObjFitness(x)


def test_multiprocessing_dispatcher_without_timelimit_without_multiprocessing():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]

    evaluator = MultiprocessingDispatcher(adapter).dispatch(prepared_objective)
    evaluated_population = evaluator(population)
    fitness = list(map(lambda x: x.fitness, evaluated_population))
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(pipelines) == len(evaluated_population), "Not all pipelines was evaluated"


def test_multiprocessing_dispatcher_with_multiprocessing():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]

    evaluator = MultiprocessingDispatcher(adapter, n_jobs=-1).dispatch(prepared_objective)
    evaluated_population = evaluator(population)
    fitness = list(map(lambda x: x.fitness, evaluated_population))
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(pipelines) == len(evaluated_population), "Not all pipelines was evaluated"

    evaluator = MultiprocessingDispatcher(adapter, n_jobs=1).dispatch(prepared_objective)
    evaluated_population = evaluator(population)
    fitness = list(map(lambda x: x.fitness, evaluated_population))
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(pipelines) == len(evaluated_population), "Not all pipelines was evaluated"


def test_multiprocessing_dispatcher_with_timelimit():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]

    timeout = datetime.timedelta(minutes=0.001)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(adapter, timer=t).dispatch(prepared_objective)
        evaluated_population = evaluator(population)
    fitness = list(map(lambda x: x.fitness, evaluated_population))
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(evaluated_population) >= 1, "At least one pipeline is evaluated"

    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(adapter, timer=t).dispatch(prepared_objective)
        evaluated_population = evaluator(population)
    fitness = list(map(lambda x: x.fitness, evaluated_population))
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(pipelines) == len(evaluated_population), "Not all pipelines was evaluated"


def test_multiprocessing_dispatcher_with_invalid_objective():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]

    evaluator = MultiprocessingDispatcher(adapter).dispatch(invalid_objective)
    with pytest.raises(AttributeError):
        evaluator(population)


def test_multiprocessing_dispatcher_with_objective_throwing_exception():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]

    evaluator = MultiprocessingDispatcher(adapter).dispatch(throwing_exception_objective)
    with pytest.raises(ZeroDivisionError):
        evaluator(population)
