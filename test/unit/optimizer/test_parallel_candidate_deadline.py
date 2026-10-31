import time
from datetime import timedelta

import numpy as np
from joblib import Parallel, delayed

from fedot.api.main import Fedot
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer


def _composer(**requirement_kwargs):
    pipeline = PipelineBuilder().add_node('logit').build()
    requirements = PipelineComposerRequirements(
        primary=['logit'], secondary=['logit'], **requirement_kwargs)
    return ComposerBuilder(Task(TaskTypesEnum.classification)).with_requirements(
        requirements
    ).with_initial_pipelines(pipeline).build()


def test_parallel_search_keeps_the_stock_optimizer():
    composer = _composer(timeout=timedelta(minutes=1), n_jobs=2)

    assert isinstance(composer.optimizer, EvoGraphOptimizer)
    assert composer.optimizer.__class__ is EvoGraphOptimizer


def test_timed_fit_stops_native_call_and_keeps_the_other_result():
    slow = InputData(
        idx=np.arange(8000),
        features=np.random.default_rng(8000).normal(size=(8000, 12)),
        target=np.zeros(8000),
        task=Task(TaskTypesEnum.ts_forecasting),
        data_type=DataTypesEnum.table,
    )
    slow.supplementary_data.is_auto_preprocessed = True
    fast = InputData(
        idx=np.arange(30),
        features=np.random.default_rng(1).normal(size=(30, 4)),
        target=np.array([0, 1] * 15),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    def run(kind):
        if kind == 'slow':
            from gph import ripser_parallel
            from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor import \
                TopologicalFeaturesImplementation

            def _slow_extract(_self, _features):
                cloud = np.random.default_rng(8000).normal(size=(8000, 12))
                ripser_parallel(cloud, maxdim=1, n_threads=1, collapse_edges=False)
                return np.zeros((len(_features), 1))

            TopologicalFeaturesImplementation._extract_features = _slow_extract
            pipeline = Pipeline(PipelineNode('topological_features'))
            data = slow
            limit = timedelta(seconds=2)
        else:
            pipeline = Pipeline(PipelineNode('logit'))
            data = fast
            limit = timedelta(seconds=30)
        started = time.monotonic()
        try:
            pipeline.fit(data, time_constraint=limit)
            return 'finished', round(time.monotonic() - started, 3)
        except TimeoutError:
            return 'timeout', round(time.monotonic() - started, 3)

    results = Parallel(n_jobs=2)(delayed(run)(kind) for kind in ('slow', 'fast'))

    assert results[0][0] == 'timeout'
    assert results[0][1] < 12
    assert results[1][0] == 'finished'


def test_parallel_fit_returns_a_pipeline():
    from test.data.datasets import get_dataset

    train, test, _threshold = get_dataset('classification')
    model = Fedot(
        problem='classification',
        timeout=0.5,
        n_jobs=2,
        preset='fast_train',
        with_tuning=False,
        pop_size=2,
        num_of_generations=1,
        max_depth=2,
        seed=42,
    )
    pipeline = model.fit(features=train, target='target')
    prediction = model.predict(features=test)

    assert pipeline is not None
    assert model.current_pipeline.is_fitted
    assert len(prediction) == len(test.target)


def test_final_refit_timeout_keeps_fitted_pipeline():
    model = Fedot(
        problem='ts_forecasting',
        task_params=TsForecastingParams(forecast_length=12),
        timeout=0.1,
        n_jobs=1,
        preset='fast_train',
        with_tuning=False,
        logging_level=40,
    )
    lagged = PipelineNode('lagged')
    initial = Pipeline(PipelineNode('ridge', nodes_from=[lagged]))
    model.fit(features=np.arange(80, dtype=float), predefined_model=initial)
    fitted = model.current_pipeline

    class _Slow:
        is_fitted = False

        def fit(self, *_args, **_kwargs):
            raise TimeoutError('fit limit')

    model.api_composer.fitted_initial_assumption = fitted
    model.current_pipeline = _Slow()
    model.params['max_pipeline_fit_time'] = 0.02

    assert model._train_pipeline_on_full_dataset(None, model.train_data) is True
    assert model.current_pipeline is fitted
    assert model.current_pipeline.is_fitted
