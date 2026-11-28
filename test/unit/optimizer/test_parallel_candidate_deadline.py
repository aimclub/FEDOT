import os
import time
from datetime import timedelta

import pytest
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


class _NativeFitPipeline(Pipeline):
    def _fit(self, *args, **kwargs):
        import ctypes
        from pathlib import Path
        Path(self.native_marker).write_text(f'{os.getpid()} {time.monotonic()}')
        if os.name == 'nt':
            ctypes.windll.kernel32.Sleep(30000)
        else:
            ctypes.CDLL(None).sleep(30)


def test_timed_fit_stops_native_call_and_keeps_the_other_result(tmp_path):
    fast = InputData(
        idx=np.arange(30),
        features=np.random.default_rng(1).normal(size=(30, 4)),
        target=np.array([0, 1] * 15),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    def run(kind):
        if kind == 'slow':
            pipeline = _NativeFitPipeline(PipelineNode('logit'))
            pipeline.native_marker = str(tmp_path / 'native-started')
            data = fast
            limit = timedelta(seconds=5)
        else:
            pipeline = Pipeline(PipelineNode('logit'))
            data = fast
            limit = timedelta(seconds=30)
        try:
            pipeline.fit(data, time_constraint=limit)
            return 'finished', os.getpid()
        except TimeoutError:
            return 'timeout', os.getpid()

    results = Parallel(n_jobs=2)(delayed(run)(kind) for kind in ('slow', 'fast'))

    assert results[0][0] == 'timeout'
    native_pid, native_started = (tmp_path / 'native-started').read_text().split()
    assert int(native_pid) != results[0][1]
    if os.name == 'posix':
        with pytest.raises(ProcessLookupError):
            os.kill(int(native_pid), 0)
    assert time.monotonic() - float(native_started) < 15
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
        use_operations_cache=False,
        use_preprocessing_cache=False,
        use_predictions_cache=False,
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


def _series():
    return InputData(
        idx=np.arange(80), features=np.arange(80, dtype=float),
        target=np.arange(80, dtype=float),
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=12)),
        data_type=DataTypesEnum.ts,
    )


def test_timed_fit_returns_corrected_node_parameters():
    lagged = PipelineNode('lagged')
    lagged.parameters = {'window_size': 1000}
    pipeline = Pipeline(PipelineNode('ridge', nodes_from=[lagged]))
    pipeline.fit(_series(), time_constraint=timedelta(seconds=30))
    assert lagged.parameters['window_size'] == lagged.fitted_operation.window_size
    assert lagged.fit_time_in_seconds is not None
    assert np.isfinite(pipeline.predict(_series()).predict).all()


def test_timed_fit_does_not_require_fork(monkeypatch):
    import multiprocessing

    real_context = multiprocessing.get_context

    def without_fork(method=None):
        if method == 'fork':
            raise ValueError('fork is unavailable on Windows')
        return real_context(method)

    monkeypatch.setattr(multiprocessing, 'get_context', without_fork)
    pipeline = Pipeline(PipelineNode('ridge', nodes_from=[PipelineNode('lagged')]))
    pipeline.fit(_series(), time_constraint=timedelta(seconds=30))
    assert pipeline.is_fitted


@pytest.mark.parametrize('assign_succeeds', [True, False])
def test_windows_job_owns_tree_before_start_and_closes_handles(monkeypatch, assign_succeeds):
    import ctypes
    from types import SimpleNamespace
    from fedot.core.pipelines.pipeline import _windows_job
    closed = []

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def set_limits(job, kind, pointer, size):
        assert job == 123 and kind == 9 and size > 0
        assert pointer._obj.basic.flags == 0x2000
        return 1

    kernel = SimpleNamespace(
        CreateJobObjectW=Function(lambda *_: 123),
        OpenProcess=Function(lambda access, inherit, pid: 456 if (access, inherit, pid) == (0x0101, False, 42) else 0),
        SetInformationJobObject=Function(set_limits),
        AssignProcessToJobObject=Function(lambda *_: assign_succeeds),
        CloseHandle=Function(lambda handle: closed.append(handle)),
    )
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_args, **_kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, 'WinError', lambda _code: OSError('job assignment failed'), raising=False)
    if assign_succeeds:
        close = _windows_job(42)
        assert closed == [456]
        close()
    else:
        with pytest.raises(OSError, match='job assignment failed'):
            _windows_job(42)
    assert sorted(closed) == [123, 456]


def _draw_random_state():
    import random
    return random.random(), np.random.random()


def test_limited_worker_preserves_callers_seed():
    import random
    from fedot.core.pipelines.pipeline import _run_in_process
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(1234)
        np.random.seed(1234)
        expected = _draw_random_state()
        random.seed(1234)
        np.random.seed(1234)
        assert _run_in_process(_draw_random_state, (), 20) == expected
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
