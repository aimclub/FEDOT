import time
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer

from fedot.api.main import Fedot
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.optimisers.candidate_isolation import run_candidates_isolated
from fedot.core.optimisers.isolated_candidate_dispatcher import IsolatedCandidateDispatcher
from fedot.core.optimisers.isolated_gp_optimizer import IsolatedEvoGraphOptimizer
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _sleep_or_return(label, delay):
    if delay:
        time.sleep(delay)
    return label


def _requirements(**kwargs):
    return PipelineComposerRequirements(primary=['logit'], secondary=['logit'], **kwargs)


def _composer(**requirement_kwargs):
    pipeline = PipelineBuilder().add_node('logit').build()
    return ComposerBuilder(Task(TaskTypesEnum.classification)).with_requirements(
        _requirements(**requirement_kwargs)
    ).with_initial_pipelines(pipeline).build()


def test_parallel_search_with_deadline_uses_isolated_optimizer():
    composer = _composer(timeout=timedelta(minutes=1), n_jobs=2)

    assert isinstance(composer.optimizer, IsolatedEvoGraphOptimizer)
    assert isinstance(composer.optimizer.eval_dispatcher, IsolatedCandidateDispatcher)


def test_single_job_keeps_stock_optimizer():
    composer = _composer(timeout=timedelta(minutes=1), n_jobs=1)

    assert isinstance(composer.optimizer, EvoGraphOptimizer)
    assert not isinstance(composer.optimizer, IsolatedEvoGraphOptimizer)


def test_infinite_timeout_keeps_stock_optimizer():
    composer = _composer(timeout=None, max_graph_fit_time=None, n_jobs=2)

    assert isinstance(composer.optimizer, EvoGraphOptimizer)
    assert not isinstance(composer.optimizer, IsolatedEvoGraphOptimizer)


def test_process_deadline_covers_each_cv_fold_and_search_budget():
    dispatcher = IsolatedCandidateDispatcher.__new__(IsolatedCandidateDispatcher)
    dispatcher._objective_eval = SimpleNamespace(
        _time_constraint=timedelta(seconds=10),
        _data_producer=partial(int, cv_folds=5),
    )
    dispatcher.timer = SimpleNamespace(timeout=timedelta(seconds=1000), seconds_from_start=0)

    assert dispatcher._candidate_timeout_seconds() == 50

    dispatcher.timer = SimpleNamespace(timeout=timedelta(seconds=30), seconds_from_start=0)
    assert dispatcher._candidate_timeout_seconds() == 30


def test_isolated_candidate_timeout_kills_process_and_keeps_sibling():
    outcomes = run_candidates_isolated(
        evaluator=_sleep_or_return,
        requests=(
            ('slow', ('slow', 8), {}),
            ('fast', ('fast', 0), {}),
        ),
        n_jobs=2,
        timeout_seconds=0.4,
        startup_timeout_seconds=20,
    )
    by_uid = {outcome.uid: outcome for outcome in outcomes}

    assert by_uid['fast'].status == 'ok'
    assert by_uid['fast'].value == 'fast'
    assert by_uid['slow'].status == 'timeout'
    assert by_uid['slow'].cleanup_confirmed is True


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
    model = Fedot(problem='classification', timeout=0.1, n_jobs=1, preset='fast_train')

    class _Pipeline:
        def __init__(self, fitted):
            self.is_fitted = fitted

        def fit(self, *_args, **_kwargs):
            raise TimeoutError('fit limit')

    current = _Pipeline(False)
    fallback = _Pipeline(True)
    model.current_pipeline = current
    model.api_composer.fitted_initial_assumption = fallback

    assert model._train_pipeline_on_full_dataset(None, object()) is True
    assert model.current_pipeline is fallback
