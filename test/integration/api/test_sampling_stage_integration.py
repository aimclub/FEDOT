from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from fedot import Fedot
from fedot.api.sampling_stage.executor import SamplingStageExecutor, SamplingStageOutput
from fedot.api.sampling_stage.providers import SamplingProvider, SamplingSubsetResult, SamplingZooProvider
from fedot.core.pipelines.pipeline_ensemble import PipelineEnsemble
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TsForecastingParams
from test.data.datasets import get_dataset


class StratifiedStubProvider(SamplingProvider):
    def sample(self,
               features: np.ndarray,
               target: np.ndarray,
               strategy: str,
               strategy_params,
               random_state,
               budget_seconds,
               strategy_kind='subset',
               injectable_params=None):
        del features, strategy, strategy_params, budget_seconds, strategy_kind
        rng = np.random.default_rng(random_state)
        indices = []

        target = np.asarray(target).reshape(-1)
        for label in np.unique(target):
            label_idx = np.where(target == label)[0]
            k = max(1, int(round(len(label_idx) * injectable_params['ratio'])))
            picked = rng.choice(label_idx, size=min(k, len(label_idx)), replace=False)
            indices.extend(picked.tolist())

        indices = np.array(sorted(set(indices)), dtype=int)
        return SamplingSubsetResult(sample_indices=indices,
                                    sample_scores=None,
                                    meta={'provider': 'stratified_stub'})


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    task_type: str
    strategy_params: Dict[str, object]
    skip_reason: Optional[str] = None


SAMPLING_STRATEGY_SPECS = [
    StrategySpec(
        name='random',
        kind='chunking',
        task_type='classification',
        strategy_params={'n_partitions': 2},
    ),
    StrategySpec(
        name='stratified',
        kind='chunking',
        task_type='classification',
        strategy_params={'n_partitions': 2},
    ),
    StrategySpec(
        name='advanced_stratified',
        kind='chunking',
        task_type='classification',
        strategy_params={'n_partitions': 2},
    ),
    StrategySpec(
        name='regression_stratified',
        kind='chunking',
        task_type='regression',
        strategy_params={
            'n_bins': 5,
            'encode': 'ordinal',
            'strategy': 'quantile',
            'n_partitions': 2,
            'use_advanced': True,
        },
    ),
    StrategySpec(
        name='temporal',
        kind='chunking',
        task_type='ts_forecasting',
        strategy_params={},
        skip_reason='Temporal strategies are not supported by sampling stage yet.',
    ),
    StrategySpec(
        name='difficulty',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'difficulty_threshold': 0.5,
            'difficulty_metric': 'f1',
            'n_partitions': 2,
            'problem': 'classification',
            'model': RandomForestClassifier(n_estimators=10, random_state=42),
            'chunks_percent': 50,
        },
    ),
    StrategySpec(
        name='uncertainty',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'uncertainty_threshold': 0.5,
            'n_partitions': 2,
            'problem': 'classification',
            'model': RandomForestClassifier(n_estimators=10, random_state=42),
            'chunks_percent': 50,
        },
    ),
    StrategySpec(
        name='balance',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'n_partitions': 2,
            'balance_method': 'smote',
            'balancer_kwargs': {},
        },
    ),
    StrategySpec(
        name='feature_clustering',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'n_partitions': 2,
            'method': 'kmeans',
            'feature_engineering': False,
        },
    ),
    StrategySpec(
        name='tsne_clustering',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'n_components': 2,
            'perplexity': 5,
        },
    ),
    StrategySpec(
        name='delaunay',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'n_partitions': 2,
            'n_clusters': 5,
            'emptiness_threshold': 0.1,
            'dim_reduction_method': 'pca',
            'dim_reduction_target': 2,
        },
    ),
    StrategySpec(
        name='hdbscan',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'min_cluster_size': 5,
            'one_cluster': True,
            'prob_threshold': 0.5,
            'all_points': True,
        },
    ),
    StrategySpec(
        name='voronoi',
        kind='chunking',
        task_type='classification',
        strategy_params={
            'n_partitions': 2,
            'emptiness_threshold': 0.1,
        },
    ),
    StrategySpec(
        name='spectral_leverage',
        kind='subset',
        task_type='classification',
        strategy_params={},
    ),
    StrategySpec(
        name='tensor_energy',
        kind='subset',
        task_type='classification',
        strategy_params={'modes': [0, 1]},
    ),
    StrategySpec(
        name='kernel',
        kind='subset',
        task_type='classification',
        strategy_params={},
    ),
]


def _sampling_zoo_available() -> bool:
    try:
        SamplingZooProvider().load_factory()
        return True
    except ModuleNotFoundError:
        return False


@pytest.fixture(scope='session')
def sampling_zoo_available():
    if not _sampling_zoo_available():
        pytest.skip('Sampling Zoo dependency is not available.')


def _build_sampling_config(spec: StrategySpec) -> Dict[str, object]:
    config: Dict[str, object] = {
        'strategy_kind': spec.kind,
        'provider': 'sampling_zoo',
        'strategy': spec.name,
        'strategy_params': spec.strategy_params,
    }
    if spec.kind == 'subset':
        config.update({
            'candidate_ratios': [0.5],
            'delta_metric_threshold': 1.0,
            'cap_max_timeout_share': 0.7,
            'min_automl_time_minutes': 0.1
        })
    return config


def test_fit_with_sampling_config_none_preserves_default_behavior():
    train_data, _, _ = get_dataset('classification', n_samples=120, n_features=6, iris_dataset=False)

    model = Fedot(problem='classification',
                  timeout=0.1,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  sampling_config=None)
    pipeline = model.fit(features=train_data)

    assert pipeline is not None
    assert model.sampling_stage_metadata is None


def test_fit_with_sampling_enabled_reduces_train_size_and_exposes_metadata(monkeypatch):
    train_data, _, _ = get_dataset('classification', n_samples=120, n_features=6, iris_dataset=False)
    original_size = len(train_data.idx)

    monkeypatch.setattr(SamplingStageExecutor,
                        '_create_provider',
                        lambda *args, **kwargs: StratifiedStubProvider())

    model = Fedot(problem='classification',
                  timeout=0.2,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  sampling_config={
                      'strategy_kind': 'subset',
                      'provider': 'sampling_zoo',
                      'strategy': 'random',
                      'candidate_ratios': [0.8],
                      'delta_metric_threshold': 1.0,
                  })

    pipeline = model.fit(features=train_data)

    assert pipeline is not None
    assert model.sampling_stage_metadata is not None
    assert model.sampling_stage_metadata['status'] == 'applied'
    assert len(model.train_data.idx) < original_size


def test_fail_fast_for_unsupported_ts_task_with_sampling_stage():
    train_data, _, _ = get_dataset('ts_forecasting', validation_blocks=1, forecast_length=5)

    model = Fedot(problem='ts_forecasting',
                  timeout=0.1,
                  task_params=TsForecastingParams(forecast_length=5),
                  sampling_config={
                      'strategy_kind': 'subset',
                      'strategy': 'random',
                      'candidate_ratios': [0.8],
                      'delta_metric_threshold': 0.1,
                  })

    with pytest.raises(ValueError, match='classification/regression'):
        model.fit(features=train_data)


def test_fail_fast_when_sampling_provider_dependency_missing(monkeypatch):
    train_data, _, _ = get_dataset('classification', n_samples=80, n_features=6, iris_dataset=False)

    monkeypatch.setattr(SamplingStageExecutor,
                        '_create_provider',
                        lambda *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError('sampling zoo missing')))

    model = Fedot(problem='classification',
                  timeout=0.2,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  sampling_config={
                      'strategy_kind': 'subset',
                      'provider': 'sampling_zoo',
                      'strategy': 'random',
                      'candidate_ratios': [0.8],
                      'delta_metric_threshold': 0.1,
                  })

    with pytest.raises(ModuleNotFoundError, match='sampling zoo missing'):
        model.fit(features=train_data)


def test_sampling_stage_does_not_persist_timeout_mutation(monkeypatch):
    train_data, _, _ = get_dataset('classification', n_samples=100, n_features=6, iris_dataset=False)

    def fake_sampling_stage(self):
        self.params.timeout = 0.01
        self.sampling_stage_metadata = {
            'status': 'applied',
            'rows_before': len(self.train_data.idx),
            'rows_after': len(self.train_data.idx),
        }

    monkeypatch.setattr(Fedot, '_run_sampling_stage_if_necessary', fake_sampling_stage)

    model = Fedot(problem='classification',
                  timeout=0.2,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  sampling_config={
                      'strategy_kind': 'subset',
                      'provider': 'sampling_zoo',
                      'strategy': 'random',
                      'candidate_ratios': [0.8],
                      'delta_metric_threshold': 0.1,
                  })

    model.fit(features=train_data)

    assert model.params.timeout == pytest.approx(0.2)


def test_sampling_stage_skipped_when_predefined_model(monkeypatch):
    train_data, _, _ = get_dataset('classification', n_samples=100, n_features=6, iris_dataset=False)

    def should_not_run_stage(self):
        raise AssertionError('sampling stage must be skipped for predefined_model')

    monkeypatch.setattr(Fedot, '_run_sampling_stage_if_necessary', should_not_run_stage)

    model = Fedot(problem='classification',
                  timeout=0.2,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  sampling_config={
                      'strategy_kind': 'subset',
                      'provider': 'sampling_zoo',
                      'strategy': 'random',
                      'candidate_ratios': [0.8],
                      'delta_metric_threshold': 0.1,
                  })

    pipeline = model.fit(features=train_data, predefined_model='rf')

    assert pipeline is not None
    assert model.sampling_stage_metadata == {'status': 'skipped', 'reason': 'predefined_model'}


def test_fail_fast_for_multimodal_input_with_sampling_stage():
    from test.data.datasets import load_categorical_multidata

    data, target = load_categorical_multidata()

    model = Fedot(problem='classification',
                  timeout=0.2,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  sampling_config={
                      'strategy_kind': 'subset',
                      'provider': 'sampling_zoo',
                      'strategy': 'random',
                      'candidate_ratios': [0.8],
                      'delta_metric_threshold': 0.1,
                  })

    with pytest.raises(ValueError, match='InputData'):
        model.fit(features=data, target=target)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize('spec', SAMPLING_STRATEGY_SPECS, ids=lambda spec: f'{spec.kind}:{spec.name}')
def test_sampling_stage_runs_all_strategies(spec: StrategySpec,
                                            sampling_zoo_available):
    if spec.skip_reason:
        pytest.skip(spec.skip_reason)

    train_data, test_data, _ = get_dataset(spec.task_type, n_samples=200, n_features=4, iris_dataset=False)
    sampling_config = _build_sampling_config(spec)

    model = Fedot(problem=spec.task_type,
                  timeout=0.5,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  with_tuning=False,
                  sampling_config=sampling_config)

    try:
        pipeline = model.fit(features=train_data)
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(str(exc))

    assert pipeline is not None
    assert model.sampling_stage_metadata is not None
    assert model.sampling_stage_metadata['status'] == 'applied'
    if spec.kind == 'chunking':
        assert isinstance(model.current_pipeline, PipelineEnsemble)
        assert isinstance(model.train_data, list)
    else:
        assert isinstance(model.current_pipeline, Pipeline)

    prediction = model.predict(features=test_data)
    assert len(prediction) == len(test_data.target)


def test_timeout_restored_after_sampling_stage_real_path(monkeypatch):
    train_data, _, _ = get_dataset('classification', n_samples=90, n_features=6, iris_dataset=False)

    def fake_execute(self, train_data_input):
        return SamplingStageOutput(
            train_data=train_data_input,
            metadata={
                'status': 'applied',
                'rows_before': len(train_data_input.idx),
                'rows_after': len(train_data_input.idx),
            },
            elapsed_seconds=1.0,
            updated_timeout_minutes=0.01,
        )

    monkeypatch.setattr(SamplingStageExecutor, 'execute', fake_execute)

    model = Fedot(problem='classification',
                  timeout=0.2,
                  preset='fast_train',
                  max_depth=1,
                  max_arity=2,
                  sampling_config={
                      'strategy_kind': 'subset',
                      'provider': 'sampling_zoo',
                      'strategy': 'random',
                      'candidate_ratios': [0.8],
                      'delta_metric_threshold': 0.1,
                  })

    model.fit(features=train_data)

    assert model.sampling_stage_metadata is not None
    assert model.sampling_stage_metadata['status'] == 'applied'
    assert model.params.timeout == pytest.approx(0.2)
