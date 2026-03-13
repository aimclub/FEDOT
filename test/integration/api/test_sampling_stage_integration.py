import numpy as np
import pytest

from fedot import Fedot
from fedot.api.sampling_stage.executor import SamplingStageExecutor, SamplingStageOutput
from fedot.api.sampling_stage.providers import SamplingProvider, SamplingSubsetResult
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
            k = max(1, int(round(len(label_idx) * ratio)))
            picked = rng.choice(label_idx, size=min(k, len(label_idx)), replace=False)
            indices.extend(picked.tolist())

        indices = np.array(sorted(set(indices)), dtype=int)
        return SamplingSubsetResult(sample_indices=indices,
                                    sample_scores=None,
                                    meta={'provider': 'stratified_stub'})


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
