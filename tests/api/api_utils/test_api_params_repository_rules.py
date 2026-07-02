import pytest
from dataclasses import dataclass

from fedot.api.api_utils.api_params_repository_rules import (
    build_default_api_params,
    default_cv_folds_for_task,
    normalize_chunked_ensemble_config,
    normalize_sampling_config,
    normalize_tensor_data_config,
    validate_api_param_keys,
)
from fedot.core.constants import AUTO_PRESET_NAME
from fedot.validation.errors import FedotInvalidKeysError
from fedot.core.pipelines.ensembling.config import ChunkedEnsembleConfig, EnsembleMethod
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class ValidatedConfig:
    strategy: str = 'random'


def test_default_cv_folds_for_task_matches_supported_tasks():
    assert default_cv_folds_for_task(TaskTypesEnum.classification) == 5
    assert default_cv_folds_for_task(TaskTypesEnum.regression) == 5
    assert default_cv_folds_for_task(TaskTypesEnum.ts_forecasting) == 3


def test_build_default_api_params_contains_expected_defaults():
    defaults = build_default_api_params(
        TaskTypesEnum.classification, 'cache_dir')

    assert defaults['preset'] == AUTO_PRESET_NAME
    assert defaults['cv_folds'] == 5
    assert defaults['cache_dir'] == 'cache_dir'
    assert defaults['history_dir'] == 'cache_dir'
    assert defaults['tensor_data_config'] is None


def test_validate_api_param_keys_rejects_unknown_keys():
    with pytest.raises(FedotInvalidKeysError, match='Invalid key parameters'):
        validate_api_param_keys({'unknown': 1}, {'known'})


def test_normalize_sampling_config_uses_validator_result():
    assert normalize_sampling_config(
        {'strategy': 'random'}, lambda config: ValidatedConfig()) == {'strategy': 'random'}
    assert normalize_sampling_config(None, lambda config: None) is None


def test_normalize_chunked_ensemble_config_uses_validator_result():
    config = ChunkedEnsembleConfig(
        validation_size=0.25,
        validation_split_seed=7,
        ensemble_method=EnsembleMethod.weighted,
        ensemble_params={'alpha': 0.5},
        batch_size=512,
    )

    assert normalize_chunked_ensemble_config({'ensemble_method': 'weighted'}, lambda _: config) == {
        'validation_size': 0.25,
        'validation_split_seed': 7,
        'ensemble_method': 'weighted',
        'ensemble_params': {'alpha': 0.5},
        'batch_size': 512,
        'min_successful_chunks': 1,
    }
    assert normalize_chunked_ensemble_config(None, lambda _: config) is None


def test_normalize_tensor_data_config_uses_validator_result():
    assert normalize_tensor_data_config(
        {'backend_name': 'gpu'},
        lambda config: {'backend_name': 'gpu', 'use_cache': False},
    ) == {'backend_name': 'gpu', 'use_cache': False}
    assert normalize_tensor_data_config(None, lambda config: {'backend_name': 'cpu'}) is None

