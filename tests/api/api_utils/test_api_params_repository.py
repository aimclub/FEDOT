import pytest

from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.validation.errors import FedotInvalidKeysError


def test_api_params_repository_builds_task_specific_defaults():
    classification_repository = ApiParamsRepository(
        TaskTypesEnum.classification)
    ts_repository = ApiParamsRepository(TaskTypesEnum.ts_forecasting)

    assert classification_repository.default_params['cv_folds'] == 5
    assert ts_repository.default_params['cv_folds'] == 3


def test_api_params_repository_preserves_valid_sampling_config():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    result = repository.apply_default_params({
        'sampling_config': {'strategy_kind': 'subset', 'strategy': 'random', 'candidate_ratios': [0.2, 0.5]},
    })

    assert result['sampling_config']['strategy_kind'] == 'subset'
    assert result['sampling_config']['strategy'] == 'random'
    assert tuple(result['sampling_config']['candidate_ratios']) == (0.2, 0.5)


def test_api_params_repository_preserves_valid_chunked_ensemble_config():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    result = repository.apply_default_params({
        'chunked_ensemble_config': {
            'validation_size': 0.25,
            'validation_split_seed': 7,
            'ensemble_method': 'weighted',
            'ensemble_params': {'alpha': 0.5},
            'batch_size': 512,
        },
    })

    assert result['chunked_ensemble_config'] == {
        'validation_size': 0.25,
        'validation_split_seed': 7,
        'ensemble_method': 'weighted',
        'ensemble_params': {'alpha': 0.5},
        'batch_size': 512,
        'min_successful_chunks': 1,
    }


def test_api_params_repository_preserves_valid_tensor_data_config():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    result = repository.apply_default_params({
        'tensor_data_config': {
            'backend_name': ' GPU ',
            'use_cache': False,
        },
    })

    assert result['tensor_data_config'] == {
        'backend_name': 'gpu',
        'use_cache': False,
    }


def test_api_params_repository_preserves_cuda_device_backend_name():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    result = repository.apply_default_params({
        'tensor_data_config': {
            'backend_name': ' CUDA:1 ',
        },
    })

    assert result['tensor_data_config']['backend_name'] == 'cuda:1'


def test_apply_default_params_adds_missing_values_and_normalizes_sampling():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    result = repository.apply_default_params({
        'sampling_config': {
            'strategy_kind': 'subset',
            'strategy': 'random',
        },
    })

    assert result['preset'] == AUTO_PRESET_NAME
    assert result['show_progress'] is True
    assert result['sampling_config']['strategy'] == 'random'


def test_api_params_repository_rejects_unknown_param_key():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    with pytest.raises(FedotInvalidKeysError, match='Invalid key parameters'):
        repository.check_and_set_default_params({'unknown': 1})



def test_params_for_composer_requirements_excludes_runtime_cv_folds():
    result = ApiParamsRepository.get_params_for_composer_requirements({
        'cv_folds': 5,
        'max_depth': 3,
        'use_input_preprocessing': True,
    })

    assert result['max_depth'] == 3
    assert 'cv_folds' not in result
