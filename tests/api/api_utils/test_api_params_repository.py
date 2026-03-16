import pytest

from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.repository.tasks import TaskTypesEnum


def test_api_params_repository_builds_task_specific_defaults():
    classification_repository = ApiParamsRepository(TaskTypesEnum.classification)
    ts_repository = ApiParamsRepository(TaskTypesEnum.ts_forecasting)

    assert classification_repository.default_params['cv_folds'] == 5
    assert ts_repository.default_params['cv_folds'] == 3


def test_api_params_repository_preserves_valid_sampling_config():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    result = repository.check_and_set_default_params({
        'sampling_config': {'strategy_kind': 'subset', 'strategy': 'random', 'candidate_ratios': [0.2, 0.5]},
    })

    assert result['sampling_config']['strategy_kind'] == 'subset'
    assert result['sampling_config']['strategy'] == 'random'
    assert tuple(result['sampling_config']['candidate_ratios']) == (0.2, 0.5)


def test_api_params_repository_rejects_unknown_param_key():
    repository = ApiParamsRepository(TaskTypesEnum.classification)

    with pytest.raises(KeyError, match='Invalid key parameters'):
        repository.check_and_set_default_params({'unknown': 1})
