import pytest

from fedot import Fedot
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.constants import FAST_TRAIN_PRESET_NAME
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.data.datasets import data_with_binary_features_and_categorical_target


minimal_sets_for_fast_train_by_task = {
    'classification': {'rf', 'logit', 'knn'},
    'regression': {'rfr', 'lasso', 'ridge', 'linear'},
    'ts_forecasting': {'ar', 'adareg', 'scaling', 'lasso'},
    'clustering': {'kmeans'}
}


@pytest.mark.parametrize('task_type', TaskTypesEnum._member_names_)
def test_presets_by_task(task_type):
    task = Task(TaskTypesEnum(task_type))
    ops_available = get_operations_for_task(task=task, mode='all')

    ops_for_best_quality = OperationsPreset(task=task, preset_name='best_quality').filter_operations_by_preset()
    ops_for_fast_train = OperationsPreset(task=task, preset_name='fast_train').filter_operations_by_preset()
    ops_for_auto = OperationsPreset(task=task, preset_name='auto').filter_operations_by_preset()

    assert len(ops_for_fast_train) < len(ops_for_best_quality) == len(ops_available) == len(ops_for_auto)
    assert minimal_sets_for_fast_train_by_task[task_type] <= set(ops_for_fast_train)


def test_presets_inserting_in_params_correct():
    """
    Check if operations from presets are correctly included in the dictionary
    with parameters for the composer
    """
    composer_params = ApiParamsRepository.default_params_for_task(TaskTypesEnum.regression)
    source_candidates = composer_params.get('available_operations')

    task = Task(TaskTypesEnum.regression)

    preset_best_quality = OperationsPreset(task=task, preset_name='best_quality')
    updated_params = preset_best_quality.composer_params_based_on_preset(composer_params)
    updated_candidates = updated_params.get('available_operations')

    assert source_candidates is None
    assert updated_candidates is not None


def test_auto_preset_converted_correctly():
    """ Checks that the proposed method of automatic preset detection correctly converts a preset """
    tiny_timeout_value = 0.001
    large_pop_size = 500
    data = data_with_binary_features_and_categorical_target()

    simple_init_assumption = Pipeline(PipelineNode('logit'))
    fedot_model = Fedot(problem='classification', preset='auto', timeout=tiny_timeout_value,
                        initial_assumption=simple_init_assumption, pop_size=large_pop_size, with_tuning=False)
    # API must return initial assumption without composing and tuning (due to population size is too large)
    fedot_model.fit(data)
    assert fedot_model.params.get('preset') == FAST_TRAIN_PRESET_NAME


def test_gpu_preset():
    task = Task(TaskTypesEnum.classification)
    preset_gpu = OperationsPreset(task=task, preset_name='gpu')
    operations_for_gpu = preset_gpu.filter_operations_by_preset()
    assert len(operations_for_gpu) > 0

    # return repository state after test
    OperationTypesRepository.assign_repo('model', 'model_repository.json')
