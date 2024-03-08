import pytest

from fedot import Fedot
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.api.api_utils.presets import OperationsPreset, PresetsEnum
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repo_enum import OperationReposEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.data.datasets import data_with_binary_features_and_categorical_target


def get_available_operation(task_type: TaskTypesEnum, preset: PresetsEnum):
    operation_preset = OperationsPreset(task=Task(task_type), preset_name=preset)
    return operation_preset.filter_operations_by_preset()


@pytest.mark.parametrize('task_type', TaskTypesEnum)
def test_presets(task_type: TaskTypesEnum):
    available_operations = OperationTypesRepository(OperationReposEnum.DEFAULT).suitable_operation(task_type)
    best_quality = get_available_operation(task_type, PresetsEnum.BEST_QUALITY)
    fast_train = get_available_operation(task_type, PresetsEnum.FAST_TRAIN)
    auto = get_available_operation(task_type, PresetsEnum.AUTO)
    assert set(fast_train) == set(auto) < set(best_quality) == set(available_operations)


def test_presets_inserting_in_params_correct():
    """
    Check if operations from presets are correctly included in the dictionary
    with parameters for the composer
    """
    composer_params = ApiParamsRepository.default_params_for_task(TaskTypesEnum.regression)
    source_candidates = composer_params.get('available_operations')

    task = Task(TaskTypesEnum.regression)

    preset_best_quality = OperationsPreset(task=task, preset_name=PresetsEnum.BEST_QUALITY)
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
    fedot_model = Fedot(problem='classification', preset=PresetsEnum.AUTO, timeout=tiny_timeout_value,
                        initial_assumption=simple_init_assumption, pop_size=large_pop_size, with_tuning=False)
    # API must return initial assumption without composing and tuning (due to population size is too large)
    fedot_model.fit(data)
    assert fedot_model.params.get('preset') == PresetsEnum.FAST_TRAIN


def test_gpu_preset():
    # TODO fix GPU preset
    # GPU preset is not prepared yet
    return
    task = Task(TaskTypesEnum.classification)
    preset_gpu = OperationsPreset(task=task, preset_name=PresetsEnum.GPU)
    operations_for_gpu = preset_gpu.filter_operations_by_preset()
    assert len(operations_for_gpu) > 0
