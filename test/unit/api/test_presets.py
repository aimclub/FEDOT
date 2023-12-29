from fedot import Fedot
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.constants import FAST_TRAIN_PRESET_NAME
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.data.datasets import data_with_binary_features_and_categorical_target


def test_presets_classification():
    task = Task(TaskTypesEnum.classification)
    class_operations = get_operations_for_task(task=task, mode='all')

    excluded_tree = ['xgboost', 'xgbreg']
    filtered_operations = set(class_operations).difference(set(excluded_tree))
    available_operations = list(filtered_operations)

    preset_best_quality = OperationsPreset(task=task, preset_name='best_quality')
    operations_for_best_quality = preset_best_quality.filter_operations_by_preset()

    preset_fast_train = OperationsPreset(task=task, preset_name='fast_train')
    operations_for_fast_train = preset_fast_train.filter_operations_by_preset()

    preset_auto = OperationsPreset(task=task, preset_name='auto')
    operations_for_auto = preset_auto.filter_operations_by_preset()

    assert len(operations_for_fast_train) < len(operations_for_best_quality) == len(available_operations) == len(
        operations_for_auto)
    assert {'dt', 'logit', 'knn'} <= set(operations_for_fast_train)


def test_presets_regression():
    task = Task(TaskTypesEnum.regression)

    regr_operations = get_operations_for_task(task=task, mode='all')

    preset_best_quality = OperationsPreset(task=task, preset_name='best_quality')
    operations_for_best_quality = preset_best_quality.filter_operations_by_preset()

    preset_fast_train = OperationsPreset(task=task, preset_name='fast_train')
    operations_for_fast_train = preset_fast_train.filter_operations_by_preset()

    preset_auto = OperationsPreset(task=task, preset_name='auto')
    operations_for_auto = preset_auto.filter_operations_by_preset()

    assert len(operations_for_fast_train) < len(operations_for_best_quality) == len(regr_operations) == len(
        operations_for_auto)
    assert {'dtreg', 'lasso', 'ridge', 'linear'} <= set(operations_for_fast_train)


def test_presets_time_series():
    task = Task(TaskTypesEnum.ts_forecasting)

    ts_operations = get_operations_for_task(task=task, mode='all')

    preset_best_quality = OperationsPreset(task=task, preset_name='best_quality')
    operations_for_best_quality = preset_best_quality.filter_operations_by_preset()

    preset_fast_train = OperationsPreset(task=task, preset_name='fast_train')
    operations_for_fast_train = preset_fast_train.filter_operations_by_preset()

    preset_auto = OperationsPreset(task=task, preset_name='auto')
    operations_for_auto = preset_auto.filter_operations_by_preset()

    assert len(operations_for_fast_train) < len(operations_for_best_quality) == len(ts_operations) == len(
        operations_for_auto)
    # TODO: add 'ar' to set below after arima fixing
    assert {'adareg', 'scaling', 'lasso'} <= set(operations_for_fast_train)


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
