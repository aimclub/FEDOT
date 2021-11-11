from fedot.api.api_utils.api_composer import ApiComposer
from ..api.test_main_api import get_dataset
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.log import default_log

from testfixtures import LogCapture


def test_compose_fedot_model_with_tuning():
    api_composer = ApiComposer('classification')
    train, test, _ = get_dataset(task_type='classification')
    task = Task(task_type=TaskTypesEnum.classification)
    generations = 1

    with LogCapture() as logs:
        _, _, history = api_composer.compose_fedot_model(api_params=dict(train_data=train,
                                                                         task=task,
                                                                         logger=default_log('test_log'),
                                                                         timeout=0.1,
                                                                         initial_pipeline=None,
                                                                         check_mode=False),
                                                         composer_params=dict(max_depth=1,
                                                                              max_arity=1,
                                                                              pop_size=2,
                                                                              num_of_generations=generations,
                                                                              available_operations=None,
                                                                              composer_metric=None,
                                                                              validation_blocks=None,
                                                                              cv_folds=None,
                                                                              genetic_scheme=None),
                                                         tuning_params=dict(with_tuning=True,
                                                                            tuner_metric=None))
    expected = ('test_log', 'INFO', 'Tuner metric is None, roc_auc_score was set as default')
    logs.check_present(expected, order_matters=False)
