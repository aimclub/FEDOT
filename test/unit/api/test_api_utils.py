from fedot.api.api_utils import compose_fedot_model
from ..api.test_main_api import get_dataset
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.log import default_log

from testfixtures import LogCapture


def test_compose_fedot_model_with_tuning():
    train, test, _ = get_dataset(task_type='classification')
    generations = 1

    with LogCapture() as logs:
        _, _, history = compose_fedot_model(train_data=train,
                                            task=Task(
                                                task_type=TaskTypesEnum.classification),
                                            logger=default_log('test_log'),
                                            max_depth=1, max_arity=1,
                                            pop_size=2, num_of_generations=generations,
                                            timeout=0.1,
                                            with_tuning=True)
    expected = ('test_log', 'INFO', 'Tuner metric is None, roc_auc_score was set as default')
    logs.check_present(expected, order_matters=False)
