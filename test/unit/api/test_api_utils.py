import numpy as np

from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.preprocessing import DataPreprocessor
from ..api.test_main_api import get_dataset
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.log import default_log

from testfixtures import LogCapture


def test_compose_fedot_model_with_tuning():
    api_composer = ApiComposer('classification')
    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)

    task = Task(task_type=TaskTypesEnum.classification)
    generations = 1

    with LogCapture() as logs:
        _, _, history = api_composer.compose_fedot_model(api_params=dict(train_data=train_input,
                                                                         task=task,
                                                                         logger=default_log('test_log'),
                                                                         timeout=0.1,
                                                                         initial_pipeline=None),
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


def test_output_classification_converting_correct():
    """ Check the correctness of correct prediction method for binary classification task """

    task = Task(task_type=TaskTypesEnum.classification)
    real = InputData(idx=[0, 1, 2], features=np.array([[0], [1], [2]]),
                     target=np.array([[0], [1], [1]]),
                     task=task, data_type=DataTypesEnum.table)
    prediction = OutputData(idx=[0, 1, 2], features=np.array([[0], [1], [2]]),
                            predict=np.array([[0.5], [0.7], [0.7]]),
                            target=np.array([[0], [1], [1]]),
                            task=task, data_type=DataTypesEnum.table)

    data_processor = ApiDataProcessor(task=task)
    # Perform correction inplace
    data_processor.correct_predictions(metric_name='f1', real=real, prediction=prediction)

    assert int(prediction.predict[1]) == 1
