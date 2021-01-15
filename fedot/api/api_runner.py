from typing import Union
import numpy as np
from fedot.api.api_utils import compose_fedot_model, save_predict, array_to_input_data
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.composer.metrics import F1Metric, MaeMetric, RmseMetric, RocAucMetric
from fedot.core.log import default_log, Log


def default_evo_params():
    return {'max_depth': 3,
            'max_arity': 3,
            'pop_size': 20,
            'num_of_generations': 20,
            'learning_time': 2}


def user_evo_params(max_depth: int = 2,
                    max_arity: int = 2,
                    pop_size: int = 20,
                    num_of_generations: int = 20,
                    learning_time: int = 2):
    return {'max_depth': max_depth,
            'max_arity': max_arity,
            'pop_size': pop_size,
            'num_of_generations': num_of_generations,
            'learning_time': learning_time}


def check_data_type(ml_task: Task,
                    features: Union[str, np.ndarray, InputData],
                    target: Union[str, np.ndarray] = None):
    if type(features) == InputData:
        data = features
    elif type(features) == np.ndarray:
        if target is None:
            target = np.array([])

        data = array_to_input_data(features_array=features,
                                   target_array=target,
                                   task_type=ml_task)
    elif type(features) == str:
        if target is None:
            target = 'target'

        data = InputData.from_csv(features, task=ml_task, target_column=target)
    else:
        raise ValueError('Please specify a features as path to csv file or as Numpy array')

    return data


class Fedot:

    def __init__(self,
                 ml_task: str,
                 composer_params: dict = None,
                 log: Log = None,
                 fedot_model_path: str = './fedot_model.json'):

        self.ml_task = ml_task
        self.fedot_model_path = fedot_model_path
        self.composer_params = composer_params
        self.current_model = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if self.composer_params is None:
            self.composer_params = default_evo_params()
        else:
            self.composer_params = {**user_evo_params(), **self.composer_params}

        task_dict = {'regression': Task(TaskTypesEnum.regression),
                     'classification': Task(TaskTypesEnum.classification),
                     'clustering': Task(TaskTypesEnum.clustering),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting)
                     }
        basic_metric_dict = {'regression': 'RMSE',
                             'classification': 'ROCAUC',
                             'multiclassification': 'F1',
                             'clustering': 'Silhouette',
                             'ts_forecasting': 'RMSE'
                             }

        if self.ml_task == 'clustering' or self.ml_task == 'ts_forecasting':
            raise ValueError('This type of task is not not supported in API now')

        self.metric_name = basic_metric_dict[self.ml_task]
        self.ml_task = task_dict[self.ml_task]

    def _get_params(self):
        param_dict = {'train_data': self.train_data,
                      'task': self.ml_task,
                      }
        return {**param_dict, **self.composer_params}

    def _obtain_model(self):
        execution_params = self._get_params()
        self.current_model = compose_fedot_model(**execution_params)
        return self.current_model

    def _check_num_classes(self,
                           train_data: InputData):
        if len(np.unique(train_data.target)) > 2:
            self.metric_name = 'F1'
        return

    def fit(self,
            features: Union[str, np.ndarray, InputData],
            target: Union[str, np.ndarray] = 'target'):
        """
        :param features: the array with features of train data
        :param target: the array with target values of train data
        :return: Chain object
        """
        self.train_data = check_data_type(ml_task=self.ml_task,
                                          features=features,
                                          target=target)
        self._check_num_classes(self.train_data)
        return self._obtain_model()

    def predict(self,
                features: Union[str, np.ndarray, InputData],
                save_predictions: bool = False):
        """
        :param features: the array with features of test data
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :return: the array with predicted values
        """
        if self.current_model is None:
            self.current_model = self._obtain_model()

        self.test_data = check_data_type(ml_task=self.ml_task,
                                         features=features)

        if self.metric_name == 'F1':
            self.predicted = self.current_model.predict(self.test_data, output_mode='labels')
        else:
            self.predicted = self.current_model.predict(self.test_data)

        if save_predictions:
            save_predict(self.predicted)
        return self.predicted.predict

    def save_model(self):
        """
        :return: the json object containing a composite model
        """
        return self.current_model.save_chain(self.fedot_model_path)

    def quality_metric(self,
                       target: np.ndarray = None,
                       metric_name: str = None):
        """
        :param target: the array with target values of test data
        :param metric_name: the name of chosen quality metric
        :return: the value of quality metric
        """
        if metric_name is None:
            metric_name = self.metric_name

        if target is not None:
            self.test_data.target = target

        __metric_dict = {'RMSE': RmseMetric.metric,
                         'MAE': MaeMetric.metric,
                         'ROCAUC': RocAucMetric.metric,
                         'F1': F1Metric.metric,
                         'Silhouette': NotImplemented
                         }

        if __metric_dict[metric_name] is NotImplemented:
            raise ValueError('This quality metric is not available now')
        else:
            metric_value = __metric_dict[metric_name](reference=self.test_data,
                                                      predicted=self.predicted)
        return metric_value
