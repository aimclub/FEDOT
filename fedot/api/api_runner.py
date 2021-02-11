from typing import Union

import numpy as np

from fedot.api.api_utils import array_to_input_data, compose_fedot_model, save_predict
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.composer.metrics import F1Metric, MaeMetric, RmseMetric, RocAucMetric
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskParams, TaskTypesEnum


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
                    target: Union[str, np.ndarray] = None,
                    is_predict=False):
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
        elif is_predict:
            target = None
        data_type = DataTypesEnum.table
        if ml_task.task_type == TaskTypesEnum.ts_forecasting:
            data_type = DataTypesEnum.ts
        data = InputData.from_csv(features, task=ml_task, target_column=target, data_type=data_type)
    else:
        raise ValueError('Please specify a features as path to csv file or as Numpy array')

    return data


class Fedot:

    def __init__(self,
                 ml_task: str,
                 composer_params: dict = None,
                 task_params: TaskParams = None,
                 log: Log = None,
                 fedot_model_path: str = './fedot_model.json'):

        self.ml_task = ml_task
        self.fedot_model_path = fedot_model_path
        self.composer_params = composer_params
        self.current_model = None

        self.train_data = None
        self.test_data = None

        self.task_params = task_params

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
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=self.task_params)
                     }
        basic_metric_dict = {'regression': 'RMSE',
                             'classification': 'ROCAUC',
                             'multiclassification': 'F1',
                             'clustering': 'Silhouette',
                             'ts_forecasting': 'RMSE'
                             }

        if self.ml_task == 'clustering':
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
                                         features=features, is_predict=True)

        if self.metric_name == 'F1':
            self.predicted = self.current_model.predict(self.test_data, output_mode='labels')
        else:
            self.predicted = self.current_model.predict(self.test_data)

        if save_predictions:
            save_predict(self.predicted)
        return self.predicted.predict

    def forecast(self,
                 pre_history: Union[str, np.ndarray, InputData],
                 forecast_length: int = 1,
                 save_predictions: bool = False):
        """
        :param pre_history: the array with features for pre-history of the forecast
        :param forecast_length: num of steps to forecast
        :param save_predictions: if True-save predictions as csv-file in working directory.
        :return: the array with predicted values
        """

        if self.ml_task.task_type != TaskTypesEnum.ts_forecasting:
            raise ValueError('Forecasting can be used only for the time series')

        self.ml_task = self.train_data.task

        self.train_data = check_data_type(ml_task=self.ml_task,
                                          features=pre_history, is_predict=True)

        if self.current_model is None:
            self.composer_params['with_tuning'] = False
            self.current_model = self._obtain_model()

        self.current_model = TsForecastingChain(self.current_model.root_node)

        last_ind = int(round(self.train_data.idx[-1]))

        supp_data = InputData(idx=list(range(last_ind, last_ind + forecast_length)),
                              features=None, target=None,
                              data_type=DataTypesEnum.ts,
                              task=self.ml_task)

        self.predicted = self.current_model.forecast(initial_data=self.train_data, supplementary_data=supp_data)

        if save_predictions:
            save_predict(self.predicted)
        return self.predicted.predict

    def save_model(self):
        """
        :return: the json object containing a composite model
        """
        return self.current_model.save_chain(self.fedot_model_path)

    def show_model(self):
        from fedot.core.composer.visualisation import ChainVisualiser
        if self.current_model is not None:
            ChainVisualiser().visualise(self.current_model)
        else:
            self.log.error('No model to visualize')

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
            if self.test_data is None:
                self.test_data = InputData(idx=range(len(target)), features=None, target=target,
                                           task=self.train_data.task,
                                           data_type=self.train_data.data_type)
            else:
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
