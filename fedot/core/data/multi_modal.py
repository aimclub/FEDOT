from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import array_to_input_data, get_indices_from_file
from fedot.core.repository.tasks import Task, TaskTypesEnum


class MultiModalData(dict):

    def __init__(self, *arg, **kw):
        super(MultiModalData, self).__init__(*arg, **kw)

    @property
    def idx(self):
        return next(iter(self.values())).idx

    @property
    def task(self):
        return ([v for v in self.values()
                 if v.supplementary_data.is_main_target])[0].task

    @task.setter
    def task(self, value):
        for data_part in self.values():
            data_part.task = value

    @property
    def target(self):
        return next(iter(self.values())).target

    @target.setter
    def target(self, value):
        for data_part in self.values():
            data_part.target = value

    @property
    def data_type(self):
        return [i.data_type for i in iter(self.values())]

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None

    def shuffle(self):
        # TODO implement multi-modal shuffle
        pass

    def subset_range(self, start: int, end: int):
        for key in self.keys():
            self[key] = self[key].subset_range(start, end)
        return self

    def subset_indices(self, selected_idx: List):
        for key in self.keys():
            self[key] = self[key].subset_indices(selected_idx)
        return self

    @staticmethod
    def from_csv_time_series(task: Task,
                             file_path=None,
                             delimiter=',',
                             is_predict=False,
                             var_names=None,
                             target_column: Optional[str] = ''):
        df = pd.read_csv(file_path, sep=delimiter)
        idx = get_indices_from_file(df, file_path)

        if not var_names:
            var_names = list(set(df.columns) - set('datetime'))

        if is_predict:
            raise NotImplementedError(
                'Multivariate predict not supported in this function yet.')
        else:
            train_data, _ = \
                prepare_multimodal_data(dataframe=df,
                                        features=var_names,
                                        forecast_length=0)

            if target_column is not None:
                target = np.array(df[target_column])
            else:
                target = np.array(df[df.columns[-1]])

            # create labels for data sources
            data_part_transformation_func = partial(array_to_input_data, idx=idx,
                                                    target_array=target, task=task)

            sources = dict((_new_key_name(data_part_key),
                            data_part_transformation_func(features_array=data_part))
                           for (data_part_key, data_part) in train_data.items())
            input_data = MultiModalData(sources)

        return input_data


def prepare_multimodal_data(dataframe: pd.DataFrame, features: list, forecast_length: int):
    """ Prepare MultiModal data for time series forecasting task in a form of
    dictionary

    :param dataframe: pandas DataFrame to process
    :param features: columns, which should be used as features in forecasting
    :param forecast_length: length of forecast

    :return multi_modal_train: dictionary with numpy arrays for train
    :return multi_modal_test: dictionary with numpy arrays for test
    """
    multi_modal_train = {}
    multi_modal_test = {}
    for feature in features:
        if forecast_length > 0:
            feature_ts = np.array(dataframe[feature])[:-forecast_length]
            idx = list(dataframe['datetime'])[:-forecast_length]
        else:
            feature_ts = np.array(dataframe[feature])
            idx = list(dataframe['datetime'])

        # Will be the same
        multi_modal_train.update({feature: feature_ts})
        multi_modal_test.update({feature: feature_ts})

    multi_modal_test['idx'] = np.asarray(idx)
    multi_modal_train['idx'] = np.asarray(idx)

    return multi_modal_train, multi_modal_test


def _new_key_name(data_part_key):
    if data_part_key == 'idx':
        return 'idx'
    return f'data_source_ts/{data_part_key}'
