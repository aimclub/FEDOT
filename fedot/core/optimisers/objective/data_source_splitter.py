from functools import partial
from typing import Optional

import numpy as np
from golem.core.log import default_log

from fedot.core.constants import default_data_split_ratio_by_task
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_objective_advisor import DataObjectiveAdvisor
from fedot.core.optimisers.objective.data_objective_eval import DataSource
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator, ts_cv_generator
from fedot.remote.remote_evaluator import RemoteEvaluator, init_data_for_remote_execution


class DataSourceSplitter:
    """
    Splitter of data that provides generator of test-train splits.
    Can provide hold-out validation and k-fold validation.

    :param cv_folds: Number of folds on data for cross-validation.
    If provided, then k-fold validation is used. Otherwise, hold-out validation is used.
    :param split_ratio: Ratio of data for splitting.
    Applied only in case of hold-out split. Not for timeseries data.
    If not provided, then default split ratios will be used.
    :param shuffle: Is shuffling required for data.
    """

    def __init__(self,
                 cv_folds: Optional[int] = None,
                 validation_blocks: Optional[int] = None,
                 split_ratio: Optional[float] = None,
                 shuffle: bool = False):
        self.cv_folds = cv_folds
        self.validation_blocks = validation_blocks
        self.split_ratio = split_ratio
        self.shuffle = shuffle
        self.advisor = DataObjectiveAdvisor()
        self.log = default_log(self)

    def build(self, data: InputData) -> DataSource:
        # Shuffle data
        if self.shuffle and data.task.task_type is not TaskTypesEnum.ts_forecasting:
            data.shuffle()

        # Calculate the number of validation blocks
        if self.validation_blocks is None and data.task.task_type is TaskTypesEnum.ts_forecasting:
            split_ratio = self.split_ratio or default_data_split_ratio_by_task[data.task.task_type]
            if not (0 < split_ratio < 1):
                raise ValueError(f'split_ratio is {split_ratio} but should be between 0 and 1')
            if self.cv_folds is not None:
                # long validation ts leads to splitting troubles
                max_test_size = data.target.shape[0] / (self.cv_folds + 1)
                test_size = (1 / split_ratio - 1) / (self.cv_folds + 1 / split_ratio - 1) * data.target.shape[0]
                test_size = min(max_test_size, test_size)
            else:
                test_size = data.target.shape[0] * (1 - split_ratio)
            self.validation_blocks = int(test_size // data.task.task_params.forecast_length)

        # Split data
        if self.cv_folds is not None:
            self.log.info("K-folds cross validation is applied.")
            data_producer = self._build_kfolds_producer(data)
        else:
            self.log.info("Hold out validation is applied.")
            data_producer = self._build_holdout_producer(data)

        return data_producer

    @staticmethod
    def _data_producer(train_data: InputData, test_data: InputData):
        yield train_data, test_data

    def _build_holdout_producer(self, data: InputData) -> DataSource:
        """
        Build trivial data producer for hold-out validation
        that always returns same data split. Equivalent to 1-fold validation.
        """

        split_ratio = self.split_ratio or default_data_split_ratio_by_task[data.task.task_type]
        train_data, test_data = train_test_data_setup(data, split_ratio, validation_blocks=self.validation_blocks)

        if RemoteEvaluator().is_enabled:
            init_data_for_remote_execution(train_data)

        return partial(self._data_producer, train_data, test_data)

    def _build_kfolds_producer(self, data: InputData) -> DataSource:
        if isinstance(data, MultiModalData):
            raise NotImplementedError('Cross-validation is not supported for multi-modal data')
        if data.task.task_type is TaskTypesEnum.ts_forecasting:
            # Perform time series cross validation
            cv_generator = partial(ts_cv_generator, data,
                                   self.cv_folds,
                                   self.validation_blocks,
                                   self.log)
        else:
            cv_generator = partial(tabular_cv_generator, data,
                                   self.cv_folds,
                                   self.advisor.propose_kfold(data))
        return cv_generator
