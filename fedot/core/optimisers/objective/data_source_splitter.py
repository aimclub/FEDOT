from functools import partial
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator, ts_cv_generator
from fedot.remote.remote_evaluator import RemoteEvaluator, init_data_for_remote_execution
from .data_objective_advisor import DataObjectiveAdvisor
from .data_objective_eval import DataSource
from ...constants import default_data_split_ratio_by_task


class DataSourceSplitter:
    """
    Splitter of data that provides generator of test-train splits.
    Can provide hold-out validation and k-fold validation.

    :param cv_folds: Number of folds on data for cross-validation.
    If provided, then k-fold validation is used. Otherwise, hold-out validation is used.
    :param validation_blocks: Number of validation blocks for time series forecasting.
    :param split_ratio: Ratio of data for splitting. Applied only in case of hold-out split.
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
            if self.validation_blocks is None:
                default_validation_blocks = 2
                self.validation_blocks = default_validation_blocks
                self.log.info('For timeseries cross validation validation_blocks number was changed ' +
                              f'from None to {default_validation_blocks} blocks')
            cv_generator = partial(ts_cv_generator, data,
                                   self.cv_folds,
                                   self.validation_blocks,
                                   self.log)
        else:
            cv_generator = partial(tabular_cv_generator, data,
                                   self.cv_folds,
                                   self.advisor.propose_kfold(data))
        return cv_generator
