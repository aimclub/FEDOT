from functools import partial
from typing import Optional
from copy import deepcopy

from golem.core.log import default_log

from fedot.core.constants import DEFAULT_DATA_SPLIT_RATIO_BY_TASK, DEFAULT_CV_FOLDS_BY_TASK
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
    Applied only in case of hold-out split.
    If not provided, then default split ratios will be used.
    :param shuffle: Is shuffling required for data.
    """

    def __init__(self,
                 cv_folds: Optional[int] = None,
                 validation_blocks: Optional[int] = None,
                 split_ratio: Optional[float] = None,
                 shuffle: bool = False,):
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

        # Check split_ratio
        split_ratio = self.split_ratio or DEFAULT_DATA_SPLIT_RATIO_BY_TASK[data.task.task_type]
        if not (0 < split_ratio < 1):
            raise ValueError(f'split_ratio is {split_ratio} but should be between 0 and 1')

        # Calculate the number of validation blocks and number of cv folds for ts forecasting
        if data.task.task_type is TaskTypesEnum.ts_forecasting:
            if self.validation_blocks is None:
                self._propose_cv_folds_and_validation_blocks(data, split_ratio)
            # when forecasting length is low and data length is high there are huge amount of validation blocks
            # some model refit each step of forecasting that may be time consuming
            # solution is set forecasting length to higher value and reduce validation blocks count
            # without reducing validation data length which is equal to forecast_length * validation_blocks
            max_validation_blocks = DEFAULT_CV_FOLDS_BY_TASK[data.task.task_type] if self.cv_folds is None else 1
            if self.validation_blocks > max_validation_blocks:
                data = self._propose_forecast_length(data, max_validation_blocks)

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

        split_ratio = self.split_ratio or DEFAULT_DATA_SPLIT_RATIO_BY_TASK[data.task.task_type]
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

    def _propose_cv_folds_and_validation_blocks(self, data, split_ratio):
        data_shape = data.target.shape[0]
        forecast_length = data.task.task_params.forecast_length
        # check that cv folds may be realized
        if self.cv_folds is not None:
            max_test_size = data_shape / (self.cv_folds + 1)
            if forecast_length > max_test_size:
                proposed_cv_folds_count = int((data_shape - forecast_length) // forecast_length)
                if proposed_cv_folds_count >= 2:
                    self.log.info((f"Cross validation  with {self.cv_folds} folds cannot be provided"
                                   f" with forecast length {data.task.task_params.forecast_length}"
                                   f" and full data length {data.target.shape[0]}."
                                   f" Cross validation folds is set to {proposed_cv_folds_count}"))
                    self.cv_folds = proposed_cv_folds_count
                else:
                    self.cv_folds = None
                    self.log.info(("Cross validation cannot be provided"
                                   f" with forecast length {data.task.task_params.forecast_length}"
                                   f" and full data length {data.target.shape[0]}."
                                   " Cross validation is switched off."))

        if self.cv_folds is None:
            test_shape = int(data_shape * (1 - split_ratio))
            if forecast_length > test_shape:
                split_ratio = 1 - forecast_length / data_shape
                self.log.info((f"Forecast length ({forecast_length}) is greater than test length"
                               f" ({test_shape}) defined by split ratio."
                               f" Split ratio is changed to {split_ratio}."))
            test_share = 1 - split_ratio
            self.split_ratio = split_ratio
        else:
            test_share = 1 / (self.cv_folds + 1)
        self.validation_blocks = int(data_shape * test_share // forecast_length)

    def _propose_forecast_length(self, data, max_validation_blocks):
        horizon = self.validation_blocks * data.task.task_params.forecast_length
        self.validation_blocks = max_validation_blocks
        # TODO: make copy without copy all data, only with task copy
        data = deepcopy(data)
        data.task.task_params.forecast_length = int(horizon // self.validation_blocks)
        return data
