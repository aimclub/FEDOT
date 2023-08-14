from functools import partial
from typing import Optional, Union

from golem.core.log import default_log

from fedot.core.constants import default_data_split_ratio_by_task
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup, _are_stratification_allowed
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_objective_eval import DataSource
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.remote.remote_evaluator import RemoteEvaluator, init_data_for_remote_execution
from fedot.core.data.cv_folds import cv_generator


class DataSourceSplitter:
    """
    Splitter of data that provides generator of test-train splits.
    Can provide hold-out validation and k-fold validation.

    :param cv_folds: Number of folds on data for cross-validation.
    If provided, then cross validation is used. Otherwise, hold-out validation is used.
    :param validation_blocks: Validation blocks count.
    Applied only for time series data.
    If not provided, then value will be calculated.
    :param split_ratio: Ratio of data for splitting.
    Applied only in case of hold-out split.
    If not provided, then default split ratios will be used.
    :param shuffle: Is shuffling required for data.
    :param stratify: If True then stratification is used for samples
    :param random_seed: Random seed for shuffle.
    :param log: Log for logging.
    """

    def __init__(self,
                 cv_folds: Optional[int] = None,
                 validation_blocks: Optional[int] = None,
                 split_ratio: Optional[float] = None,
                 shuffle: bool = False,
                 stratify: bool = True,
                 random_seed: int = 42):
        self.cv_folds = cv_folds
        self.validation_blocks = validation_blocks
        self.split_ratio = split_ratio
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_seed = random_seed
        self.log = default_log(self)

    def build(self, data: Union[InputData, MultiModalData]) -> DataSource:
        # define split_ratio
        self.split_ratio = self.split_ratio or default_data_split_ratio_by_task[data.task.task_type]

        # Check cv_folds
        if self.cv_folds is not None:
            if not isinstance(self.cv_folds, int):
                if self.cv_folds % 1 != 0:
                    raise ValueError(f"cv_folds is not integer: {self.cv_folds}")
                self.cv_folds = int(self.cv_folds)
            if self.cv_folds < 2:
                self.cv_folds = None
            if self.cv_folds > data.target.shape[0] - 1:
                raise ValueError((f"cv_folds ({self.cv_folds}) is greater than"
                                  f" the maximum allowed count {data.target.shape[0] - 1}"))

        # Calculate the number of validation blocks for timeseries forecasting
        if data.task.task_type is TaskTypesEnum.ts_forecasting and self.validation_blocks is None:
            self._propose_cv_folds_and_validation_blocks(data)

        # Check split_ratio
        if self.cv_folds is None and not (0 < self.split_ratio < 1):
            raise ValueError(f'split_ratio is {self.split_ratio} but should be between 0 and 1')

        if self.stratify:
            # check that stratification can be done
            # for cross validation split ratio is defined as validation_size / all_data_size
            split_ratio = self.split_ratio if self.cv_folds is None else (1 - 1 / (self.cv_folds + 1))
            self.stratify = _are_stratification_allowed(data, split_ratio)
            if not self.stratify:
                self.log.info("Stratificated splitting of data is disabled.")

        # Stratification can not be done without shuffle
        self.shuffle |= self.stratify

        # Random seed depends on shuffle
        self.random_seed = (self.random_seed or 42) if self.shuffle else None

        # Split data
        if self.cv_folds is not None:
            self.log.info("K-folds cross validation is applied.")
            data_producer = partial(cv_generator,
                                    data=data,
                                    shuffle=self.shuffle,
                                    cv_folds=self.cv_folds,
                                    random_seed=self.random_seed,
                                    stratify=self.stratify,
                                    validation_blocks=self.validation_blocks)
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

        train_data, test_data = train_test_data_setup(data,
                                                      split_ratio=self.split_ratio,
                                                      stratify=self.stratify,
                                                      random_seed=self.random_seed,
                                                      shuffle=self.shuffle,
                                                      validation_blocks=self.validation_blocks)

        if RemoteEvaluator().is_enabled:
            init_data_for_remote_execution(train_data)

        return partial(self._data_producer, train_data, test_data)

    def _propose_cv_folds_and_validation_blocks(self, data, expected_window_size=20):
        data_shape = data.target.shape[0]
        # first expected_window_size points should to be guaranteed for prediction at fit stage
        data_shape -= expected_window_size
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
            test_shape = int(data_shape * (1 - self.split_ratio))
            if forecast_length > test_shape:
                self.split_ratio = 1 - forecast_length / data_shape
                self.log.info((f"Forecast length ({forecast_length}) is greater than test length"
                               f" ({test_shape}) defined by split ratio."
                               f" Split ratio is changed to {self.split_ratio}."))
            test_share = 1 - self.split_ratio
        else:
            test_share = 1 / (self.cv_folds + 1)
        self.validation_blocks = int(data_shape * test_share // forecast_length)
