from datetime import timedelta
from functools import partial
from typing import Optional

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.split import ts_cv_generator, tabular_cv_generator
from fedot.remote.remote_evaluator import RemoteEvaluator, init_data_for_remote_execution
from .objective import Objective
from .objective_eval import ObjectiveEvaluate
from .data_objective_eval import DataSource, PipelineObjectiveEvaluate
from ...constants import default_data_split_ratio_by_task


class DataObjectiveBuilder:

    def __init__(self,
                 objective: Objective,
                 max_pipeline_fit_time: Optional[timedelta] = None,
                 cv_folds: Optional[int] = None,
                 validation_blocks: Optional[int] = None,
                 cache: Optional[OperationsCache] = None,
                 log: Log = None):

        self.objective = objective
        self.max_pipeline_fit_time = max_pipeline_fit_time
        self.cv_folds = cv_folds
        self.validation_blocks = validation_blocks
        self.cache = cache
        self.log = log or default_log(self.__class__.__name__)

    def build(self, data: InputData, **kwargs) -> ObjectiveEvaluate:
        """ Compose evaluator object with desired parameters """
        if self.cv_folds is not None:
            data_producer = self._build_kfolds_producer(data, **kwargs)
        else:
            data_producer = self._build_holdout_producer(data, **kwargs)

        objective_evaluate = PipelineObjectiveEvaluate(objective=self.objective,
                                                       data_producer=data_producer,
                                                       time_constraint=self.max_pipeline_fit_time,
                                                       validation_blocks=self.validation_blocks,
                                                       cache=self.cache, log=self.log)
        return objective_evaluate

    def _build_holdout_producer(self, data: InputData, **kwargs) -> DataSource:
        """
        Build trivial data producer for hold-out validation
        that always returns same data split. Equivalent to 1-fold validation.
        """

        self.log.info("Hold out validation for graph composing was applied.")
        split_ratio = default_data_split_ratio_by_task[data.task.task_type]
        train_data, test_data = train_test_data_setup(data, split_ratio,
                                                      **{'validation_blocks': kwargs.get('validation_blocks')})

        def data_producer(): yield train_data, test_data

        if RemoteEvaluator().use_remote:
            init_data_for_remote_execution(train_data)

        return data_producer

    def _build_kfolds_producer(self, data: InputData, **kwargs) -> DataSource:
        if isinstance(data, MultiModalData):
            raise NotImplementedError('Cross-validation is not supported for multi-modal data')
        if data.task.task_type is TaskTypesEnum.ts_forecasting:
            # Perform time series cross validation
            self.log.info("Time series cross validation for pipeline composing was applied.")
            if self.validation_blocks is None:
                default_validation_blocks = 3
                self.log.info(f'For ts cross validation validation_blocks number was changed ' +
                              f'from None to {default_validation_blocks} blocks')
            cv_generator = partial(ts_cv_generator, data,
                                   self.cv_folds,
                                   self.validation_blocks,
                                   self.log)
        else:
            self.log.info("KFolds cross validation for pipeline composing was applied.")
            cv_generator = partial(tabular_cv_generator, data,
                                   self.cv_folds)
        return cv_generator
