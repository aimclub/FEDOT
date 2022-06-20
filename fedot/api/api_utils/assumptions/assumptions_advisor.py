import datetime
import traceback

from typing import Union, List, Optional

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.time import ApiTime
from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.pipelines.pipeline import Pipeline


class AssumptionsHandler:
    def __init__(self, log: Log,
                 train_data: InputData):
        """
        Class for handling operations related with assumptions

        :param log: log object
        :param train_data: data for pipelines
        """
        self.log = log
        self.train_data = train_data

    def propose_assumptions(self,
                            initial_assumption: Union[List[Pipeline], Pipeline, None],
                            available_operations: List) -> List[Pipeline]:
        """
        Method to propose  initial assumptions if needed

        :param initial_assumption: initial assumption given by user
        :param available_operations:
        """

        if initial_assumption is None:
            assumptions_builder = AssumptionsBuilder \
                .get(self.train_data) \
                .from_operations(available_operations) \
                .with_logger(self.log)
            initial_assumption = assumptions_builder.build()
        elif isinstance(initial_assumption, Pipeline):
            initial_assumption = [initial_assumption]
        return initial_assumption

    def fit_assumption_and_check_correctness(self,
                                             pipeline: Pipeline,
                                             timer: ApiTime,
                                             data: Union[InputData, MultiModalData],
                                             cache: Optional[OperationsCache] = None, n_jobs=1):
        """ Test is initial pipeline can be fitted on presented data and give predictions """
        try:
            with timer.launch_assumption_fit():
                _, data_test = train_test_data_setup(data)
                self.log.message('Initial pipeline fitting started')
                pipeline.fit(data, n_jobs=n_jobs)
                if cache is not None:
                    cache.save_pipeline(pipeline)
                pipeline.predict(data_test)
                self.log.message('Initial pipeline was fitted successfully')
        except Exception as ex:
            self._raise_evaluating_exception(ex)
        self.log.message(f'Initial pipeline was fitted for {timer.assumption_fit_spend_time.total_seconds()} sec.')
        return pipeline, timer.assumption_fit_spend_time

    def _raise_evaluating_exception(self, ex: Exception):
        fit_failed_info = f'Initial pipeline fit was failed due to: {ex}.'
        advice_info = f'{fit_failed_info} Check pipeline structure and the correctness of the data'

        self.log.info(fit_failed_info)
        print(traceback.format_exc())
        raise ValueError(advice_info)
