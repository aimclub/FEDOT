import datetime
import traceback

from typing import Union, List, Optional

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.api_utils.presets import change_preset_based_on_initial_fit
from fedot.api.time import ApiTime
from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import Log
from fedot.core.pipelines.pipeline import Pipeline


class AssumptionsHandler:
    def __init__(self, log: Log,
                 data: InputData):
        """
        Class for handling operations related with assumptions

        :param log: log object
        :param data: data for pipelines
        """
        self.log = log
        self.data = data

    def propose_assumptions(self,
                            initial_assumption: Union[List[Pipeline], Pipeline, None],
                            available_operations: List) -> List[Pipeline]:
        """
        Method to propose  initial assumptions if needed

        :param initial_assumption: initial assumption given by user
        :param available_operations: list of available operations defined by user
        """

        if initial_assumption is None:
            assumptions_builder = AssumptionsBuilder \
                .get(self.data) \
                .from_operations(available_operations) \
                .with_logger(self.log)
            initial_assumption = assumptions_builder.build()
        elif isinstance(initial_assumption, Pipeline):
            initial_assumption = [initial_assumption]
        return initial_assumption

    def fit_assumption_and_check_correctness(self,
                                             pipeline: Pipeline,
                                             timer: ApiTime,
                                             cache: Optional[OperationsCache] = None) -> [Pipeline, datetime.timedelta]:
        """
        Check is initial pipeline can be fitted on a presented data

        :param pipeline: pipeline for checking
        :param timer: ApiTime object fot handling time
        :param cache: cache object
        """
        try:
            with timer.launch_assumption_fit():
                data_train, data_test = train_test_data_setup(self.data)
                self.log.message('Initial pipeline fitting started')
                pipeline.fit(data_train)
                if cache is not None:
                    cache.save_pipeline(pipeline)
                pipeline.predict(data_test)
                self.log.message('Initial pipeline was fitted successfully')
        except Exception as ex:
            self._raise_evaluating_exception(ex)
        self.log.message(f'Initial pipeline was fitted for {timer.assumption_fit_spend_time.total_seconds()} sec.')
        return pipeline

    def _raise_evaluating_exception(self, ex: Exception):
        fit_failed_info = f'Initial pipeline fit was failed due to: {ex}.'
        advice_info = f'{fit_failed_info} Check pipeline structure and the correctness of the data'
        self.log.info(fit_failed_info)
        print(traceback.format_exc())
        raise ValueError(advice_info)

    def propose_preset(self, preset: Union[str, None], timer: ApiTime, timeout: int) -> str:
        """
        Proposes the most suitable preset for current data

        :param preset: predefined preset
        :param timer: ApiTime object
        :param timeout: timeout from api

        """
        if not preset or preset == 'auto':
            preset = change_preset_based_on_initial_fit(timer, timeout)
            self.log.info(f"Preset was changed to {preset}")
        return preset
