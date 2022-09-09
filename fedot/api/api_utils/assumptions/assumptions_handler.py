import traceback
from typing import List, Optional, Union

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.api_utils.presets import change_preset_based_on_initial_fit
from fedot.api.time import ApiTime
from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline


class AssumptionsHandler:
    def __init__(self, data: InputData):
        """
        Class for handling operations related with assumptions

        :param data: data for pipelines
        """
        self.log = default_log(self)
        self.data = data

    def propose_assumptions(self,
                            initial_assumption: Union[List[Pipeline], Pipeline, None],
                            available_operations: List) -> List[Pipeline]:
        """
        Method to propose initial assumptions if needed

        :param initial_assumption: initial assumption given by user
        :param available_operations: list of available operations defined by user
        """

        if initial_assumption is None:
            assumptions_builder = AssumptionsBuilder \
                .get(self.data) \
                .from_operations(available_operations)
            initial_assumption = assumptions_builder.build()
        elif isinstance(initial_assumption, Pipeline):
            initial_assumption = [initial_assumption]
        return initial_assumption

    def fit_assumption_and_check_correctness(self,
                                             pipeline: Pipeline,
                                             pipelines_cache: Optional[OperationsCache] = None,
                                             preprocessing_cache: Optional[PreprocessingCache] = None) -> Pipeline:
        """
        Check if initial pipeline can be fitted on a presented data

        :param pipeline: pipeline for checking
        :param pipelines_cache: Cache manager for fitted models, optional.
        :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
        """
        try:
            data_train, data_test = train_test_data_setup(self.data)
            self.log.info('Initial pipeline fitting started')
            # load preprocessing
            pipeline.try_load_from_cache(pipelines_cache, preprocessing_cache)
            pipeline.fit(data_train)

            if pipelines_cache is not None:
                pipelines_cache.save_pipeline(pipeline)
            if preprocessing_cache is not None:
                preprocessing_cache.add_preprocessor(pipeline)

            pipeline.predict(data_test)
            self.log.info('Initial pipeline was fitted successfully')

        except Exception as ex:
            self._raise_evaluating_exception(ex)
        return pipeline

    def _raise_evaluating_exception(self, ex: Exception):
        fit_failed_info = f'Initial pipeline fit was failed due to: {ex}.'
        advice_info = f'{fit_failed_info} Check pipeline structure and the correctness of the data'
        self.log.info(fit_failed_info)
        print(traceback.format_exc())
        raise ValueError(advice_info)

    def propose_preset(self, preset: Union[str, None], timer: ApiTime, n_jobs: int) -> str:
        """
        Proposes the most suitable preset for current data

        :param preset: predefined preset
        :param timer: ApiTime object
        :param n_jobs: n_jobs parameter

        """
        if not preset or preset == 'auto':
            preset = change_preset_based_on_initial_fit(timer, n_jobs)
            self.log.info(f"Preset was changed to {preset}")
        return preset
