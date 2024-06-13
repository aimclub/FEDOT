import traceback
from typing import List, Optional, Union

from golem.core.log import default_log

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.api_utils.presets import change_preset_based_on_initial_fit
from fedot.api.time import ApiTime
from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.utilities.memory import MemoryAnalytics


class AssumptionsHandler:
    def __init__(self, data: InputData):
        """
        Class for handling operations related with assumptions

        Args:
            data: data for pipelines.
        """
        self.log = default_log(self)
        self.data = data

    def propose_assumptions(self,
                            initial_assumption: Union[List[Pipeline], Pipeline, None],
                            available_operations: List,
                            use_input_preprocessing: bool = True) -> List[Pipeline]:
        """
        Method to propose initial assumptions if needed

        Args:
            initial_assumption: initial assumption given by user
            available_operations: list of available operations defined by user
            use_input_preprocessing: whether to do preprocessing of initial data

        Returns:
            list of initial assumption pipelines
        """

        if initial_assumption is None:
            assumptions_builder = AssumptionsBuilder \
                .get(self.data) \
                .from_operations(available_operations)
            initial_assumption = assumptions_builder.build(use_input_preprocessing=use_input_preprocessing)
        elif isinstance(initial_assumption, Pipeline):
            initial_assumption = [initial_assumption]
        return initial_assumption

    def fit_assumption_and_check_correctness(self,
                                             pipeline: Pipeline,
                                             pipelines_cache: Optional[OperationsCache] = None,
                                             preprocessing_cache: Optional[PreprocessingCache] = None,
                                             eval_n_jobs: int = -1) -> Pipeline:
        """
        Check if initial pipeline can be fitted on a presented data

        :param pipeline: pipeline for checking
        :param pipelines_cache: Cache manager for fitted models, optional.
        :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
        :param eval_n_jobs: number of jobs to fit the initial pipeline
        """
        try:
            data_train, data_test = train_test_data_setup(self.data)
            self.log.info('Initial pipeline fitting started')
            # load preprocessing
            pipeline.try_load_from_cache(pipelines_cache, preprocessing_cache)
            pipeline.fit(data_train, n_jobs=eval_n_jobs)

            if pipelines_cache is not None:
                pipelines_cache.save_pipeline(pipeline)
            if preprocessing_cache is not None:
                preprocessing_cache.add_preprocessor(pipeline)

            pipeline.predict(data_test)
            self.log.info('Initial pipeline was fitted successfully')

            MemoryAnalytics.log(self.log, additional_info='fitting of the initial pipeline')

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
            self.log.message(f"Preset was changed to {preset} due to fit time estimation for initial model.")
        return preset
