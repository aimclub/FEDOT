from typing import List, Optional, Union

from golem.core.log import default_log
from pymonad.either import Left, Right

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.api_utils.assumptions.assumptions_handler_rules import (
    build_assumption_fit_error,
    decide_preset,
    resolve_initial_assumption,
)
from fedot.api.api_utils.presets import change_preset_based_on_initial_fit
from fedot.api.api_utils.schemas import raise_from_assumption_fit_error
from fedot.api.time import ApiTime
from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.utilities.memory import MemoryAnalytics

# TODO @romankuklo: refactor this class
class AssumptionsHandler:
    def __init__(self, data: TensorData):
        """
        Class for handling operations related with assumptions

        Args:
            data: data for pipelines.
        """
        self.log = default_log(self)
        self.data = data

    def propose_assumptions_with_tensordata(
            self,
            initial_assumption: Union[List[Pipeline], Pipeline, None],
            available_operations: Optional[List] = None,
            use_input_preprocessing: bool = False) -> List[Pipeline]:
        """Return user-provided or automatically built initial assumptions for TensorData."""
        use_input_preprocessing = False
        return resolve_initial_assumption(
            initial_assumption,
            builder=lambda: AssumptionsBuilder
            .get(self.data)
            .from_operations(available_operations)
            .build(use_input_preprocessing=use_input_preprocessing),
        )

    def fit_assumption_and_check_correctness_with_tensordata(
            self,
            pipeline: Pipeline,
            operations_cache: Optional[OperationsCache] = None,
            preprocessing_cache: Optional[PreprocessingCache] = None,
            eval_n_jobs: int = -1) -> Pipeline:
        """
        Check if initial pipeline can be fitted on TensorData.

        :param pipeline: pipeline for checking
        :param operations_cache: Cache manager for fitted models, optional.
        :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
        :param eval_n_jobs: number of jobs to fit the initial pipeline
        """
        fit_result = self.try_fit_assumption_with_tensordata(
            pipeline=pipeline,
            operations_cache=operations_cache,
            preprocessing_cache=preprocessing_cache,
            eval_n_jobs=eval_n_jobs,
        )
        if fit_result.is_left():
            fit_error = fit_result.monoid[0] if getattr(
                fit_result, 'monoid', None) else fit_result.value
            raise_from_assumption_fit_error(fit_error)
        return fit_result.value

    def try_fit_assumption_with_tensordata(
            self,
            pipeline: Pipeline,
            operations_cache: Optional[OperationsCache] = None,
            preprocessing_cache: Optional[PreprocessingCache] = None,
            eval_n_jobs: int = -1):
        try:
            data_source = DataSourceSplitter().build_tensordata(self.data)
            data_train, data_test = next(data_source())
            self.log.info('Initial pipeline fitting started')
            pipeline.try_load_from_cache(operations_cache, preprocessing_cache)
            pipeline.fit_tensordata(data_train, n_jobs=eval_n_jobs)

            if operations_cache is not None:
                operations_cache.save_pipeline(pipeline)
            if preprocessing_cache is not None:
                preprocessing_cache.add_preprocessor(pipeline)

            pipeline.predict_tensordata(data_test)
            self.log.info('Initial pipeline was fitted successfully')

            MemoryAnalytics.log(
                self.log, additional_info='fitting of the initial pipeline')
            return Right(pipeline)

        except Exception as ex:
            fit_error = build_assumption_fit_error(ex)
            self.log.exception(
                f'Initial pipeline fit was failed due to: {fit_error.cause}.')
            return Left(fit_error)

    def propose_preset(self, preset: Union[str, None], timer: ApiTime, n_jobs: int) -> str:
        """
        Proposes the most suitable preset for current data

        :param preset: predefined preset
        :param timer: ApiTime object
        :param n_jobs: n_jobs parameter

        """
        decision = decide_preset(
            preset=preset,
            timer=timer,
            n_jobs=n_jobs,
            chooser=change_preset_based_on_initial_fit,
        )
        if decision.was_changed:
            self.log.message(
                f"Preset was changed to {decision.preset} due to fit time estimation for initial model.")
        return decision.preset
