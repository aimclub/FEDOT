from fedot.core.data.merge.data_merger import ImageDataMerger, TSDataMerger, DataMerger
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor import (
    TopologicalFeaturesImplementation, )
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
    LaggedImplementation,
    TsSmoothingImplementation,
)
from fedot.core.operations.operation import Operation
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.verification import class_rules, ts_rules, common_rules
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.data.data_split import _split_any, _split_time_series
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.api.api_utils.api_composer import ApiComposer
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.optimisers.genetic.operators.reproduction import ReproductionController

import fedot.core.data.data_split as fedot_data_split
import golem.core.tuning.optuna_tuner as OptunaImpl

def resolve_context(context_name: str = "core", backend: str = "default") -> ExecutionContext:
    if context_name == "core":
        return ExecutionContext(backend=backend)

    from fedot.extensions.registry import get_registered_extension, get_registered_extensions

    ext = get_registered_extension(context_name)
    if ext is not None:
        factory = ext.value.manifest.protocols.get("context_factory")
        if factory:
            return factory(backend=backend)

    raise ValueError(f"Unknown context: {context_name}")

class ExecutionContext:
    def __init__(self, backend: str = "default") -> None:
        """Initializes ExecutionContext with default configuration."""
        self.backend = backend
        self._init_defaults()
        self._apply_protocols()

    def _init_defaults(self):
        """Sets default implementations for all pipeline components."""
        self.evaluator_evaluate = PipelineObjectiveEvaluate.evaluate
        self.search_space_get_parameters_dict = PipelineSearchSpace.get_parameters_dict
        self.api_params_repository__get_default_mutations = ApiParamsRepository._get_default_mutations
        self.merger_find_main_output = DataMerger.find_main_output
        self.merger_get = DataMerger.get
        self.merger_merge_predicts = DataMerger.merge_predicts
        self.image_merger_preprocess_predicts = ImageDataMerger.preprocess_predicts
        self.image_merger_merge_predicts = ImageDataMerger.merge_predicts
        self.ts_merger_merge_predicts = TSDataMerger.merge_predicts
        self.ts_merger_merge_targets = TSDataMerger.merge_targets
        self.ts_merger_postprocess_predicts = TSDataMerger.postprocess_predicts
        self.ts_merger_preprocess_predicts = TSDataMerger.preprocess_predicts
        self.data_source_splitter_build = DataSourceSplitter.build
        self.data_split__split_any = fedot_data_split._split_any
        self.data_split__split_time_series = fedot_data_split._split_time_series
        self.operation__predict = Operation._predict
        self.operation_predict = Operation.predict
        self.operation_predict_for_fit = Operation.predict_for_fit
        self.lagged__update_column_types = LaggedImplementation._update_column_types
        self.lagged_transform = LaggedImplementation.transform
        self.lagged_transform_for_fit = LaggedImplementation.transform_for_fit
        self.lagged__check_and_correct_window_size = LaggedImplementation._check_and_correct_window_size
        self.topo_features_fit = TopologicalFeaturesImplementation.fit
        self.topo_features_transform = TopologicalFeaturesImplementation.transform
        self.ts_smoothing_transform = TsSmoothingImplementation.transform
        self.optuna_optuna_tuner = OptunaImpl.OptunaTuner
        self.api_composer_tune_final_pipeline = ApiComposer.tune_final_pipeline
        self.reproduction_reproduce = ReproductionController.reproduce
        self.reproduction_reproduce_uncontrolled = ReproductionController.reproduce_uncontrolled
        self.class_rules = class_rules.copy()
        self.ts_rules = ts_rules.copy()
        self.common_rules = common_rules.copy()

    def _apply_protocols(self):
        # Splitters
        splitters = resolve_protocol_instance("splitters", backend=self.backend)
        if splitters:
            self.data_split__split_any = splitters.split_any
            self.data_split__split_time_series = splitters.split_time_series

        # Mergers
        mergers = resolve_protocol_instance("mergers", backend=self.backend)
        if mergers:
            self.merger_find_main_output = mergers.find_main_output
            self.merger_get = mergers.get
            self.merger_merge_predicts = mergers.merge_predicts
            if hasattr(mergers, 'preprocess_predicts'):
                self.image_merger_preprocess_predicts = mergers.preprocess_predicts
                self.ts_merger_preprocess_predicts = mergers.preprocess_predicts
            if hasattr(mergers, 'postprocess_predicts'):
                self.ts_merger_postprocess_predicts = mergers.postprocess_predicts
            if hasattr(mergers, 'merge_targets'):
                self.ts_merger_merge_targets = mergers.merge_targets
            # Image merge обычно совпадает с основным
            self.image_merger_merge_predicts = mergers.merge_predicts

        # DataSourceSplitter
        splitter_builder = resolve_protocol_instance("data_source_splitter", backend=self.backend)
        if splitter_builder:
            self.data_source_splitter_build = splitter_builder.build

        # Tuner class
        tuner_class = resolve_protocol_instance("tuner_class", backend=self.backend)
        if tuner_class:
            self.optuna_optuna_tuner = tuner_class

        # Reproduction
        reproduction = resolve_protocol_instance("reproduction", backend=self.backend)
        if reproduction:
            self.reproduction_reproduce = reproduction.reproduce
            if hasattr(reproduction, 'reproduce_uncontrolled'):
                self.reproduction_reproduce_uncontrolled = reproduction.reproduce_uncontrolled

        # Evaluator
        evaluator = resolve_protocol_instance("evaluator", backend=self.backend)
        if evaluator:
            self.evaluator_evaluate = evaluator.evaluate

        # Search space
        search_space = resolve_protocol_instance("search_space", backend=self.backend)
        if search_space:
            self.search_space_get_parameters_dict = search_space.get_parameters_dict

        # Mutations
        mutations = resolve_protocol_instance("default_mutations", backend=self.backend)
        if mutations:
            self.api_params_repository__get_default_mutations = mutations

        # Operation predict
        op_predict = resolve_protocol_instance("operation_predict", backend=self.backend)
        if op_predict:
            self.operation_predict = op_predict.predict
            self.operation_predict_for_fit = op_predict.predict_for_fit
            if hasattr(op_predict, '_predict'):
                self.operation__predict = op_predict._predict

        # Lagged transformer
        lagged = resolve_protocol_instance("lagged_transformer", backend=self.backend)
        if lagged:
            self.lagged__update_column_types = lagged._update_column_types
            self.lagged_transform = lagged.transform
            self.lagged_transform_for_fit = lagged.transform_for_fit
            self.lagged__check_and_correct_window_size = lagged._check_and_correct_window_size

        # Topological features
        topo = resolve_protocol_instance("topological_features", backend=self.backend)
        if topo:
            self.topo_features_fit = topo.fit
            self.topo_features_transform = topo.transform

        # TS Smoothing
        smoothing = resolve_protocol_instance("ts_smoothing", backend=self.backend)
        if smoothing:
            self.ts_smoothing_transform = smoothing.transform

        # ApiComposer tune
        tune = resolve_protocol_instance("api_composer_tune", backend=self.backend)
        if tune:
            self.api_composer_tune_final_pipeline = tune

        @cached_property
        def set_operation_registry(self) -> OperationTypesRepository:
            return OperationTypesRepository()