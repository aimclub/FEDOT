import fedot.industrial.core.repository.model_repository as MODEL_REPO
from fedot.industrial.core.metrics.pipeline import industrial_evaluate_pipeline
from fedot.industrial.core.repository.constanst_repository import IND_DATA_OPERATION_PATH, IND_MODEL_OPERATION_PATH, DEFAULT_DATA_OPERATION_PATH, DEFAULT_MODEL_OPERATION_PATH
from fedot.industrial.core.repository.industrial_implementations.abstract import (
    preprocess_industrial_predicts, merge_industrial_predicts, merge_industrial_targets,
    build_industrial, postprocess_industrial_predicts, split_any_industrial,
    split_time_series_industrial, predict_operation_industrial, predict_industrial,
    predict_for_fit_industrial, update_column_types_industrial, fit_topo_extractor_industrial,
    transform_topo_extractor_industrial, find_main_output_industrial, get_merger_industrial
)
from fedot.industrial.core.repository.industrial_implementations.data_transformation import (
    transform_lagged_industrial, transform_lagged_for_fit_industrial,
    _check_and_correct_window_size_industrial, transform_smoothing_industrial
)
from fedot.industrial.core.repository.industrial_implementations.ml_optimisation import (
    DaskOptunaTuner, tune_pipeline_industrial
)
from fedot.industrial.core.repository.industrial_implementations.optimisation import (
    _get_default_industrial_mutations, has_no_lagged_conflicts_in_ts_pipeline,
    reproduce_controlled_industrial, reproduce_industrial,
    has_no_data_flow_conflicts_in_industrial_pipeline
)
from fedot.industrial.core.tuning.search_space import get_industrial_search_space

from fedot.core.context import ExecutionContext


class IndustrialExtension:
    """Overrides ExecutionContext with industrial implementations."""
    def __init__(self, backend: str = "default") -> None:
        self.backend = backend

    def apply(self, context: ExecutionContext) -> None:
        """Mutates context with industrial implementations."""
        context.optuna_optuna_tuner = DaskOptunaTuner if "dask" in self.backend else OptunaTuner
        context.evaluator_evaluate = industrial_evaluate_pipeline
        context.search_space_get_parameters_dict = get_industrial_search_space
        context.api_params_repository__get_default_mutations = _get_default_industrial_mutations
        context.merger_find_main_output = find_main_output_industrial
        context.merger_get = get_merger_industrial
        context.merger_merge_predicts = merge_industrial_predicts
        context.image_merger_preprocess_predicts = preprocess_industrial_predicts
        context.image_merger_merge_predicts = merge_industrial_predicts
        context.ts_merger_merge_predicts = merge_industrial_predicts
        context.ts_merger_merge_targets = merge_industrial_targets
        context.ts_merger_postprocess_predicts = postprocess_industrial_predicts
        context.ts_merger_preprocess_predicts = preprocess_industrial_predicts
        context.data_source_splitter_build = build_industrial
        context.data_split__split_any = split_any_industrial
        context.data_split__split_time_series = split_time_series_industrial
        context.operation__predict = predict_operation_industrial
        context.operation_predict = predict_industrial
        context.operation_predict_for_fit = predict_for_fit_industrial
        context.lagged__update_column_types = update_column_types_industrial
        context.lagged_transform = transform_lagged_industrial
        context.lagged_transform_for_fit = transform_lagged_for_fit_industrial
        context.lagged__check_and_correct_window_size = _check_and_correct_window_size_industrial
        context.topo_features_fit = fit_topo_extractor_industrial
        context.topo_features_transform = transform_topo_extractor_industrial
        context.ts_smoothing_transform = transform_smoothing_industrial
        context.api_composer_tune_final_pipeline = tune_pipeline_industrial
        context.reproduction_reproduce = reproduce_industrial
        context.reproduction_reproduce_uncontrolled = reproduce_controlled_industrial
        context.class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)
        context.ts_rules.append(has_no_lagged_conflicts_in_ts_pipeline)

class IndustrialContext(ExecutionContext):
    """Fedot.Industrial execution context"""
    def __init__(self, backend: str = "default") -> None:
        super().__init__()

        self.operation_registry = OperationTypesRepository()
        self.operation_registry.load_operations(IND_DATA_OPERATION_PATH, 'data_operation')
        self.operation_registry.load_operations(IND_MODEL_OPERATION_PATH, 'model')

        self.extension = IndustrialExtension(backend=backend)
        self.extension.apply(self)