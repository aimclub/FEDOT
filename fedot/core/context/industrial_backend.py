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


class IndustrialSplitter:
    def split_any(self, data: InputData, split_ratio: float, shuffle: bool,
                  stratify: bool, random_seed: int, **kwargs):
        return split_any_industrial(data, split_ratio, shuffle, stratify, random_seed, **kwargs)

    def split_time_series(self, data: InputData, validation_blocks: Optional[int] = None, **kwargs):
        return split_time_series_industrial(data, validation_blocks, **kwargs)


class IndustrialDataMerger:
    @staticmethod
    def get(outputs: List[OutputData]):
        return get_merger_industrial(outputs)

    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        return merge_industrial_predicts(predicts)

    @staticmethod
    def find_main_output(outputs: List[OutputData]) -> OutputData:
        return find_main_output_industrial(outputs)


class IndustrialImageMerger:
    def preprocess_predicts(self, predicts: List[np.ndarray]) -> List[np.ndarray]:
        return image_preprocess(predicts)

    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        return merge_industrial_predicts(predicts)


class IndustrialTSMerger:
    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        return merge_industrial_predicts(predicts)

    def merge_targets(self, targets: List[np.ndarray]) -> np.ndarray:
        return merge_industrial_targets(targets)

    def preprocess_predicts(self, predicts: List[np.ndarray]) -> List[np.ndarray]:
        return image_preprocess(predicts)  # или своя ts_preprocess

    def postprocess_predicts(self, merged: np.ndarray) -> np.ndarray:
        return postprocess_industrial_predicts(merged)


class IndustrialTextMerger:
    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        return merge_industrial_predicts(predicts)

    def postprocess_predicts(self, merged: np.ndarray) -> np.ndarray:
        return merged


class IndustrialDataSourceSplitterBuilder:
    def build(self, data: Union[InputData, 'MultiModalData']):
        return build_industrial(data)


class IndustrialTunerClass:
    def __init__(self, backend: str = "default"):
        self.backend = backend

    def __call__(self, objective_evaluate, task, iterations, max_lead_time=None, **kwargs):
        if "dask" in self.backend:
            return DaskOptunaTuner(objective_evaluate, task, iterations, max_lead_time, **kwargs)
        else:
            return OptunaTuner(objective_evaluate, task, iterations, max_lead_time, **kwargs)


class IndustrialReproduction:
    def reproduce(self, population, evaluator, **kwargs):
        return reproduce_industrial(population, evaluator, **kwargs)

    def reproduce_uncontrolled(self, population, **kwargs):
        return reproduce_controlled_industrial(population, **kwargs)


class IndustrialEvaluator:
    def evaluate(self, graph: Pipeline) -> Fitness:
        return industrial_evaluate_pipeline(graph)


class IndustrialSearchSpace:
    def get_parameters_dict(self):
        return get_industrial_search_space()


class IndustrialDefaultMutations:
    @staticmethod
    def __call__(task_type: TaskTypesEnum, params):
        return _get_default_industrial_mutations(task_type, params)


class IndustrialOperationPredict:
    def predict(self, fitted_operation, data: InputData, params=None, output_mode='default'):
        return predict_industrial(fitted_operation, data, params, output_mode)

    def predict_for_fit(self, fitted_operation, data: InputData, params=None, output_mode='default'):
        return predict_for_fit_industrial(fitted_operation, data, params, output_mode)

    def _predict(self, fitted_operation, data: InputData, params=None, output_mode='default',
                 is_fit_stage=False, predictions_cache=None, fold_id=None, descriptive_id=None):
        return predict_operation_industrial(fitted_operation, data, params, output_mode,
                                            is_fit_stage, predictions_cache, fold_id, descriptive_id)


class IndustrialLaggedTransformer:
    def _update_column_types(self, output_data: OutputData):
        update_column_types_industrial(output_data)

    def transform(self, input_data: InputData) -> OutputData:
        return transform_lagged_industrial(input_data)

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        return transform_lagged_for_fit_industrial(input_data)

    def _check_and_correct_window_size(self, time_series: np.ndarray, forecast_length: int):
        _check_and_correct_window_size_industrial(time_series, forecast_length)


class IndustrialTopologicalFeatures:
    def fit(self, input_data: InputData):
        return fit_topo_extractor_industrial(input_data)

    def transform(self, input_data: InputData) -> np.ndarray:
        return transform_topo_extractor_industrial(input_data)


class IndustrialTsSmoothing:
    def transform(self, input_data: InputData) -> OutputData:
        return transform_smoothing_industrial(input_data)


class IndustrialApiComposerTune:
    def tune_pipeline_industrial(self, train_data: InputData, pipeline: Pipeline, execution_plan=None) -> Pipeline:
        return tune_pipeline_industrial(train_data, pipeline, execution_plan)