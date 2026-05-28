from typing import List, Optional, Union, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from fedot.core.data.data import InputData, OutputData
    from fedot.core.data.multi_modal import MultiModalData
    from fedot.core.pipelines.pipeline import Pipeline
    from golem.core.optimisers.fitness import Fitness
    from golem.core.tuning.optuna_tuner import OptunaTuner
    from fedot.core.repository.tasks import TaskTypesEnum
    from fedot.core.pipelines.tuning.tuner import BaseTuner

from fedot.core.protocols.protocols import (
    SplitterProtocol,
    DataMergerProtocol,
    ImageMergerProtocol,
    TSMergerProtocol,
    TextMergerProtocol,
    DataSourceSplitterProtocol,
    TunerClassProtocol,
    ReproductionProtocol,
    EvaluatorProtocol,
    SearchSpaceProtocol,
    DefaultMutationsProtocol,
    OperationPredictProtocol,
    LaggedTransformerProtocol,
    TopologicalFeaturesProtocol,
    TsSmoothingProtocol,
    ApiComposerTuneProtocol,
)


class IndustrialSplitter(SplitterProtocol):
    def split_any(self, data: 'InputData', split_ratio: float, shuffle: bool,
                  stratify: bool, random_seed: int, **kwargs):
        from fedot.industrial.core.repository.industrial_implementations.abstract import split_any_industrial
        return split_any_industrial(data, split_ratio, shuffle, stratify, random_seed, **kwargs)

    def split_time_series(self, data: 'InputData', validation_blocks: Optional[int] = None, **kwargs):
        from fedot.industrial.core.repository.industrial_implementations.abstract import split_time_series_industrial
        return split_time_series_industrial(data, validation_blocks, **kwargs)


class IndustrialDataMerger(DataMergerProtocol):
    @staticmethod
    def get(outputs: List['OutputData']):
        from fedot.industrial.core.repository.industrial_implementations.abstract import get_merger_industrial
        return get_merger_industrial(outputs)

    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        from fedot.industrial.core.repository.industrial_implementations.abstract import merge_industrial_predicts
        return merge_industrial_predicts(predicts)

    @staticmethod
    def find_main_output(outputs: List['OutputData']) -> 'OutputData':
        from fedot.industrial.core.repository.industrial_implementations.abstract import find_main_output_industrial
        return find_main_output_industrial(outputs)


class IndustrialImageMerger(ImageMergerProtocol):
    def preprocess_predicts(self, predicts: List[np.ndarray]) -> List[np.ndarray]:
        from fedot.industrial.core.repository.industrial_implementations.abstract import preprocess_industrial_predicts
        return preprocess_industrial_predicts(predicts)


class IndustrialTSMerger(TSMergerProtocol):

    def postprocess_predicts(self, merged: np.ndarray) -> np.ndarray:
        from fedot.industrial.core.repository.industrial_implementations.abstract import postprocess_industrial_predicts
        return postprocess_industrial_predicts(merged)


class IndustrialTextMerger(TextMergerProtocol):
    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        from fedot.industrial.core.repository.industrial_implementations.abstract import merge_industrial_predicts
        return merge_industrial_predicts(predicts)


class IndustrialDataSourceSplitterBuilder(DataSourceSplitterProtocol):
    def build(self, data: Union['InputData', 'MultiModalData']):
        from fedot.industrial.core.repository.industrial_implementations.abstract import build_industrial
        return build_industrial(data)


class IndustrialTunerClass(TunerClassProtocol):
    def __init__(self, **kwargs):
        self.backend = kwargs.get("backend", "default")

    def optuna_tuner(self, objective_evaluate, task, iterations, max_lead_time=None, **kwargs):
        from golem.core.tuning.optuna_tuner import OptunaTuner
        from fedot.industrial.core.repository.industrial_implementations.ml_optimisation import DaskOptunaTuner

        if "dask" in self.backend:
            return DaskOptunaTuner(objective_evaluate, task, iterations, max_lead_time, **kwargs)
        else:
            return OptunaTuner(objective_evaluate, task, iterations, max_lead_time, **kwargs)


class IndustrialReproduction(ReproductionProtocol):
    def reproduce(self, population, evaluator, **kwargs):
        from fedot.industrial.core.repository.industrial_implementations.optimisation import reproduce_industrial
        return reproduce_industrial(population, evaluator, **kwargs)

    def reproduce_uncontrolled(self, population, **kwargs):
        from fedot.industrial.core.repository.industrial_implementations.optimisation import \
            reproduce_controlled_industrial
        return reproduce_controlled_industrial(population, **kwargs)


class IndustrialEvaluator(EvaluatorProtocol):
    def evaluate(self, graph: 'Pipeline') -> 'Fitness':
        from fedot.industrial.core.metrics.pipeline import industrial_evaluate_pipeline
        return industrial_evaluate_pipeline(graph)


class IndustrialSearchSpace(SearchSpaceProtocol):
    def get_parameters_dict(self):
        from fedot.industrial.core.tuning.search_space import get_industrial_search_space
        return get_industrial_search_space()


class IndustrialDefaultMutations(DefaultMutationsProtocol):
    @staticmethod
    def get_default_mutations(task_type: 'TaskTypesEnum', params):
        from fedot.industrial.core.repository.industrial_implementations.optimisation import \
            _get_default_industrial_mutations
        return _get_default_industrial_mutations(task_type, params)


class IndustrialOperationPredict(OperationPredictProtocol):
    def predict(self, fitted_operation, data: 'InputData', params=None, output_mode='default'):
        from fedot.industrial.core.repository.industrial_implementations.abstract import predict_industrial
        return predict_industrial(fitted_operation, data, params, output_mode)

    def predict_for_fit(self, fitted_operation, data: 'InputData', params=None, output_mode='default'):
        from fedot.industrial.core.repository.industrial_implementations.abstract import predict_for_fit_industrial
        return predict_for_fit_industrial(fitted_operation, data, params, output_mode)

    def _predict(self, fitted_operation, data: 'InputData', params=None, output_mode='default',
                 is_fit_stage=False, predictions_cache=None, fold_id=None, descriptive_id=None):
        from fedot.industrial.core.repository.industrial_implementations.abstract import predict_operation_industrial
        return predict_operation_industrial(fitted_operation, data, params, output_mode,
                                            is_fit_stage, predictions_cache, fold_id, descriptive_id)


class IndustrialLaggedTransformer(LaggedTransformerProtocol):
    def _update_column_types(self, output_data: 'OutputData'):
        from fedot.industrial.core.repository.industrial_implementations.data_transformation import \
            update_column_types_industrial
        return update_column_types_industrial(output_data)

    def transform(self, input_data: 'InputData') -> 'OutputData':
        from fedot.industrial.core.repository.industrial_implementations.data_transformation import \
            transform_lagged_industrial
        return transform_lagged_industrial(input_data)

    def transform_for_fit(self, input_data: 'InputData') -> 'OutputData':
        from fedot.industrial.core.repository.industrial_implementations.data_transformation import \
            transform_lagged_for_fit_industrial
        return transform_lagged_for_fit_industrial(input_data)

    def _check_and_correct_window_size(self, time_series: np.ndarray, forecast_length: int):
        from fedot.industrial.core.repository.industrial_implementations.data_transformation import \
            _check_and_correct_window_size_industrial
        return _check_and_correct_window_size_industrial(time_series, forecast_length)


class IndustrialTopologicalFeatures(TopologicalFeaturesProtocol):
    def fit(self, input_data: 'InputData'):
        from fedot.industrial.core.repository.industrial_implementations.abstract import fit_topo_extractor_industrial
        return fit_topo_extractor_industrial(input_data)

    def transform(self, input_data: 'InputData') -> np.ndarray:
        from fedot.industrial.core.repository.industrial_implementations.abstract import \
            transform_topo_extractor_industrial
        return transform_topo_extractor_industrial(input_data)


class IndustrialTsSmoothing(TsSmoothingProtocol):
    def transform(self, input_data: 'InputData') -> 'OutputData':
        from fedot.industrial.core.repository.industrial_implementations.data_transformation import \
            transform_smoothing_industrial
        return transform_smoothing_industrial(input_data)


class IndustrialApiComposerTune(ApiComposerTuneProtocol):
    def tune_pipeline(self, train_data: 'InputData', pipeline: 'Pipeline', execution_plan=None) -> 'Pipeline':
        from fedot.industrial.core.repository.industrial_implementations.ml_optimisation import tune_pipeline_industrial
        return tune_pipeline_industrial(train_data, pipeline, execution_plan)
