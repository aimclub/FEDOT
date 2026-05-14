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


class CoreEvaluator(EvaluatorProtocol):
    def evaluate(self, graph):
        from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
        # from golem.core.optimisers.fitness import Fitness
        return PipelineObjectiveEvaluate.evaluate(graph)


class CoreSearchSpace(SearchSpaceProtocol):
    def get_parameters_dict(self) -> dict:
        from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
        return PipelineSearchSpace.get_parameters_dict()


class CoreDefaultMutations(DefaultMutationsProtocol):
    @staticmethod
    def __call__(task_type, params):
        from fedot.api.api_utils.api_params_repository import ApiParamsRepository
        # from typing import Sequence
        return ApiParamsRepository._get_default_mutations(task_type, params)


class CoreDataMerger(DataMergerProtocol):
    @staticmethod
    def get(outputs):
        from fedot.core.data.merge.data_merger import DataMerger
        return DataMerger.get(outputs)

    def merge_predicts(self, predicts):
        from fedot.core.data.merge.data_merger import DataMerger
        return DataMerger.merge_predicts(predicts)

    @staticmethod
    def find_main_output(outputs):
        from fedot.core.data.merge.data_merger import DataMerger
        return DataMerger.find_main_output(outputs)

    def preprocess_predicts(self, predicts):
        return predicts

    def postprocess_predicts(self, merged):
        return merged


class CoreImageMerger(ImageMergerProtocol):
    def preprocess_predicts(self, predicts):
        from fedot.core.data.merge.data_merger import ImageDataMerger
        return ImageDataMerger.preprocess_predicts(predicts)

    def merge_predicts(self, predicts):
        from fedot.core.data.merge.data_merger import ImageDataMerger
        return ImageDataMerger.merge_predicts(predicts)


class CoreTSMerger(TSMergerProtocol):
    def merge_predicts(self, predicts):
        from fedot.core.data.merge.data_merger import TSDataMerger
        return TSDataMerger.merge_predicts(predicts)

    def merge_targets(self, targets):
        from fedot.core.data.merge.data_merger import TSDataMerger
        return TSDataMerger.merge_targets(targets)

    def preprocess_predicts(self, predicts):
        from fedot.core.data.merge.data_merger import TSDataMerger
        return TSDataMerger.preprocess_predicts(predicts)

    def postprocess_predicts(self, merged):
        from fedot.core.data.merge.data_merger import TSDataMerger
        return TSDataMerger.postprocess_predicts(merged)


class CoreTextMerger(TextMergerProtocol):
    def merge_predicts(self, predicts):
        from fedot.core.data.merge.data_merger import TextDataMerger
        return TextDataMerger.merge_predicts(predicts)

    def postprocess_predicts(self, merged):
        return merged


class CoreDataSourceSplitter(DataSourceSplitterProtocol):
    def build(self, data):
        from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
        return DataSourceSplitter.build(data)


class CoreSplitter(SplitterProtocol):
    def split_any(self, data, split_ratio, shuffle, stratify, random_seed, **kwargs):
        from fedot.core.data.data_split import _split_any
        return _split_any(data, split_ratio, shuffle, stratify, random_seed, **kwargs)

    def split_time_series(self, data, validation_blocks=None, **kwargs):
        from fedot.core.data.data_split import _split_time_series
        return _split_time_series(data, validation_blocks, **kwargs)


class CoreOperationPredict(OperationPredictProtocol):
    def predict(self, fitted_operation, data, params=None, output_mode='default'):
        from fedot.core.operations.operation import Operation
        return Operation.predict(fitted_operation, data, params, output_mode)

    def predict_for_fit(self, fitted_operation, data, params=None, output_mode='default'):
        from fedot.core.operations.operation import Operation
        return Operation.predict_for_fit(fitted_operation, data, params, output_mode)

    def _predict(self, fitted_operation, data, params=None, output_mode='default',
                 is_fit_stage=False, predictions_cache=None, fold_id=None, descriptive_id=None):
        from fedot.core.operations.operation import Operation
        return Operation._predict(fitted_operation, data, params, output_mode,
                                  is_fit_stage, predictions_cache, fold_id, descriptive_id)


class CoreLaggedTransformer(LaggedTransformerProtocol):
    def _update_column_types(self, output_data):
        from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
            LaggedImplementation
        )
        return LaggedImplementation._update_column_types(output_data)

    def transform(self, input_data):
        from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
            LaggedImplementation
        )
        return LaggedImplementation.transform(input_data)

    def transform_for_fit(self, input_data):
        from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
            LaggedImplementation
        )
        return LaggedImplementation.transform_for_fit(input_data)

    def _check_and_correct_window_size(self, time_series, forecast_length):
        from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
            LaggedImplementation
        )
        return LaggedImplementation._check_and_correct_window_size(time_series, forecast_length)


class CoreTopologicalFeatures(TopologicalFeaturesProtocol):
    def fit(self, input_data):
        from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor import (
            TopologicalFeaturesImplementation
        )
        return TopologicalFeaturesImplementation.fit(input_data)

    def transform(self, input_data):
        from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor import (
            TopologicalFeaturesImplementation
        )
        return TopologicalFeaturesImplementation.transform(input_data)


class CoreTsSmoothing(TsSmoothingProtocol):
    def transform(self, input_data):
        from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
            TsSmoothingImplementation
        )
        return TsSmoothingImplementation.transform(input_data)


class CoreTuner(TunerClassProtocol):
    def __init__(self, **kwargs):
        self.backend = kwargs.get("backend", "default")

    def __call__(self, objective_evaluate, task, iterations, max_lead_time=None, **kwargs):
        from golem.core.tuning.optuna_tuner import OptunaTuner, DaskOptunaTuner
        # from fedot.core.pipelines.tuning.tuner import BaseTuner

        if "dask" in self.backend:
            return DaskOptunaTuner(objective_evaluate, task, iterations, max_lead_time, **kwargs)
        return OptunaTuner(objective_evaluate, task, iterations, max_lead_time, **kwargs)


class CoreApiComposerTune(ApiComposerTuneProtocol):
    def __call__(self, train_data, pipeline, execution_plan=None):
        from fedot.api.api_utils.api_composer import ApiComposer
        return ApiComposer.tune_final_pipeline(train_data, pipeline, execution_plan)


class CoreReproduction(ReproductionProtocol):
    def reproduce(self, population, evaluator, **kwargs):
        from golem.core.optimisers.genetic.operators.reproduction import ReproductionController
        return ReproductionController.reproduce(population, evaluator, **kwargs)

    def reproduce_uncontrolled(self, population, **kwargs):
        from golem.core.optimisers.genetic.operators.reproduction import ReproductionController
        return ReproductionController.reproduce_uncontrolled(population, **kwargs)