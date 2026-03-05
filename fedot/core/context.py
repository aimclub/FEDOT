from fedot.core.data.merge.data_merger import ImageDataMerger, TSDataMerger, DataMerger
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor import (
    TopologicalFeaturesImplementation,
)
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


class ExecutionContext:
    def __init__(self) -> None:
        self.operation_registry = OperationTypesRepository()
        self.evaluator_evaluate = PipelineObjectiveEvaluate
        self.search_space_get_parameters_dict = PipelineSearchSpace
        self.api_params_repository__get_default_mutations = ApiParamsRepository
        self.merger_find_main_output = DataMerger
        self.merger_get = DataMerger
        self.merger_merge_predicts = DataMerger
        self.image_merger_preprocess_predicts = ImageDataMerger
        self.image_merger_merge_predicts = ImageDataMerger
        self.ts_merger_merge_predicts = TSDataMerger
        self.ts_merger_merge_targets = TSDataMerger
        self.ts_merger_postprocess_predicts = TSDataMerger
        self.ts_merger_preprocess_predicts = TSDataMerger
        self.data_source_splitter_build = DataSourceSplitter
        self.data_split__split_any = fedot_data_split._split_any
        self.data_split__split_time_series = fedot_data_split._split_time_series
        self.operation__predict = Operation
        self.operation_predict = Operation
        self.operation_predict_for_fit = Operation
        self.lagged__update_column_types = LaggedImplementation
        self.lagged_transform = LaggedImplementation
        self.lagged_transform_for_fit = LaggedImplementation
        self.lagged__check_and_correct_window_size = LaggedImplementation
        self.topo_features_fit = TopologicalFeaturesImplementation
        self.topo_features_transform = TopologicalFeaturesImplementation
        self.ts_smoothing_transform = TsSmoothingImplementation
        self.optuna_optuna_tuner = OptunaImpl.OptunaTuner
        self.api_composer_tune_final_pipeline = ApiComposer
        self.reproduction_reproduce = ReproductionController
        self.reproduction_reproduce_uncontrolled = ReproductionController
        self.class_rules = class_rules.copy()
        self.ts_rules = ts_rules.copy()
        self.common_rules = common_rules.copy()

    def __getstate__(self):
        return {
            "industrial": True,
            "class_rules": self.class_rules,
            "ts_rules": self.ts_rules,
            "common_rules": self.common_rules,
        }

    def __setstate__(self, state):
        self.__init__()
        self.class_rules = state["class_rules"]
        self.ts_rules = state["ts_rules"]
        self.common_rules = state["common_rules"]