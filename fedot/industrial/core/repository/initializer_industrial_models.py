import fedot.core.data.data_split as fedot_data_split
import golem.core.tuning.optuna_tuner as OptunaImpl
from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.data.merge.data_merger import ImageDataMerger, TSDataMerger, DataMerger
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor \
    import TopologicalFeaturesImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    LaggedImplementation, TsSmoothingImplementation
from fedot.core.operations.operation import Operation
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.verification import class_rules, ts_rules
from fedot.core.pipelines.verification import common_rules
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from golem.core.optimisers.genetic.operators.reproduction import ReproductionController

import fedot.industrial.core.repository.model_repository as MODEL_REPO
from fedot.industrial.core.metrics.pipeline import industrial_evaluate_pipeline
from fedot.industrial.core.repository.constanst_repository import IND_DATA_OPERATION_PATH, IND_MODEL_OPERATION_PATH, \
    DEFAULT_DATA_OPERATION_PATH, DEFAULT_MODEL_OPERATION_PATH
from fedot.industrial.core.repository.industrial_implementations.abstract import preprocess_industrial_predicts, \
    merge_industrial_predicts, merge_industrial_targets, build_industrial, postprocess_industrial_predicts, \
    split_any_industrial, split_time_series_industrial, predict_operation_industrial, predict_industrial, \
    predict_for_fit_industrial, update_column_types_industrial, \
    fit_topo_extractor_industrial, transform_topo_extractor_industrial, find_main_output_industrial, \
    get_merger_industrial
from fedot.industrial.core.repository.industrial_implementations.data_transformation import transform_lagged_industrial, \
    transform_lagged_for_fit_industrial, _check_and_correct_window_size_industrial, transform_smoothing_industrial
from fedot.industrial.core.repository.industrial_implementations.ml_optimisation import DaskOptunaTuner, \
    tune_pipeline_industrial
from fedot.industrial.core.repository.industrial_implementations.optimisation import _get_default_industrial_mutations, \
    has_no_lagged_conflicts_in_ts_pipeline, reproduce_controlled_industrial, reproduce_industrial
from fedot.industrial.core.repository.industrial_implementations.optimisation import \
    has_no_data_flow_conflicts_in_industrial_pipeline
from fedot.industrial.core.repository.model_repository import SKLEARN_REG_MODELS, SKLEARN_CLF_MODELS, FEDOT_PREPROC_MODEL
from fedot.industrial.core.repository.model_repository import overload_model_implementation
from fedot.industrial.core.tuning.search_space import get_industrial_search_space

from fedot.core.context import ExecutionContext
from fedot.industrial.industrial_extension import IndustrialExtension


def has_no_resample(pipeline: Pipeline):
    """
    Pipeline can have only one resample operation located in start of the pipeline

    :param pipeline: pipeline for checking
    """
    for node in pipeline.nodes:
        if node.name == 'resample':
            raise ValueError("Pipeline can not have resample operation")
    return True


def initialize_industrial_context(backend: str = "default") -> ExecutionContext:
    context = ExecutionContext()
    extension = IndustrialExtension(backend=backend)
    extension.apply(context)
    return context


class IndustrialModels:
    def __init__(self, backend: str = "default"):
        self.industrial_data_operation_path = IND_DATA_OPERATION_PATH
        self.industrial_model_path = IND_MODEL_OPERATION_PATH
        self.base_data_operation_path = DEFAULT_DATA_OPERATION_PATH
        self.base_model_path = DEFAULT_MODEL_OPERATION_PATH

        self.backend = backend
        self.extension = IndustrialExtension(backend=backend)
        self.context: ExecutionContext | None = None

    def get_industrial_context(self) -> ExecutionContext:
        self.context = ExecutionContext()
        self.extension.apply(self.context)
        return self.context

    def setup_repository(self) -> OperationTypesRepository:
        OperationTypesRepository.__repository_dict__.update({
            'data_operation': {
                'file': self.industrial_data_operation_path,
                'initialized_repo': True,
                'default_tags': []
            }
        })
        OperationTypesRepository.assign_repo('data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update({
            'model': {
                'file': self.industrial_model_path,
                'initialized_repo': True,
                'default_tags': []
            }
        })
        OperationTypesRepository.assign_repo('model', self.industrial_model_path)

        self.get_industrial_context()
        self.context.class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)
        self.context.ts_rules.append(has_no_lagged_conflicts_in_ts_pipeline)

        return OperationTypesRepository

    def setup_default_repository(self) -> OperationTypesRepository:
        OperationTypesRepository.__repository_dict__.update({
            'data_operation': {
                'file': self.base_data_operation_path,
                'initialized_repo': None,
                'default_tags': [OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]
            }
        })
        OperationTypesRepository.assign_repo('data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update({
            'model': {
                'file': self.base_model_path,
                'initialized_repo': None,
                'default_tags': []
            }
        })
        OperationTypesRepository.assign_repo('model', self.base_model_path)
        self.context = ExecutionContext()
        self.context.common_rules.append(has_no_resample)

        return OperationTypesRepository

    def __enter__(self) -> ExecutionContext:
        self.setup_repository()
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.setup_default_repository()
        self.context = None
