import fedot.core.data.split.data_split as fedot_data_split
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

FEDOT_METHOD_TO_REPLACE = [(PipelineObjectiveEvaluate, "evaluate"),
                           (PipelineSearchSpace, "get_parameters_dict"),
                           (ApiParamsRepository, "_get_default_mutations"),
                           (DataMerger, "find_main_output"),
                           (DataMerger, "get"),
                           (DataMerger, "merge_predicts"),
                           (ImageDataMerger, "preprocess_predicts"),
                           (ImageDataMerger, "merge_predicts"),
                           (TSDataMerger, "merge_predicts"),
                           (TSDataMerger, "merge_targets"),
                           (TSDataMerger, 'postprocess_predicts'),
                           (TSDataMerger, 'preprocess_predicts'),
                           (DataSourceSplitter, "build"),
                           (fedot_data_split, "_split_any"),
                           (fedot_data_split, "_split_time_series"),
                           (Operation, "_predict"),
                           (Operation, "predict"),
                           (Operation, "predict_for_fit"),
                           (LaggedImplementation, '_update_column_types'),
                           (LaggedImplementation, 'transform'),
                           (TopologicalFeaturesImplementation, 'fit'),
                           (TopologicalFeaturesImplementation, 'transform'),
                           (LaggedImplementation, 'transform_for_fit'),
                           (LaggedImplementation, '_check_and_correct_window_size'),
                           (TsSmoothingImplementation, 'transform'),
                           (OptunaImpl, 'OptunaTuner'),
                           (ApiComposer, 'tune_final_pipeline'),
                           (ReproductionController, 'reproduce_uncontrolled'),
                           (ReproductionController, 'reproduce')]
INDUSTRIAL_REPLACE_METHODS = [industrial_evaluate_pipeline,
                              get_industrial_search_space,
                              _get_default_industrial_mutations,
                              find_main_output_industrial,
                              get_merger_industrial,
                              merge_industrial_predicts,
                              preprocess_industrial_predicts,
                              merge_industrial_predicts,
                              merge_industrial_predicts,
                              merge_industrial_targets,
                              postprocess_industrial_predicts,
                              preprocess_industrial_predicts,
                              build_industrial,
                              split_any_industrial,
                              split_time_series_industrial,
                              predict_operation_industrial,
                              predict_industrial,
                              predict_for_fit_industrial,
                              update_column_types_industrial,
                              transform_lagged_industrial,
                              fit_topo_extractor_industrial,
                              transform_topo_extractor_industrial,
                              transform_lagged_for_fit_industrial,
                              _check_and_correct_window_size_industrial,
                              transform_smoothing_industrial,
                              DaskOptunaTuner,
                              tune_pipeline_industrial,
                              reproduce_controlled_industrial,
                              reproduce_industrial]

DEFAULT_METHODS = [getattr(class_impl[0], class_impl[1])
                   for class_impl in FEDOT_METHOD_TO_REPLACE]
DEFAULT_MODELS_TO_REPLACE = [(MODEL_REPO, 'SKLEARN_REG_MODELS'),
                             (MODEL_REPO, 'SKLEARN_CLF_MODELS'),
                             (MODEL_REPO, 'FEDOT_PREPROC_MODEL')]


def has_no_resample(pipeline: Pipeline):
    """
    Pipeline can have only one resample operation located in start of the pipeline

    :param pipeline: pipeline for checking
    """
    for node in pipeline.nodes:
        if node.name == 'resample':
            raise ValueError(
                f'Pipeline can not have resample operation')
    return True


class IndustrialModels:
    def __init__(self):

        self.industrial_data_operation_path = IND_DATA_OPERATION_PATH
        self.industrial_model_path = IND_MODEL_OPERATION_PATH

        self.base_data_operation_path = DEFAULT_DATA_OPERATION_PATH
        self.base_model_path = DEFAULT_MODEL_OPERATION_PATH

    def _replace_operation(self, to_industrial=True, backend: str = 'default'):
        method = INDUSTRIAL_REPLACE_METHODS if to_industrial else DEFAULT_METHODS
        for class_impl, method_to_replace in zip(FEDOT_METHOD_TO_REPLACE, method):
            setattr(class_impl[0], class_impl[1], method_to_replace)
        if backend.__contains__('dask'):
            model_to_overload = [SKLEARN_REG_MODELS, SKLEARN_CLF_MODELS, FEDOT_PREPROC_MODEL]
            overloaded_model = overload_model_implementation(model_to_overload, backend=backend)
            for model_impl, new_backend_impl in zip(DEFAULT_MODELS_TO_REPLACE, overloaded_model):
                setattr(model_impl[0], model_impl[1], new_backend_impl)

    def setup_repository(self, backend: str = 'default'):
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.industrial_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.industrial_model_path)
        # replace mutations
        self._replace_operation(to_industrial=True, backend=backend)

        class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)
        ts_rules.append(has_no_lagged_conflicts_in_ts_pipeline)
        return OperationTypesRepository

    def setup_default_repository(self, backend: str = 'default'):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.base_data_operation_path,
                                'initialized_repo': None,
                                'default_tags': [
                                    OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]}})
        OperationTypesRepository.assign_repo(
            'data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.base_model_path,
                       'initialized_repo': None,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.base_model_path)
        self._replace_operation(to_industrial=False, backend=backend)
        common_rules.append(has_no_resample)
        return OperationTypesRepository

    def __enter__(self):
        """
        Switching to industrial models
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.industrial_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.industrial_model_path)

        setattr(PipelineSearchSpace, "get_parameters_dict",
                get_industrial_search_space)
        setattr(ApiComposer, "_get_default_mutations",
                _get_default_industrial_mutations)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.base_data_operation_path,
                                'initialized_repo': None,
                                'default_tags': [
                                    OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]}})
        OperationTypesRepository.assign_repo(
            'data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.base_model_path,
                       'initialized_repo': None,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.base_model_path)
