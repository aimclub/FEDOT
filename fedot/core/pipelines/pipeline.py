from copy import deepcopy
from datetime import timedelta
from os import PathLike
from typing import Dict, List, Optional, Sequence, Tuple, Union

import func_timeout
from golem.core.dag.graph import Graph
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import distance_to_primary_level, graph_structure
from golem.core.dag.linked_graph import LinkedGraph
from golem.core.log import default_log
from golem.core.optimisers.timer import Timer
from golem.core.paths import copy_doc
from golem.utilities.serializable import Serializable
from golem.visualisation.graph_viz import NodeColorType

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.bridges.tensor_to_input import tensordata_to_input_data
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_rules import (
    build_pipeline_postprocess_plan,
    build_pipeline_preprocess_plan
)
from fedot.core.pipelines.schemas import (
    validate_pipeline_is_fitted,
    validate_single_root_node,
)
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from fedot.preprocessing.dummy_preprocessing import DummyPreprocessor
from fedot.preprocessing.preprocessing import DataPreprocessor
from fedot.utilities.composer_timer import fedot_composer_timer

ERROR_PREFIX = 'Invalid pipeline configuration:'


class Pipeline(GraphDelegate, Serializable):
    """Base class used for composite model structure definition

    Args:
        nodes: :obj:`PipelineNode` object(s)
        use_input_preprocessing: whether to do input preprocessing or not, ``True`` by default.
    """

    def __init__(self, nodes: Union[PipelineNode, Sequence[PipelineNode]] = (), use_input_preprocessing: bool = True):
        super().__init__(nodes, _graph_nodes_to_pipeline_nodes)

        self.computation_time = None
        self.log = default_log(self)

        # Used externally, outside of this class
        self.use_input_preprocessing = use_input_preprocessing
        # Define data preprocessor
        self.preprocessor = DataPreprocessor(
        ) if use_input_preprocessing else DummyPreprocessor()

    def fit_from_scratch(self, tensor_data: TensorData = None):
        """[Obsolete] Method used for training the pipeline without using saved information

        Args:
            input_data: data used for operation training
        """

        # Clean all saved states and fit all operations
        self.unfit()
        self.fit(tensor_data)

    def _fit_with_time_limit(self,
                                        tensor_data: Optional[TensorData],
                                        time: timedelta,
                                        predictions_cache: Optional[PredictionsCache] = None,
                                        fold_id: Optional[int] = None) -> TensorData:
        """Runs TensorData training process in all pipeline nodes with time limit."""
        time = int(time.total_seconds())
        process_state_dict = {}
        fitted_operations = []
        try:
            func_timeout.func_timeout(
                time, self._fit,
                args=(tensor_data, process_state_dict,
                      fitted_operations, predictions_cache, fold_id)
            )
        except func_timeout.FunctionTimedOut:
            raise TimeoutError(
                f'Pipeline fitness evaluation time limit is expired (more then {time} seconds)')

        self.computation_time = process_state_dict['computation_time_in_seconds']
        for node_num, _ in enumerate(self.nodes):
            self.nodes[node_num].fitted_operation = fitted_operations[node_num]
        return process_state_dict['train_predicted']
    
    # TODO romankuklo: add preprocessing after new features creating

    def _fit(self,
                        tensor_data: Optional[TensorData] = None,
                        process_state_dict: dict = None,
                        fitted_operations: list = None,
                        predictions_cache: Optional[PredictionsCache] = None,
                        fold_id: Optional[int] = None) -> Optional[TensorData]:
        """Runs training process in all the pipeline nodes starting with root on TensorData."""
        with Timer() as t:
            computation_time_update = not self.root_node.fitted_operation or self.computation_time is None
            train_predicted = self.root_node.fit(
                tensor_data=tensor_data,
                predictions_cache=predictions_cache,
                fold_id=fold_id,
            )
            if computation_time_update:
                self.computation_time = round(t.minutes_from_start, 3)

        if process_state_dict is None:
            return train_predicted
        else:
            process_state_dict['train_predicted'] = train_predicted
            process_state_dict['computation_time_in_seconds'] = self.computation_time
            for node in self.nodes:
                fitted_operations.append(node.fitted_operation)

    # TODO romankuklo: refactor this method to use tensordata
    def _postprocess(self, copied_input_data: Optional[InputData], result: OutputData,
                     output_mode: str = 'default') -> OutputData:
        """
        Postprocesses output of the model

        Args:
            copied_input_data: preprocessed copy of the original data
            result: output of the model
            output_mode: desired form of output for operations

        Returns:
            OutputData: postprocessed ``result`` parameter
        """
        postprocess_plan = build_pipeline_postprocess_plan(
            output_mode, result.task.task_type)
        result = self.preprocessor.restore_index(copied_input_data, result)
        if postprocess_plan.should_restore_inverse_target_encoding:
            result.predict = self.preprocessor.apply_inverse_target_encoding(
                result.predict)
        if postprocess_plan.should_flatten_prediction:
            result.predict = result.predict.ravel()
        return result

    def fit(self,
                       tensor_data: TensorData,
                       time_constraint: Optional[timedelta] = None,
                       n_jobs: int = 1,
                       predictions_cache: Optional[PredictionsCache] = None,
                       fold_id: Optional[int] = None) -> TensorData:
        self.replace_n_jobs_in_nodes(n_jobs)

        copied_tensor_data = deepcopy(tensor_data)
        copied_tensor_data = self._assign_data_to_nodes(copied_tensor_data)

        if time_constraint is None:
            return self._fit(
                tensor_data=tensor_data,
                predictions_cache=predictions_cache,
                fold_id=fold_id,
            )
        return self._fit_with_time_limit(
            tensor_data=tensor_data,
            time=time_constraint,
            predictions_cache=predictions_cache,
            fold_id=fold_id,
        )

    @property
    def is_fitted(self) -> bool:
        """Property showing whether pipeline is fitted

        Returns:
            flag showing if all the pipeline nodes are already fitted
        """

        return all(node.fitted_operation is not None for node in self.nodes)

    def unfit(self, mode='all', unfit_preprocessor: bool = True):
        """Removes fitted operations for chosen type of nodes.

        Args:
            mode: the name of mode

                .. details:: possible ``mode`` options:

                        - ``all`` -> (default) All models will be unfitted
                        - ``data_operations`` -> All data operations will be unfitted

            unfit_preprocessor: should we unfit preprocessor
        """

        for node in self.nodes:
            if mode == 'all' or (mode == 'data_operations' and isinstance(node.content['name'], DataOperation)):
                node.unfit()

        if unfit_preprocessor:
            self.unfit_preprocessor()

    def unfit_preprocessor(self):
        self.preprocessor = type(self.preprocessor)()

    def sync_preprocessing_mode(self, use_input_preprocessing: bool):
        """Synchronizes input preprocessing mode with the parent entities

            Args:
                use_input_preprocessing: whether to do input preprocessing or not.
        """

        if use_input_preprocessing != self.use_input_preprocessing:
            self.use_input_preprocessing = use_input_preprocessing
            self.preprocessor = DataPreprocessor(
            ) if use_input_preprocessing else DummyPreprocessor()

    def try_load_from_cache(
            self,
            operations_cache: Optional[OperationsCache] = None,
            preprocessing_cache: Optional[PreprocessingCache] = None,
            fold_id: Optional[int] = None):
        """
        Tries to load pipeline nodes if ``cache`` is provided

        Args:
            cache: pipeline nodes cacher
            fold_id: optional part of the cache item UID
               (can be used to specify the number of CV fold)

        Returns:
            bool: indicating if at least one node was loaded
        """

        if operations_cache is not None:
            operations_cache.try_load_into_pipeline(self, fold_id)
        if preprocessing_cache is not None:
            preprocessing_cache.try_load_preprocessor(self, fold_id)

    def predict(self,
                           tensor_data: TensorData,
                           output_mode: str = 'default',
                           predictions_cache: Optional[PredictionsCache] = None,
                           fold_id: Optional[int] = None) -> TensorData:
        validate_pipeline_is_fitted(self.is_fitted)

        output_mode = output_mode if output_mode is not None else 'default'
        
        copied_tensor_data = deepcopy(tensor_data)

        copied_tensor_data = self._assign_data_to_nodes(copied_tensor_data)
        result = self.root_node.predict(
            tensor_data=copied_tensor_data,
            output_mode=output_mode,
            predictions_cache=predictions_cache,
            fold_id=fold_id,
        )
        # TODO romankuklo: add postprocess for tensor data
        result = self._postprocess(copied_tensor_data, result, output_mode)
        return result

    def save(self, path: str = None, create_subdir: bool = True, is_datetime_in_path: bool = False) -> Tuple[str, dict]:
        """
        Saves the pipeline to JSON representation with pickled fitted operations

        Args:
            path: custom path to dir where to save JSON or name of json file where to save pipeline.
            If only file name is specified, then absolute path to this file will be created.
            create_subdir: if True -- create one more dir in the last dir
                           if False -- save to the last dir in specified path
            is_datetime_in_path: is it required to add the datetime timestamp to the path

        Returns:
            Tuple[str, dict]: :obj:`JSON representation of the pipeline structure`,
            :obj:`dict of paths to fitted models`
        """

        template = PipelineTemplate(self)
        json_object, dict_fitted_operations = template.export_pipeline(path, root_node=self.root_node,
                                                                       create_subdir=create_subdir,
                                                                       is_datetime_in_path=is_datetime_in_path)
        return json_object, dict_fitted_operations

    def load(self, source: Union[str, dict], dict_fitted_operations: Optional[dict] = None):
        """Loads the pipeline ``JSON`` representation with pickled fitted operations.

        Args:
            source: where to load the pipeline from
            dict_fitted_operations: dictionary of the fitted operations
        """

        self.nodes: Optional[List[PipelineNode]] = []
        template = PipelineTemplate(self)
        template.import_pipeline(source, dict_fitted_operations)
        return self

    @property
    def root_node(self) -> Optional[PipelineNode]:
        """Finds pipelines sink-node

        Returns:
            the final predictor-node
        """
        if not self.nodes:
            return None
        root = [node for node in self.nodes
                if not any(self.node_children(node))]
        validate_single_root_node(len(root))
        return root[0]

    @property
    def primary_nodes(self) -> List[PipelineNode]:
        """Finds pipeline's primary nodes

        Returns:
            list of primary nodes
        """
        if not self.nodes:
            return []
        primary_nodes = [node for node in self.nodes
                         if node.is_primary]
        return primary_nodes

    @property
    def nodes(self) -> List[PipelineNode]:
        return self.operator.nodes

    @nodes.setter
    def nodes(self, new_nodes: List[PipelineNode]) -> None:
        self.operator.nodes = new_nodes

    def pipeline_for_side_task(self, task_type: TaskTypesEnum) -> 'Pipeline':
        """Returns pipeline formed from the last node solving the given problem and all its parents

        Args:
            task_type: task type of the last node to search for

        Returns:
            pipeline formed from the last node solving the given problem and all of its parents
        """

        max_distance = 0
        side_root_node = None
        for node in self.nodes:
            if (task_type in node.operation.acceptable_task_types and
                    isinstance(node.operation, Model) and
                    distance_to_primary_level(node) >= max_distance):
                side_root_node = node
                max_distance = distance_to_primary_level(node)

        pipeline = Pipeline(side_root_node)
        pipeline.preprocessor = self.preprocessor
        return pipeline

    def _assign_data_to_nodes(self, input_data: Union[InputData, MultiModalData, TensorData]
                              ) -> Optional[Union[InputData, TensorData]]:
        """In case of provided ``input_data`` is of type :class:`MultiModalData`
        assigns :attr:`PipelineNode.node_data` from the ``input_data`` if ``PipelineNode.nodes_from`` is None

        Args:
            input_data: data to assign to :attr:`PipelineNode.node_data`

        Returns:
            ``None`` in case of :class:`MultiModalData` and ``input_data`` otherwise
        """

        if isinstance(input_data, MultiModalData):
            for node in (n for n in self.nodes if (isinstance(n, PipelineNode) and n.is_primary)):
                if node.operation.operation_type in input_data:
                    node.node_data = input_data[node.operation.operation_type]
                    node.direct_set = True
                else:
                    raise ValueError(f'No data for primary node {node}')
            return None
        return input_data

    def print_structure(self):
        """ Prints structure of the pipeline
        """

        print(
            'Pipeline structure:',
            graph_structure(self),
            sep='\n'
        )

    def replace_n_jobs_in_nodes(self, n_jobs: int):
        """
        Changes number of jobs for nodes

        :param n_jobs: required number of the jobs to assign to the nodes
        """
        for node in self.nodes:
            for param in ['n_jobs', 'num_threads']:
                if param in node.content['params']:
                    node.content['params'][param] = n_jobs

    @copy_doc(Graph.show)
    def show(self, save_path: Optional[Union[PathLike, str]] = None, engine: Optional[str] = None,
             node_color: Optional[NodeColorType] = None, dpi: Optional[int] = None,
             node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
             edge_curvature_scale: Optional[float] = None,
             nodes_labels: Dict[int, str] = None, edges_labels: Dict[int, str] = None):
        PipelineVisualizer(self).visualise(save_path=save_path, engine=engine, node_color=node_color,
                                           dpi=dpi, node_size_scale=node_size_scale, font_size_scale=font_size_scale,
                                           nodes_labels=nodes_labels, edges_labels=edges_labels)


def _graph_nodes_to_pipeline_nodes(operator: LinkedGraph, nodes: Sequence[PipelineNode]):
    """
    Method to update nodes type after performing some action on the pipeline
        via GraphOperator, if any of them are of GraphNode type

    Args:
        nodes: :obj:`PipelineNode` object(s)
    """

    for node in nodes:
        if not isinstance(node, GraphNode):
            continue
        if not node.nodes_from and not operator.node_children(node) and node != operator.root_node:
            operator.nodes.remove(node)
