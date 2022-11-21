import json
import os
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import joblib
import numpy as np

from fedot.core.log import default_log
from fedot.core.operations.atomized_template import AtomizedModelTemplate
from fedot.core.operations.operation_template import OperationTemplate, check_existing_path
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.repository.operation_types_repository import atomized_model_type


class NumpyIntEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class PipelineTemplate:
    """
    Pipeline wrapper with 'export_pipeline'/'import_pipeline' methods
    allowing user to upload a pipeline to JSON format and import it from JSON.

    :params pipeline: Pipeline object to export or empty Pipeline to import
    """

    def __init__(self, pipeline: 'Pipeline' = None):
        self.total_pipeline_operations = Counter()
        self.operation_templates: List[OperationTemplate] = []
        self.unique_pipeline_id = str(uuid4())
        self.metadata: Dict[str, Any] = {}
        if pipeline is not None:
            self.depth = pipeline.depth
            self.metadata['computation_time_in_seconds'] = pipeline.computation_time

            # Save preprocessing operations
            self.data_preprocessor = pipeline.preprocessor
        else:
            self.depth = 0
            self.data_preprocessor = None

        self.log = default_log(self)

        self._pipeline_to_template(pipeline)

    def _pipeline_to_template(self, pipeline):
        try:
            # TODO improve for graph with several roots
            self._extract_pipeline_structure(pipeline.root_node, 0, [])
        except Exception as ex:
            self.log.info(f'Cannot export to template: {ex}')
        self.link_to_empty_pipeline = pipeline

    def _extract_pipeline_structure(self, node: Node, operation_id: int, visited_nodes: List[Node]):
        """
        Recursively go through the Pipeline from 'root_node' to PrimaryNode's,
        creating a OperationTemplate with unique id for each Node. In addition,
        checking whether this Node has been visited yet.
        """
        if node and node.nodes_from:
            nodes_from = []
            for node_parent in node.nodes_from:
                if node_parent in visited_nodes:
                    nodes_from.append(visited_nodes.index(node_parent) + 1)
                else:
                    visited_nodes.append(node_parent)
                    nodes_from.append(len(visited_nodes))
                    self._extract_pipeline_structure(node_parent, len(visited_nodes), visited_nodes)
        else:
            nodes_from = []

        # TODO resolve as to GraphTemplate
        if hasattr(node, 'operation'):
            if (not isinstance(node.operation, str) and
                    node.operation.operation_type == atomized_model_type()):
                operation_template = AtomizedModelTemplate(node, operation_id, sorted(nodes_from))
            else:
                operation_template = OperationTemplate(node, operation_id, sorted(nodes_from))

            self.operation_templates.append(operation_template)
            self.total_pipeline_operations[operation_template.operation_type] += 1

    def export_pipeline(self, path: Optional[str] = None, root_node: Optional[Node] = None,
                        create_subdir: bool = True,
                        is_datetime_in_path: bool = False) -> Tuple[str, dict]:
        """
        Save JSON to path and return this JSON like object

        Args:
            path: custom path to save JSON to
            root_node: root node of the exported pipeline
            create_subdir: if True -- create one more dir in the last dir.
                           Useful for saving pipelines with fitted_operations & preprocessing.
                           if False -- save to the last dir in this path.
                           Useful for pipelines without fitted_operations or
                           when you want to know the exact name of the folder.
            is_datetime_in_path: is it required to add the datetime timestamp to the path.

        :return: <JSON representation of the pipeline structure>, <dict of paths to fitted models>
        """

        pipeline_template_dict = self.convert_to_dict(root_node)
        fitted_ops = {}
        if path is None:
            fitted_ops = self._create_fitted_operations()
            if fitted_ops is not None:
                for operation in pipeline_template_dict['nodes']:
                    saved_key = f'operation_{operation["operation_id"]}'
                    if saved_key not in fitted_ops:
                        saved_key = None
                    pipeline_template_dict['fitted_operation_path'] = saved_key

        json_data = json.dumps(pipeline_template_dict, indent=4, cls=NumpyIntEncoder)

        if path is None:
            return json_data, fitted_ops

        path_to_dir, path_to_pipe = self._prepare_paths(path, create_subdir=create_subdir,
                                                        is_datetime_in_path=is_datetime_in_path)
        with open(path_to_pipe, 'w', encoding='utf-8') as f:
            f.write(json_data)
            self.log.debug(f'The pipeline saved in the path: {path_to_pipe}.')

        dict_fitted_operations = self._create_fitted_operations(path_to_dir)

        return json_data, dict_fitted_operations

    def convert_to_dict(self, root_node: Node = None) -> dict:
        """ Generate pipeline description in a form of dictionary """

        json_nodes = list(map(lambda op_template: op_template.convert_to_dict(), self.operation_templates))
        for node in json_nodes:
            if 'custom_params' in node and isinstance(node['custom_params'], dict):
                for key in node['custom_params']:
                    if isinstance(node['custom_params'][key], Callable):
                        node['custom_params'][key] = None

        # Store information about preprocessing
        preprocessing_path = ['preprocessing', 'data_preprocessor.pkl']

        json_object = {
            "total_pipeline_operations": list(self.total_pipeline_operations),
            "depth": self.depth,
            "nodes": json_nodes,
            "preprocessing": preprocessing_path
        }
        if root_node:
            json_object['descriptive_id'] = root_node.descriptive_id

        return json_object

    def _create_fitted_operations(self, path=None):
        """ Create .pkl files for operations using absolute path """
        dict_fitted_operations = {}
        for operation in self.operation_templates:
            dict_fitted_operations[f'operation_{operation.operation_id}'] = operation.export_operation(path)

        if all(val is None for val in dict_fitted_operations.values()):
            return None

        # Save preprocessing module
        preprocessing_path = self.export_preprocessing(path)
        if isinstance(preprocessing_path, str):
            preprocessing_splitted = os.path.split(preprocessing_path)
            preprocessing_path = [preprocessing_splitted[-2], preprocessing_splitted[-1]]
        dict_fitted_operations['preprocessing'] = preprocessing_path
        return dict_fitted_operations

    @staticmethod
    def _prepare_paths(path: str, create_subdir: bool, is_datetime_in_path: bool = False) -> Tuple[str, str]:
        """ Constructs paths to pipeline folder and pipeline json
        Args:
            path: path to json object or to the folder
            create_subdir: bool indicating whether to save pipeline in specified dir or create additional subdir
            is_datetime_in_path: bool indicating whether to save with timestamp or not

        :return: paths to pipeline dir and to pipeline json
        """

        path = os.path.abspath(path)

        # explicitly specify where to save json, fitted_operation and etc will be saved in the same dir
        if path.endswith('.json'):
            path_to_dir = os.path.split(path)[0]
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)
            path_to_json = path
            return path_to_dir, path_to_json

        if not os.path.exists(path):
            os.makedirs(path)

        suffix = '_pipeline_saved'

        if not create_subdir:
            path_to_dir = path
            folder_name = os.path.split(path_to_dir)[-1]
            if is_datetime_in_path:
                json_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{folder_name}"
            else:
                json_name = folder_name
            path_to_json = os.path.join(path_to_dir, f'{json_name}.json')
            return path_to_dir, path_to_json

        if is_datetime_in_path:
            folder_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{suffix}"
        else:
            # get index for pipeline
            files_by_path = os.listdir(path)
            max_pipeline_idx = -1
            for file in files_by_path:
                if suffix in file:
                    cur_pipeline_idx = file.split(suffix)[0] \
                        if '_' not in file.split(suffix)[0] else 0
                    if cur_pipeline_idx.isdigit():
                        cur_pipeline_idx = int(cur_pipeline_idx)
                    else:
                        continue
                    max_pipeline_idx = cur_pipeline_idx if cur_pipeline_idx > max_pipeline_idx else max_pipeline_idx
            folder_name = f"{max_pipeline_idx+1}{suffix}"

        path_to_dir = os.path.join(path, folder_name)
        path_to_dir = path_to_dir
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        folder_name = os.path.split(path_to_dir)[-1]
        path_to_json = os.path.join(path_to_dir, f'{folder_name}.json')

        return path_to_dir, path_to_json

    @staticmethod
    def _is_nested_path(path):
        return path.find('nested') == -1

    def import_pipeline(self, source: Union[str, dict], dict_fitted_operations: Optional[dict] = None):
        """
        Imports pipeline from source into the :attr:`link_to_empty_pipeline`

        :param source: where to load the pipeline from: can be specified as dict or as str
        :param dict_fitted_operations: dictionary of the fitted operations
        """
        path_to_json_pipe = None

        if source is None:
            raise ValueError('Cannot import pipeline: the source is None')
        elif isinstance(source, dict):
            json_object_pipeline = source
            self.log.debug('The pipeline was imported from dict.')
        else:
            path = os.path.abspath(source)
            if os.path.isdir(path):
                path_to_json_pipe = None
                for file in os.listdir(path):
                    if file.endswith('.json'):
                        path_to_json_pipe = os.path.join(path, file)
                if not path_to_json_pipe:
                    raise FileNotFoundError('Path to pipeline is incorrect')
            else:
                path_to_json_pipe = path

            with open(path_to_json_pipe) as json_file:
                json_object_pipeline = json.load(json_file)
                self.log.debug(f'The pipeline was imported from the path: {path_to_json_pipe}.')

        self._extract_operations(json_object_pipeline, path_to_json_pipe)
        self.convert_to_pipeline(self.link_to_empty_pipeline, path_to_json_pipe, dict_fitted_operations)
        self.depth = self.link_to_empty_pipeline.depth

    def _extract_operations(self, pipeline_json: dict, path: str):
        operation_objects = pipeline_json['nodes']

        # Update info about fitted operation
        for operation_object in operation_objects:
            if operation_object['operation_type'] == atomized_model_type():
                filename = operation_object['atomized_model_json_path'] + '.json'
                curr_path = os.path.join(os.path.dirname(path), operation_object['atomized_model_json_path'], filename)
                operation_template = AtomizedModelTemplate(path=curr_path)
            else:
                operation_template = OperationTemplate()

            operation_template.import_json(operation_object)
            self.operation_templates.append(operation_template)
            self.total_pipeline_operations[operation_template.operation_type] += 1

    def convert_to_pipeline(self, pipeline, path: str = None, dict_fitted_operations: dict = None):
        if path is not None:
            path = os.path.abspath(os.path.dirname(path))
        visited_nodes = {}
        root_template = [op_template for op_template in self.operation_templates if op_template.operation_id == 0][0]

        root_node = self.roll_pipeline_structure(root_template, visited_nodes, path, dict_fitted_operations)
        pipeline.nodes.clear()
        pipeline.add_node(root_node)

        preprocessor_file = None
        if path is not None and 'preprocessing' in os.listdir(path):
            # Load data preprocessor and store it into the
            preprocessor_file = os.path.join(path, 'preprocessing', 'data_preprocessor.pkl')
        elif dict_fitted_operations and 'preprocessing' in dict_fitted_operations:
            preprocessor_file = dict_fitted_operations['preprocessing']
        if preprocessor_file:
            try:
                pipeline.preprocessor = joblib.load(preprocessor_file)
            except ModuleNotFoundError as ex:
                self.log.warning(f'Could not load preprocessor from file `{preprocessor_file}` '
                                 f'due to legacy incompatibility. Please refit the preprocessor.')

    def roll_pipeline_structure(self, operation_object: Union['OperationTemplate', 'AtomizedModelTemplate'],
                                visited_nodes: dict, path: str = None, dict_fitted_operations: dict = None):
        """
        The function recursively traverses all disjoint operations
        and connects the operations in a pipeline.

        :params operation_object: operationTemplate or AtomizedOperationTemplate
        :params visited_nodes: array to remember which node was visited
        :params path: path to save
        :return: root_node
        """
        fitted_operation = None
        if operation_object.operation_id in visited_nodes:
            return visited_nodes[operation_object.operation_id]

        if operation_object.operation_type == atomized_model_type():
            atomized_model = operation_object.next_pipeline_template
            if operation_object.nodes_from:
                node = SecondaryNode(operation_type=atomized_model)
            else:
                node = PrimaryNode(operation_type=atomized_model)
        else:
            if operation_object.nodes_from:
                node = SecondaryNode(operation_object.operation_type)
            else:
                node = PrimaryNode(operation_object.operation_type)

            node.parameters = operation_object.custom_params

            node.rating = operation_object.rating

        if hasattr(operation_object,
                   'fitted_operation_path') and operation_object.fitted_operation_path and path is not None:
            path_to_operation = os.path.join(path, operation_object.fitted_operation_path)

            if 'h2o' in operation_object.operation_type:
                fitted_operation = load_h2o(path, self.log)

            elif not os.path.isfile(path_to_operation):
                message = f"Fitted operation on the path: {path_to_operation} does not exist."
                self.log.error(message)
            else:
                fitted_operation = joblib.load(path_to_operation)
        elif dict_fitted_operations is not None:
            if 'h2o' in operation_object.operation_type:
                message = 'Loading h2o models from dict is not supported'
                self.log.error(message)
                raise TypeError(message)
            else:
                fitted_operation = joblib.load(dict_fitted_operations[f'operation_{operation_object.operation_id}'])

        operation_object.fitted_operation = fitted_operation
        node.fitted_operation = fitted_operation

        nodes_from = [operation_template for operation_template in self.operation_templates
                      if operation_template.operation_id in operation_object.nodes_from]

        node.nodes_from = [self.roll_pipeline_structure(node_from, visited_nodes, path, dict_fitted_operations) for
                           node_from in nodes_from]

        visited_nodes[operation_object.operation_id] = node
        return node

    def export_preprocessing(self, path: str = None):
        """ Save preprocessing operations in pkl files and store full paths into dictionary """
        if path:
            path_fitted_preprocessing = os.path.join(path, 'preprocessing')
            check_existing_path(path_fitted_preprocessing)

            data_preprocessor_path = os.path.join(path_fitted_preprocessing, 'data_preprocessor.pkl')
            if self.data_preprocessor is not None:
                joblib.dump(self.data_preprocessor, data_preprocessor_path)
                return data_preprocessor_path
        else:
            # dictionary with bytes of fitted operations
            if self.data_preprocessor:
                bytes_container = BytesIO()
                joblib.dump(self.data_preprocessor, bytes_container)
                return bytes_container


def extract_subtree_root(root_operation_id: int, pipeline_template: PipelineTemplate):
    root_node = [operation_template for operation_template in pipeline_template.operation_templates
                 if operation_template.operation_id == root_operation_id][0]
    root_node = pipeline_template.roll_pipeline_structure(root_node, {})

    return root_node


def load_h2o(path_to_operation, log):
    from fedot.core.operations.evaluation.automl import H2OSerializationWrapper
    try:
        return H2OSerializationWrapper.load_operation(path_to_operation)
    except EnvironmentError as e:
        message = f'Obtained type of H2O pipeline does not serializable: {e}'
        log.error(message)
        raise EnvironmentError(message)
