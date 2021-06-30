import json
import os
from collections import Counter
from datetime import datetime
from typing import List
from uuid import uuid4

import joblib

from fedot.core.log import Log, default_log
from fedot.core.operations.atomized_template import AtomizedModelTemplate
from fedot.core.operations.operation_template import OperationTemplate
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.repository.operation_types_repository import atomized_model_type


class PipelineTemplate:
    """
    Pipeline wrapper with 'export_pipeline'/'import_pipeline' methods
    allowing user to upload a pipeline to JSON format and import it from JSON.

    :params pipeline: Pipeline object to export or empty Pipeline to import
    :params log: Log object to record messages
    """

    def __init__(self, pipeline=None, log: Log = None):
        self.total_pipeline_operations = Counter()
        self.depth = pipeline.depth
        self.operation_templates = []
        self.unique_pipeline_id = str(uuid4()) if not pipeline.uid else pipeline.uid
        try:
            self.computation_time = pipeline.computation_time
        except AttributeError:
            self.computation_time = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        self._pipeline_to_template(pipeline)

    def _pipeline_to_template(self, pipeline):
        try:
            if isinstance(pipeline.root_node, list):
                # TODO improve for graph with several roots
                self._extract_pipeline_structure(pipeline.root_node[0], 0, [])
            else:
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
                operation_template = AtomizedModelTemplate(node, operation_id, nodes_from)
            else:
                operation_template = OperationTemplate(node, operation_id, nodes_from)

            self.operation_templates.append(operation_template)
            self.total_pipeline_operations[operation_template.operation_type] += 1

    def export_pipeline(self, path: str):
        """
        Save JSON to path and return this JSON like object.
        :param path: custom path to save
        :return: JSON like object
        """

        path = self._prepare_paths(path)
        absolute_path = os.path.abspath(path)

        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        pipeline_template_dict = self.convert_to_dict()
        json_data = json.dumps(pipeline_template_dict, indent=4)

        with open(os.path.join(absolute_path, f'{self.unique_pipeline_id}.json'), 'w', encoding='utf-8') as f:
            f.write(json_data)
            resulted_path = os.path.join(absolute_path, f'{self.unique_pipeline_id}.json')
            self.log.message(f"The pipeline saved in the path: {resulted_path}.")

        self._create_fitted_operations(absolute_path)

        return json_data

    def convert_to_dict(self) -> dict:
        json_nodes = list(map(lambda op_template: op_template.convert_to_dict(), self.operation_templates))

        json_object = {
            "total_pipeline_operations": self.total_pipeline_operations,
            "depth": self.depth,
            "nodes": json_nodes,
        }

        return json_object

    def _create_fitted_operations(self, path):
        for operation in self.operation_templates:
            operation.export_operation(path)

    def _prepare_paths(self, path: str):
        absolute_path = os.path.abspath(path)
        path, folder_name = os.path.split(path)
        folder_name = os.path.splitext(folder_name)[0]

        if not os.path.isdir(os.path.dirname(absolute_path)):
            message = f"The path to save a pipeline is not a directory: {absolute_path}."
            self.log.error(message)
            raise FileNotFoundError(message)

        self.unique_pipeline_id = folder_name

        if _is_nested_path(folder_name):
            folder_name = f"{datetime.now().strftime('%B-%d-%Y,%H-%M-%S,%p')} {folder_name}"

        path_to_save = os.path.join(path, folder_name)

        return path_to_save

    def import_pipeline(self, path: str):
        self._check_path_correct(path)

        with open(path) as json_file:
            json_object_pipeline = json.load(json_file)
            self.log.message(f"The pipeline was imported from the path: {path}.")

        self._extract_operations(json_object_pipeline, path)
        self.convert_to_pipeline(self.link_to_empty_pipeline, path)
        self.depth = self.link_to_empty_pipeline.depth

    def _check_path_correct(self, path: str):
        absolute_path = os.path.abspath(path)
        name_of_file = os.path.basename(absolute_path)

        if os.path.isfile(absolute_path):
            self.unique_pipeline_id = os.path.splitext(name_of_file)[0]
        else:
            message = f"The path to load a pipeline is not correct: {absolute_path}."
            self.log.error(message)
            raise FileNotFoundError(message)

    def _extract_operations(self, pipeline_json, path):
        operation_objects = pipeline_json['nodes']

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

    def convert_to_pipeline(self, pipeline, path: str = None):
        if path is not None:
            path = os.path.abspath(os.path.dirname(path))
        visited_nodes = {}
        root_template = [op_template for op_template in self.operation_templates if op_template.operation_id == 0][0]
        root_node = self.roll_pipeline_structure(root_template, visited_nodes, path)
        pipeline.nodes.clear()
        pipeline.add_node(root_node)

    def roll_pipeline_structure(self, operation_object: ['OperationTemplate',
                                                         'AtomizedModelTemplate'],
                                visited_nodes: dict, path: str = None):
        """
        The function recursively traverses all disjoint operations
        and connects the operations in a pipeline.

        :params operation_object: operationTemplate or AtomizedOperationTemplate
        :params visited_nodes: array to remember which node was visited
        :params path: path to save
        :return: root_node
        """
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
            node.operation.params = operation_object.params
            node.rating = operation_object.rating

        if hasattr(operation_object,
                   'fitted_operation_path') and operation_object.fitted_operation_path and path is not None:
            path_to_operation = os.path.join(path, operation_object.fitted_operation_path)
            if not os.path.isfile(path_to_operation):
                message = f"Fitted operation on the path: {path_to_operation} does not exist."
                self.log.error(message)
                raise FileNotFoundError(message)

            fitted_operation = joblib.load(path_to_operation)
            operation_object.fitted_operation = fitted_operation
            node.fitted_operation = fitted_operation

        nodes_from = [operation_template for operation_template in self.operation_templates
                      if operation_template.operation_id in operation_object.nodes_from]
        node.nodes_from = [self.roll_pipeline_structure(node_from, visited_nodes, path) for node_from
                           in nodes_from]

        visited_nodes[operation_object.operation_id] = node
        return node


def _is_nested_path(path):
    return path.find('nested') == -1


def extract_subtree_root(root_operation_id: int, pipeline_template: PipelineTemplate):
    root_node = [operation_template for operation_template in pipeline_template.operation_templates
                 if operation_template.operation_id == root_operation_id][0]
    root_node = pipeline_template.roll_pipeline_structure(root_node, {})

    return root_node
