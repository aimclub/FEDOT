import json
import os
from collections import Counter
from datetime import datetime
from typing import List
from uuid import uuid4

import joblib

from fedot.core.chains.node import Node, PrimaryNode, SecondaryNode
from fedot.core.log import Log, default_log
from fedot.core.operations.atomized_template import AtomizedModelTemplate
from fedot.core.operations.operation_template import OperationTemplate
from fedot.core.repository.operation_types_repository import atomized_model_type


class ChainTemplate:
    """
    Chain wrapper with 'export_chain'/'import_chain' methods
    allowing user to upload a chain to JSON format and import it from JSON.

    :params chain: Chain object to export or empty Chain to import
    :params log: Log object to record messages
    """

    def __init__(self, chain=None, log: Log = None):
        self.total_chain_operations = Counter()
        self.depth = chain.depth
        self.operation_templates = []
        self.unique_chain_id = str(uuid4())
        self.computation_time = chain.computation_time

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        self._chain_to_template(chain)

    def _chain_to_template(self, chain):
        if chain.root_node:
            self._extract_chain_structure(chain.root_node, 0, [])
        else:
            self.link_to_empty_chain = chain

    def _extract_chain_structure(self, node: Node, operation_id: int, visited_nodes: List[Node]):
        """
        Recursively go through the Chain from 'root_node' to PrimaryNode's,
        creating a OperationTemplate with unique id for each Node. In addition,
        checking whether this Node has been visited yet.
        """

        if node.nodes_from:
            nodes_from = []
            for node_parent in node.nodes_from:
                if node_parent in visited_nodes:
                    nodes_from.append(visited_nodes.index(node_parent) + 1)
                else:
                    visited_nodes.append(node_parent)
                    nodes_from.append(len(visited_nodes))
                    self._extract_chain_structure(node_parent, len(visited_nodes), visited_nodes)
        else:
            nodes_from = []

        if node.operation.operation_type == atomized_model_type():
            operation_template = AtomizedModelTemplate(node, operation_id, nodes_from)
        else:
            operation_template = OperationTemplate(node, operation_id, nodes_from)

        self.operation_templates.append(operation_template)
        self.total_chain_operations[operation_template.operation_type] += 1

        return operation_template

    def export_chain(self, path: str):
        """
        Save JSON to path and return this JSON like object.
        :param path: custom path to save
        :return: JSON like object
        """

        path = self._prepare_paths(path)
        absolute_path = os.path.abspath(path)

        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        chain_template_dict = self.convert_to_dict()
        json_data = json.dumps(chain_template_dict)

        with open(os.path.join(absolute_path, f'{self.unique_chain_id}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(json_data), indent=4))
            resulted_path = os.path.join(absolute_path, f'{self.unique_chain_id}.json')
            self.log.message(f"The chain saved in the path: {resulted_path}.")

        self._create_fitted_operations(absolute_path)

        return json_data

    def convert_to_dict(self) -> dict:
        json_nodes = list(map(lambda op_template: op_template.convert_to_dict(), self.operation_templates))

        json_object = {
            "total_chain_operations": self.total_chain_operations,
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
            message = f"The path to save a chain is not a directory: {absolute_path}."
            self.log.error(message)
            raise FileNotFoundError(message)

        self.unique_chain_id = folder_name

        if _is_nested_path(folder_name):
            folder_name = f"{datetime.now().strftime('%B-%d-%Y,%H-%M-%S,%p')} {folder_name}"

        path_to_save = os.path.join(path, folder_name)

        return path_to_save

    def import_chain(self, path: str):
        self._check_path_correct(path)

        with open(path) as json_file:
            json_object_chain = json.load(json_file)
            self.log.message(f"The chain was imported from the path: {path}.")

        self._extract_operations(json_object_chain, path)
        self.convert_to_chain(self.link_to_empty_chain, path)
        self.depth = self.link_to_empty_chain.depth

    def _check_path_correct(self, path: str):
        absolute_path = os.path.abspath(path)
        name_of_file = os.path.basename(absolute_path)

        if os.path.isfile(absolute_path):
            self.unique_chain_id = os.path.splitext(name_of_file)[0]
        else:
            message = f"The path to load a chain is not correct: {absolute_path}."
            self.log.error(message)
            raise FileNotFoundError(message)

    def _extract_operations(self, chain_json, path):
        operation_objects = chain_json['nodes']

        for operation_object in operation_objects:
            if operation_object['operation_type'] == atomized_model_type():
                filename = operation_object['atomized_model_json_path'] + '.json'
                curr_path = os.path.join(os.path.dirname(path), operation_object['atomized_model_json_path'], filename)
                operation_template = AtomizedModelTemplate(path=curr_path)
            else:
                operation_template = OperationTemplate()

            operation_template.import_json(operation_object)
            self.operation_templates.append(operation_template)
            self.total_chain_operations[operation_template.operation_type] += 1

    def convert_to_chain(self, chain, path: str = None):
        if path is not None:
            path = os.path.abspath(os.path.dirname(path))
        visited_nodes = {}
        root_template = [op_template for op_template in self.operation_templates if op_template.operation_id == 0][0]
        root_node = self.roll_chain_structure(root_template, visited_nodes, path)
        chain.nodes.clear()
        chain.add_node(root_node)

    def roll_chain_structure(self, operation_object: ['OperationTemplate',
                                                      'AtomizedModelTemplate'],
                             visited_nodes: dict, path: str = None):
        """
        The function recursively traverses all disjoint operations
        and connects the operations in a chain.

        :params operation_object: operationTemplate or AtomizedOperationTemplate
        :params visited_nodes: array to remember which node was visited
        :params path: path to save
        :return: root_node
        """
        if operation_object.operation_id in visited_nodes:
            return visited_nodes[operation_object.operation_id]

        if operation_object.operation_type == atomized_model_type():
            atomized_model = operation_object.next_chain_template
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
        node.nodes_from = [self.roll_chain_structure(node_from, visited_nodes, path) for node_from
                           in nodes_from]

        visited_nodes[operation_object.operation_id] = node
        return node


def _is_nested_path(path):
    return path.find('nested') == -1


def extract_subtree_root(root_operation_id: int, chain_template: ChainTemplate):
    root_node = [operation_template for operation_template in chain_template.operation_templates
                 if operation_template.operation_id == root_operation_id][0]
    root_node = chain_template.roll_chain_structure(root_node, {})

    return root_node
