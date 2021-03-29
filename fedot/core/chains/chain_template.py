from datetime import datetime
import joblib
import json
import os

from collections import Counter
from typing import List
from uuid import uuid4

from fedot.core.chains.node import CachedState, Node, PrimaryNode, SecondaryNode
from fedot.core.log import default_log, Log


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

    def _extract_chain_structure(self, node: Node, operation_id: int, visited_nodes: List[str]):
        """
        Recursively go through the Chain from 'root_node' to PrimaryNode's,
        creating a OperationTemplate with unique id for each Node. In addition,
        checking whether this Node has been visited yet.
        """
        if node.nodes_from:
            nodes_from = []
            for node_parent in node.nodes_from:
                if node_parent.descriptive_id in visited_nodes:
                    nodes_from.append(visited_nodes.index(node_parent.descriptive_id) + 1)
                else:
                    visited_nodes.append(node_parent.descriptive_id)
                    nodes_from.append(len(visited_nodes))
                    self._extract_chain_structure(node_parent, len(visited_nodes), visited_nodes)
        else:
            nodes_from = []

        operation_template = OperationTemplate(node, operation_id, nodes_from)

        self.operation_templates.append(operation_template)
        self.total_chain_operations[operation_template.operation_type] += 1

        return operation_template

    def export_chain(self, path: str):
        path = self._prepare_paths(path)
        absolute_path = os.path.abspath(path)

        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        chain_template_dict = self.convert_to_dict()
        json_data = json.dumps(chain_template_dict)

        with open(os.path.join(absolute_path, f'{self.unique_chain_id}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(json_data), indent=4))
            resulted_path = os.path.join(absolute_path, f'{self.unique_chain_id}.json')
            self.log.info(f"The chain saved in the path: {resulted_path}.")

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
        path_fitted_operations = os.path.join(path, 'fitted_operations')

        if not os.path.exists(path_fitted_operations):
            os.makedirs(path_fitted_operations)

        for operation in self.operation_templates:
            operation.export_pkl_operation(path)

    def _prepare_paths(self, path: str):
        absolute_path = os.path.abspath(path)
        path, folder_name = os.path.split(path)
        folder_name = os.path.splitext(folder_name)[0]

        if not os.path.isdir(os.path.dirname(absolute_path)):
            message = f"The path to save a chain is not a directory: {absolute_path}."
            self.log.error(message)
            raise FileNotFoundError(message)

        self.unique_chain_id = folder_name
        folder_name = f"{datetime.now().strftime('%B-%d-%Y,%H-%M-%S,%p')} {folder_name}"
        path_to_save = os.path.join(path, folder_name)

        return path_to_save

    def import_chain(self, path: str):
        self._check_path_correct(path)

        with open(path) as json_file:
            json_object_chain = json.load(json_file)
            self.log.info(f"The chain was imported from the path: {path}.")

        self._extract_operations(json_object_chain)
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

    def _extract_operations(self, chain_json):
        operation_objects = chain_json['nodes']

        for operation_object in operation_objects:
            operation_template = OperationTemplate()
            operation_template.import_from_json(operation_object)
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

    def roll_chain_structure(self, operation_object: 'OperationTemplate', visited_nodes: dict, path: str = None):
        """
        The function recursively traverses all disjoint operations
        and connects the operations in a chain.
        :return: root_node
        """
        if operation_object.operation_id in visited_nodes:
            return visited_nodes[operation_object.operation_id]
        if operation_object.nodes_from:
            node = SecondaryNode(operation_object.operation_type)
        else:
            node = PrimaryNode(operation_object.operation_type)

        node.operation.params = operation_object.params
        nodes_from = [operation_template for operation_template in self.operation_templates
                      if operation_template.operation_id in operation_object.nodes_from]
        node.nodes_from = [self.roll_chain_structure(node_from, visited_nodes, path) for node_from
                           in nodes_from]

        if operation_object.fitted_operation_path and path is not None:
            path_to_operation = os.path.join(path, operation_object.fitted_operation_path)
            if not os.path.isfile(path_to_operation):
                message = f"Fitted operation on the path: {path_to_operation} does not exist."
                self.log.error(message)
                raise FileNotFoundError(message)

            fitted_operation = joblib.load(path_to_operation)
            node.cache.append(CachedState(operation=fitted_operation))
        visited_nodes[operation_object.operation_id] = node
        return node


class OperationTemplate:
    def __init__(self, node: Node = None, operation_id: int = None,
                 nodes_from: list = None, log: Log = None):
        self.operation_id = None
        self.operation_type = None
        self.operation_name = None
        self.custom_params = None
        self.params = None
        self.nodes_from = None
        self.fitted_operation = None
        self.fitted_operation_path = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if node:
            self._operation_to_template(node, operation_id, nodes_from)

    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list):
        self.operation_id = operation_id
        self.operation_type = node.operation.operation_type
        self.custom_params = node.operation.params
        self.params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if _is_node_fitted(node):
            self.operation_name = _extract_operation_name(node)
            self._extract_fields_of_fitted_operation(node)

    def _extract_fields_of_fitted_operation(self, node: Node):
        operation_name = f'operation_{str(self.operation_id)}.pkl'
        self.fitted_operation_path = os.path.join('fitted_operations', operation_name)
        self.fitted_operation = node.cache.actual_cached_state.operation

    def export_pkl_operation(self, path: str):
        if self.fitted_operation:
            joblib.dump(self.fitted_operation, os.path.join(path, self.fitted_operation_path))

    def _create_full_params(self, node: Node) -> dict:
        wrapped_operations = ['base_estimator', 'estimator']

        params = {}
        if _is_node_fitted(node):
            params = _extract_operation_params(node)

            # Check if it is needed to process "model in model" cases
            # such strategy is needed for RANSAC or RFE algorithms
            for wrapped_operation in wrapped_operations:
                if wrapped_operation in params:
                    del params[wrapped_operation]

            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    params[key] = value

        return params

    def convert_to_dict(self) -> dict:

        operation_object = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_name": self.operation_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "fitted_operation_path": self.fitted_operation_path,
        }

        return operation_object

    def import_from_json(self, operation_object: dict):
        self._validate_json_operation_template(operation_object)

        self.operation_id = operation_object['operation_id']
        self.operation_type = operation_object['operation_type']
        self.params = operation_object['params']
        self.nodes_from = operation_object['nodes_from']
        if "fitted_operation_path" in operation_object:
            self.fitted_operation_path = operation_object['fitted_operation_path']
        if "custom_params" in operation_object:
            self.custom_params = operation_object['custom_params']
        if "operation_name" in operation_object:
            self.operation_name = operation_object['operation_name']

    def _validate_json_operation_template(self, operation_object: dict):
        required_fields = ['operation_id', 'operation_type', 'params', 'nodes_from']

        for field in required_fields:
            if field not in operation_object:
                message = f"Required field '{field}' is expected, but not found."
                self.log.error(message)
                raise RuntimeError(message)


def _extract_operation_params(node: Node):
    return node.cache.actual_cached_state.operation.get_params()


def _extract_operation_name(node: Node):
    return node.cache.actual_cached_state.operation.__class__.__name__


def _is_node_fitted(node: Node) -> bool:
    return bool(node.cache.actual_cached_state)


def extract_subtree_root(root_operation_id: int, chain_template: ChainTemplate):
    root_node = [operation_template for operation_template in chain_template.operation_templates
                 if operation_template.operation_id == root_operation_id][0]
    root_node = chain_template.roll_chain_structure(root_node, {})

    return root_node
