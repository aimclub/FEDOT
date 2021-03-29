import json
import os
from typing import List
from uuid import uuid4

import networkx as nx

import joblib

from fedot.core.chains.node import CachedState, Node, PrimaryNode, SecondaryNode
from fedot.core.utils import default_fedot_data_dir

DEFAULT_FITTED_OPERATIONS_PATH = os.path.join(default_fedot_data_dir(), 'fitted_operations')


class ChainTemplate:
    def __init__(self, chain):
        self.total_chain_types = {}
        self.depth = None
        self.operation_templates = []
        self.unique_chain_id = str(uuid4())

        if chain.root_node:
            self._chain_to_template(chain)
        else:
            self.link_to_empty_chain = chain

    def _chain_to_template(self, chain):
        self._extract_chain_structure(chain.root_node, 0, [])
        self.depth = chain.depth

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

        operation_template = OperationTemplate(node, operation_id, nodes_from, self.unique_chain_id)

        self.operation_templates.append(operation_template)
        self._add_chain_type_to_state(operation_template.operation_type)

        return operation_template

    def _add_chain_type_to_state(self, operation_type: str):
        if operation_type in self.total_chain_types:
            self.total_chain_types[operation_type] += 1
        else:
            self.total_chain_types[operation_type] = 1

    def export_to_json(self, path: str):
        path = self._create_unique_chain_id(path)
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        chain_template_dict = self.convert_to_dict()
        json_data = json.dumps(chain_template_dict)
        with open(os.path.join(absolute_path, f'{self.unique_chain_id}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(json_data), indent=4))
            resulted_path = os.path.join(absolute_path, f'{self.unique_chain_id}.json')
            print(f"The chain saved in the path: {resulted_path}")

        return json_data

    def _create_unique_chain_id(self, path_to_save: str):
        name_of_files = path_to_save.split('/')
        last_file = name_of_files[-1].split('.')
        if len(last_file) >= 2:
            if last_file[-1] == 'json':
                if os.path.exists(path_to_save):
                    raise FileExistsError(f"File on path: '{os.path.abspath(path_to_save)}' exists")
                self.unique_chain_id = ''.join(last_file[0:-1])
                return '/'.join(name_of_files[0:-1])
            else:
                raise ValueError(f"Could not save chain in"
                                 f" '{last_file[-1]}' extension, use 'json' format")
        else:
            return path_to_save

    def convert_to_dict(self) -> dict:
        chain_types = self.total_chain_types
        json_nodes = list(map(lambda operation_template: operation_template.export_to_json(), self.operation_templates))

        json_object = {
            "total_chain_types": chain_types,
            "depth": self.depth,
            "nodes": json_nodes,
        }

        return json_object

    def import_from_json(self, path: str):
        self._check_for_json_existence(path)

        with open(path) as json_file:
            json_object_chain = json.load(json_file)

        self._extract_operations(json_object_chain)
        self.convert_to_chain(self.link_to_empty_chain)
        self.depth = self.link_to_empty_chain.depth
        self.link_to_empty_chain = None

    def _check_for_json_existence(self, path: str):
        absolute_path = os.path.abspath(path)
        path, name_of_file = os.path.split(path)
        name_of_file = name_of_file.split('.')
        if len(name_of_file) >= 2:
            if name_of_file[-1] == 'json':
                if not os.path.isfile(absolute_path):
                    raise FileNotFoundError(f"File on the path: {absolute_path} does not exist.")
                self.unique_chain_id = ''.join(name_of_file[0:-1])
            else:
                raise ValueError(f"Could not load chain in"
                                 f" '{name_of_file[-1]}' extension, use 'json' format")
        else:
            raise FileNotFoundError(f"Write path to 'json' format file")

    def _extract_operations(self, chain_json):
        operation_objects = chain_json['nodes']

        for operation_object in operation_objects:
            operation_template = OperationTemplate()
            operation_template.import_from_json(operation_object)
            self._add_chain_type_to_state(operation_template.operation_type)
            self.operation_templates.append(operation_template)

    def convert_to_chain(self, chain_to_convert_to: 'Chain'):
        visited_nodes = {}
        root_template = [template for template in self.operation_templates if template.operation_id == 0][0]
        root_node = _roll_chain_structure(root_template, visited_nodes, self)
        chain_to_convert_to.nodes.clear()
        chain_to_convert_to.add_node(root_node)


def _roll_chain_structure(operation_object: 'OperationTemplate', visited_nodes: dict, chain_object: ChainTemplate):
    if operation_object.operation_id in visited_nodes:
        return visited_nodes[operation_object.operation_id]
    if operation_object.nodes_from:
        node = SecondaryNode(operation_object.operation_type)
    else:
        node = PrimaryNode(operation_object.operation_type)

    node.operation.params = operation_object.params
    nodes_from = [operation_template for operation_template in chain_object.operation_templates
                  if operation_template.operation_id in operation_object.nodes_from]
    node.nodes_from = [_roll_chain_structure(node_from, visited_nodes, chain_object) for node_from
                       in nodes_from]

    if operation_object.fitted_operation_path:
        path_to_operation = os.path.abspath(operation_object.fitted_operation_path)
        if not os.path.isfile(path_to_operation):
            raise FileNotFoundError(f"File on the path: {path_to_operation} does not exist.")

        fitted_operation = joblib.load(path_to_operation)
        node.cache.append(CachedState(operation=fitted_operation))
    visited_nodes[operation_object.operation_id] = node
    return node


class OperationTemplate:
    def __init__(self, node: Node = None, operation_id: int = None,
                 nodes_from: list = None, chain_id: str = None):
        self.operation_id = None
        self.operation_type = None
        self.operation_name = None
        self.custom_params = None
        self.params = None
        self.nodes_from = None
        self.fitted_operation_path = None

        if node:
            self._operation_to_template(node, operation_id, nodes_from, chain_id)

    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list, chain_id: str):
        self.operation_id = operation_id
        self.operation_type = node.operation.operation_type
        self.custom_params = node.operation.params
        self.params = {}
        self.nodes_from = nodes_from

        if _is_node_fitted(node):
            self.operation_name = _extract_operation_name(node)
            self.params = _extract_operation_params(node, self.custom_params)
            self.fitted_operation_path = _extract_and_save_fitted_operation(node, chain_id, self.operation_id)

    def export_to_json(self) -> dict:

        operation_object = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_name": self.operation_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "trained_operation_path": self.fitted_operation_path,
        }

        return operation_object

    def import_from_json(self, operation_object: dict):
        _validate_json_operation_template(operation_object)

        self.operation_id = operation_object['operation_id']
        self.operation_type = operation_object['operation_type']
        self.params = operation_object['params']
        self.nodes_from = operation_object['nodes_from']
        if "trained_operation_path" in operation_object:
            self.fitted_operation_path = operation_object['trained_operation_path']
        if "custom_params" in operation_object:
            self.custom_params = operation_object['custom_params']
        if "operation_name" in operation_object:
            self.operation_name = operation_object['operation_name']


def _validate_json_operation_template(operation_object: dict):
    required_fields = ['operation_id', 'operation_type', 'params', 'nodes_from']

    for field in required_fields:
        if field not in operation_object:
            raise RuntimeError(f"Required field '{field}' is expected, but not found")


def _extract_operation_params(node: Node, custom_params: [str, dict]):
    params = node.cache.actual_cached_state.operation.get_params()

    if isinstance(custom_params, dict):
        params.update(custom_params)

    return params


def _extract_operation_name(node: Node):
    return node.cache.actual_cached_state.operation.__class__.__name__


def _extract_and_save_fitted_operation(node: Node, chain_id: str, operation_id: int) -> str:
    absolute_path = os.path.abspath(os.path.join(DEFAULT_FITTED_OPERATIONS_PATH, chain_id))

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    operation_name = f'operation_{str(operation_id)}.pkl'
    fitted_operation_path = os.path.join(DEFAULT_FITTED_OPERATIONS_PATH, chain_id, operation_name)
    joblib.dump(node.cache.actual_cached_state.operation, fitted_operation_path)

    return fitted_operation_path


def _is_node_fitted(node: Node) -> bool:
    return bool(node.cache.actual_cached_state)


def extract_subtree_root(root_operation_id: int, chain_template: ChainTemplate):
    root_node = [operation_template for operation_template in chain_template.operation_templates
                 if operation_template.operation_id == root_operation_id][0]
    root_node = _roll_chain_structure(root_node, {}, chain_template)

    return root_node


def chain_template_as_nx_graph(chain: ChainTemplate):
    graph = nx.DiGraph()
    node_labels = {}
    for operation in chain.operation_templates:
        unique_id, label = operation.operation_id, operation.operation_type
        node_labels[unique_id] = label
        graph.add_node(unique_id)

    def add_edges(graph, chain):
        for operation in chain.operation_templates:
            if operation.nodes_from is not None:
                for child in operation.nodes_from:
                    graph.add_edge(child, operation.operation_id)

    add_edges(graph, chain)
    return graph, node_labels
