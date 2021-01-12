from datetime import datetime
import joblib
import json
import os

from collections import Counter
from typing import List
from uuid import uuid4

from fedot.core.chains.node import CachedState, Node, PrimaryNode, SecondaryNode
from fedot.core.data.preprocessing import preprocessing_strategy_class_by_label, preprocessing_strategy_label_by_class
from fedot.core.log import default_log, Log


class ChainTemplate:
    """
    Chain wrapper with 'export_chain'/'import_chain' methods
    allowing user to upload a chain to JSON format and import it from JSON.

    :params chain: Chain object to export or empty Chain to import
    :params log: Log object to record messages
    """
    def __init__(self, chain=None, log: Log = None):
        self.total_chain_models = Counter()
        self.depth = chain.depth
        self.model_templates = []
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

    def _extract_chain_structure(self, node: Node, model_id: int, visited_nodes: List[str]):
        """
        Recursively go through the Chain from 'root_node' to PrimaryNode's,
        creating a ModelTemplate with unique id for each Node. In addition,
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

        model_template = ModelTemplate(node, model_id, nodes_from)

        self.model_templates.append(model_template)
        self.total_chain_models[model_template.model_type] += 1

        return model_template

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

        self._create_fitted_models(absolute_path)

        return json_data

    def convert_to_dict(self) -> dict:
        json_nodes = list(map(lambda model_template: model_template.convert_to_dict(), self.model_templates))

        json_object = {
            "total_chain_models": self.total_chain_models,
            "depth": self.depth,
            "nodes": json_nodes,
        }

        return json_object

    def _create_fitted_models(self, path):
        path_fitted_models = os.path.join(path, 'fitted_models')

        if not os.path.exists(path_fitted_models):
            os.makedirs(path_fitted_models)

        for model in self.model_templates:
            model.export_pkl_model(path)

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

        self._extract_models(json_object_chain)
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

    def _extract_models(self, chain_json):
        model_objects = chain_json['nodes']

        for model_object in model_objects:
            model_template = ModelTemplate()
            model_template.import_from_json(model_object)
            self.model_templates.append(model_template)
            self.total_chain_models[model_template.model_type] += 1

    def convert_to_chain(self, chain, path: str = None):
        if path is not None:
            path = os.path.abspath(os.path.dirname(path))
        visited_nodes = {}
        root_template = [model_template for model_template in self.model_templates if model_template.model_id == 0][0]
        root_node = self.roll_chain_structure(root_template, visited_nodes, path)
        chain.nodes.clear()
        chain.add_node(root_node)

    def roll_chain_structure(self, model_object: 'ModelTemplate', visited_nodes: dict, path: str = None):
        """
        The function recursively traverses all disjoint models
        and connects the models in a chain.
        :return: root_node
        """
        if model_object.model_id in visited_nodes:
            return visited_nodes[model_object.model_id]
        if model_object.nodes_from:
            node = SecondaryNode(model_object.model_type)
        else:
            node = PrimaryNode(model_object.model_type)

        node.model.params = model_object.params
        nodes_from = [model_template for model_template in self.model_templates
                      if model_template.model_id in model_object.nodes_from]
        node.nodes_from = [self.roll_chain_structure(node_from, visited_nodes, path) for node_from
                           in nodes_from]

        if model_object.fitted_model_path and path is not None:
            path_to_model = os.path.join(path, model_object.fitted_model_path)
            if not os.path.isfile(path_to_model):
                message = f"Fitted model on the path: {path_to_model} does not exist."
                self.log.error(message)
                raise FileNotFoundError(message)

            fitted_model = joblib.load(path_to_model)
            node.cache.append(CachedState(preprocessor=model_object.preprocessor,
                                          model=fitted_model))
        visited_nodes[model_object.model_id] = node
        return node


class ModelTemplate:
    def __init__(self, node: Node = None, model_id: int = None,
                 nodes_from: list = None, log: Log = None):
        self.model_id = None
        self.model_type = None
        self.model_name = None
        self.custom_params = None
        self.params = None
        self.nodes_from = None
        self.fitted_model = None
        self.fitted_model_path = None
        self.preprocessor = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if node:
            self._model_to_template(node, model_id, nodes_from)

    def _model_to_template(self, node: Node, model_id: int, nodes_from: list):
        self.model_id = model_id
        self.model_type = node.model.model_type
        self.custom_params = node.model.params
        self.params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if _is_node_fitted(node) and not _is_node_not_cached(node):
            self.model_name = _extract_model_name(node)
            self._extract_fields_of_fitted_model(node)

    def _extract_fields_of_fitted_model(self, node: Node):
        model_name = f'model_{str(self.model_id)}.pkl'
        self.fitted_model_path = os.path.join('fitted_models', model_name)
        self.preprocessor = _extract_preprocessing_strategy(node)
        self.fitted_model = node.cache.actual_cached_state.model

    def export_pkl_model(self, path: str):
        if self.fitted_model:
            joblib.dump(self.fitted_model, os.path.join(path, self.fitted_model_path))

    def _create_full_params(self, node: Node) -> dict:
        params = {}
        if _is_node_fitted(node) and not _is_node_not_cached(node):
            params = _extract_model_params(node)
            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    params[key] = value

        return params

    def convert_to_dict(self) -> dict:
        preprocessor_strategy = preprocessing_strategy_label_by_class(self.preprocessor)

        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "fitted_model_path": self.fitted_model_path,
            "preprocessor": preprocessor_strategy
        }

        return model_object

    def import_from_json(self, model_object: dict):
        self._validate_json_model_template(model_object)

        self.model_id = model_object['model_id']
        self.model_type = model_object['model_type']
        self.params = model_object['params']
        self.nodes_from = model_object['nodes_from']
        if "fitted_model_path" in model_object:
            self.fitted_model_path = model_object['fitted_model_path']
        if "custom_params" in model_object:
            self.custom_params = model_object['custom_params']
        if "model_name" in model_object:
            self.model_name = model_object['model_name']
        if "preprocessor" in model_object:
            preprocessor_strategy = preprocessing_strategy_class_by_label(model_object['preprocessor'])
            if preprocessor_strategy:
                self.preprocessor = preprocessor_strategy()

    def _validate_json_model_template(self, model_object: dict):
        required_fields = ['model_id', 'model_type', 'params', 'nodes_from', 'preprocessor']

        for field in required_fields:
            if field not in model_object:
                message = f"Required field '{field}' is expected, but not found."
                self.log.error(message)
                raise RuntimeError(message)


def _extract_model_params(node: Node):
    return node.cache.actual_cached_state.model.get_params()


def _extract_model_name(node: Node):
    return node.cache.actual_cached_state.model.__class__.__name__


def _is_node_fitted(node: Node) -> bool:
    return bool(node.cache.actual_cached_state)


def _is_node_not_cached(node: Node) -> bool:
    return bool(node.model.model_type in ['direct_data_model', 'trend_data_model', 'residual_data_model'])


def _extract_preprocessing_strategy(node: Node) -> str:
    return node.cache.actual_cached_state.preprocessor


def extract_subtree_root(root_model_id: int, chain_template: ChainTemplate):
    root_node = [model_template for model_template in chain_template.model_templates
                 if model_template.model_id == root_model_id][0]
    root_node = chain_template.roll_chain_structure(root_node, {})

    return root_node
