import json
import os
from collections import Counter
from datetime import datetime
from typing import List
from uuid import uuid4

import joblib

from fedot.core.chains.node import Node, PrimaryNode, SecondaryNode
from fedot.core.log import Log, default_log
from fedot.core.models.atomized_template import AtomizedModelTemplate
from fedot.core.models.model_template import ModelTemplate
from fedot.core.repository.model_types_repository import atomized_model_type


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

    def _extract_chain_structure(self, node: Node, model_id: int, visited_nodes: List[Node]):
        """
        Recursively go through the Chain from 'root_node' to PrimaryNode's,
        creating a ModelTemplate with unique id for each Node. In addition,
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

        if node.model.model_type == atomized_model_type():
            model_template = AtomizedModelTemplate(node, model_id, nodes_from)
        else:
            model_template = ModelTemplate(node, model_id, nodes_from)

        self.model_templates.append(model_template)
        self.total_chain_models[model_template.model_type] += 1

        return model_template

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
        for model in self.model_templates:
            model.export_model(path)

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

        self._extract_models(json_object_chain, path)
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

    def _extract_models(self, chain_json, path):
        model_objects = chain_json['nodes']

        for model_object in model_objects:
            if model_object['model_type'] == atomized_model_type():
                filename = model_object['atomized_model_json_path'] + '.json'
                curr_path = os.path.join(os.path.dirname(path), model_object['atomized_model_json_path'], filename)
                model_template = AtomizedModelTemplate(path=curr_path)
            else:
                model_template = ModelTemplate()

            model_template.import_json(model_object)
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

    def roll_chain_structure(self, model_object: ['ModelTemplate', 'AtomizedModelTemplate'],
                             visited_nodes: dict, path: str = None):
        """
        The function recursively traverses all disjoint models
        and connects the models in a chain.

        :params model_object: ModelTemplate or AtomizedModelTemplate
        :params visited_nodes: array to remember which node was visited
        :params path: path to save
        :return: root_node
        """
        if model_object.model_id in visited_nodes:
            return visited_nodes[model_object.model_id]

        if model_object.model_type == atomized_model_type():
            atomized_model = model_object.next_chain_template
            if model_object.nodes_from:
                node = SecondaryNode(model_type=atomized_model)
            else:
                node = PrimaryNode(model_type=atomized_model)
        else:
            if model_object.nodes_from:
                node = SecondaryNode(model_object.model_type)
            else:
                node = PrimaryNode(model_object.model_type)
            node.model.params = model_object.params

        if hasattr(model_object, 'fitted_model_path') and model_object.fitted_model_path and path is not None:
            path_to_model = os.path.join(path, model_object.fitted_model_path)
            if not os.path.isfile(path_to_model):
                message = f"Fitted model on the path: {path_to_model} does not exist."
                self.log.error(message)
                raise FileNotFoundError(message)

            fitted_model = joblib.load(path_to_model)
            model_object.fitted_model = fitted_model
            node.fitted_model = fitted_model
            node.fitted_preprocessor = model_object.preprocessor

        nodes_from = [model_template for model_template in self.model_templates
                      if model_template.model_id in model_object.nodes_from]
        node.nodes_from = [self.roll_chain_structure(node_from, visited_nodes, path) for node_from
                           in nodes_from]

        visited_nodes[model_object.model_id] = node
        return node


def _is_nested_path(path):
    return path.find('nested') == -1


def extract_subtree_root(root_model_id: int, chain_template: ChainTemplate):
    root_node = [model_template for model_template in chain_template.model_templates
                 if model_template.model_id == root_model_id][0]
    root_node = chain_template.roll_chain_structure(root_node, {})

    return root_node
