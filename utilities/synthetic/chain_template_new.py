import json
import os
from uuid import uuid4
from utilities.synthetic.get_application_settings_path import DEFAULT_PATH
from utilities.synthetic.custom_errors import JsonFileExtensionValidation

import joblib

from core.composer.node import PrimaryNode, SecondaryNode, Node

DEFAULT_FITTED_MODELS_PATH = os.path.join(DEFAULT_PATH, 'fitted_models')


class ChainTemplate:
    def __init__(self, chain):
        self.total_model_types = {}
        self.depth = None
        self.model_templates = []
        self.unique_chain_id = None

        if chain.root_node:
            self._chain_to_chain_template(chain)

    def _chain_to_chain_template(self, chain):

        def extract_chain_structure(node: Node, model_id: int):
            nonlocal counter

            if node.nodes_from:
                nodes_from = []
                for index, node_parent in enumerate(node.nodes_from):
                    if node_parent.descriptive_id in visited_nodes:
                        nodes_from.append(visited_nodes.index(node_parent.descriptive_id))
                    else:
                        counter += 1
                        visited_nodes.append(node_parent.descriptive_id)
                        nodes_from.append(counter)
                        extract_chain_structure(node_parent, counter)
            else:
                nodes_from = []

            model_template = ModelTemplate(node, str(model_id), sorted(nodes_from), self.unique_chain_id)

            self.model_templates.append(model_template)
            self._add_model_type_to_state(model_template.model_type)

            return model_template

        counter = 0
        visited_nodes = []
        extract_chain_structure(chain.root_node, counter)
        self.depth = chain.depth

    def _add_model_type_to_state(self, model_type: str):
        if model_type in self.total_model_types:
            self.total_model_types[model_type] += 1
        else:
            self.total_model_types[model_type] = 1

    def export_to_json(self, path: str):
        def create_unique_chain_id(path_to_save: str):
            name_of_files = path_to_save.split('/')
            last_file = name_of_files[-1].split('.')
            if len(last_file) == 2:
                if last_file[-1] == 'json':
                    if os.path.exists(path_to_save):
                        raise FileExistsError(f"File on path: '{os.path.abspath(path_to_save)}' exists")
                    self.unique_chain_id = ''.join(last_file[0:-1])
                    return '/'.join(name_of_files[0:-1])
                else:
                    raise JsonFileExtensionValidation(f"Could not save chain in"
                                                      f" '{last_file[-1]}' extension, use 'json'")
            else:
                self.unique_chain_id = str(uuid4())
                return path

        def create_absolute_path_if_not_exist(path_to_save: str):
            abs_path = os.path.abspath(path_to_save)
            if not os.path.exists(path_to_save):
                os.makedirs(abs_path)
            return abs_path

        path = create_unique_chain_id(path)
        absolute_path = create_absolute_path_if_not_exist(path)

        data = self.make_json()
        with open(os.path.join(absolute_path, f'{self.unique_chain_id}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(data), indent=4))
            print(f"The chain saved in the path: {os.path.join(absolute_path, f'{self.unique_chain_id}.json')}")

        return self

    def make_json(self):
        json_object = {
            "total_model_types": self.total_model_types,
            "depth": self.depth,
            "nodes": list(map(lambda model_template: model_template.export_to_json(), self.model_templates))
        }

        return json.dumps(json_object)

    def import_from_json(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File on the path: {path} does not exist.")

        def roll_chain_structure(node_object: dict) -> Node:
            if node_object['nodes_from']:
                if node_object['trained_model_path']:
                    pass
                    # TODO import fitted model
                else:
                    secondary_node = SecondaryNode(node_object['model_type'])
                    secondary_node.custom_params = node_object['params']
                    secondary_node.nodes_from = [roll_chain_structure(node_from) for node_from
                                                 in node_object['nodes_from']]
                    return secondary_node
            else:
                if node_object['trained_model_path']:
                    pass
                    # TODO import fitted model
                else:
                    primary_node = PrimaryNode(node_object['model_type'])
                    primary_node.custom_params = node_object['params']
                    primary_node.nodes_from = []
                    return primary_node

        with open(path) as json_file:
            json_object_chain = json.loads(json_file)
            print(json_object_chain)
            root_node = filter(lambda key: key.model_id == 0, json_object_chain['nodes'])
            print(root_node)
            exit(1)
            # root_node = roll_chain_structure(root_node)

    def _json_to_chain_template(self, chain_json):
        nodes_objects = chain_json['nodes']

        for node_object in nodes_objects:
            model_template = ModelTemplate(node_object)
            self._add_model_type_to_state(model_template.model_type)
            self.model_templates.append(model_template)

        self.depth = self._find_depth_chain_template()

    def _find_depth_chain_template(self):
        def recursive_traversal(node, counter=0):
            if node.nodes_from:
                for node_from in node.nodes_from:
                    return recursive_traversal(node_from, counter + 1)
            return counter

        return max([recursive_traversal(node) for node in self.model_templates])


class ModelTemplate:
    # TODO issues_1: make decision get name of model and full
    #  params from different types of model (sklearns, statmodels)
    def __init__(self, node: Node, model_id: str, nodes_from: list, chain_id: str):
        self.model_id = None
        self.model_type = None
        self.model_name = None
        self.custom_params = None
        self.full_params = None
        self.nodes_from = None
        self.fitted_model_path = None

        self._model_to_model_template(node, model_id, nodes_from, chain_id)

    def _model_to_model_template(self, node: Node, model_id: str, nodes_from: list, chain_id: str):
        self.model_id = model_id
        self.model_type = node.model.model_type
        self.custom_params = node.model.params
        self.full_params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if node.cache.actual_cached_state:
            # TODO issues_1
            self.model_name = node.cache.actual_cached_state.model.__class__.__name__
            self.fitted_model_path = os.path.join(DEFAULT_FITTED_MODELS_PATH, chain_id, 'model_'
                                                  + str(self.model_id) + '.pkl')
            joblib.dump(node.cache.actual_cached_state.model, self.fitted_model_path)

    def _create_full_params(self, node: Node) -> dict:
        full_params = {}
        if node.cache.actual_cached_state:
            # TODO issues_1
            full_params = node.cache.actual_cached_state.model.get_params()
            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    full_params[key] = value

        return full_params

    def export_to_json(self) -> object:

        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "custom_params": self.custom_params,
            "full_params": self.full_params,
            "nodes_from": self.nodes_from,
            "trained_model_path": self.fitted_model_path,
        }

        return model_object

    # def _json_to_model_template(self, model_object: object):
    #     self.model_id = model_object['model_id']
    #     self.model_type = model_object['model_type']
    #     self.custom_params = model_object['custom_params']
    #     self.nodes_from = model_object['nodes_from']
