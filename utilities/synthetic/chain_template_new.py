import json
import os
import joblib
from uuid import uuid4

from utilities.synthetic.get_application_settings_path import DEFAULT_PATH
from utilities.synthetic.custom_errors import JsonFileExtensionValidation, JsonFileInvalid

from core.composer.node import PrimaryNode, SecondaryNode, Node

DEFAULT_FITTED_MODELS_PATH = os.path.join(DEFAULT_PATH, 'fitted_models')


class ChainTemplate:
    def __init__(self, chain):
        self.total_model_types = {}
        self.depth = None
        self.model_templates = []
        self.unique_chain_id = str(uuid4())

        if chain.root_node:
            self._chain_to_chain_template(chain)
        else:
            self.link_to_empty_chain = chain

    def _chain_to_chain_template(self, chain):

        def extract_chain_structure(node: Node, model_id: int):
            nonlocal counter

            if node.nodes_from:
                nodes_from = []
                for index, node_parent in enumerate(node.nodes_from):
                    if node_parent.descriptive_id in visited_nodes:
                        nodes_from.append(visited_nodes.index(node_parent.descriptive_id) + 1)
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
            if len(last_file) >= 2:
                if last_file[-1] == 'json':
                    if os.path.exists(path_to_save):
                        raise FileExistsError(f"File on path: '{os.path.abspath(path_to_save)}' exists")
                    self.unique_chain_id = ''.join(last_file[0:-1])
                    return '/'.join(name_of_files[0:-1])
                else:
                    raise JsonFileExtensionValidation(f"Could not save chain in"
                                                      f" '{last_file[-1]}' extension, use 'json' format")
            else:
                return path_to_save

        path = create_unique_chain_id(path)
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        data = self.make_json()
        with open(os.path.join(absolute_path, f'{self.unique_chain_id}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(data), indent=4))
            print(f"The chain saved in the path: {os.path.join(absolute_path, f'{self.unique_chain_id}.json')}")

        return data

    def make_json(self):
        json_object = {
            "total_model_types": dict(sorted(self.total_model_types.items(), key=lambda x: x[0])),
            "depth": self.depth,
            "nodes": sorted(list(map(lambda model_template: model_template.export_to_json(), self.model_templates)),
                            key=lambda model: model['model_id'])
        }

        return json.dumps(json_object)

    def import_from_json(self, path: str):

        def check_for_current_path(path_to_load_chain: str):
            absolute_path = os.path.abspath(path)
            name_of_files = path_to_load_chain.split('/')
            last_file = name_of_files[-1].split('.')
            if len(last_file) >= 2:
                if last_file[-1] == 'json':
                    if not os.path.isfile(absolute_path):
                        raise FileNotFoundError(f"File on the path: {absolute_path} does not exist.")
                    self.unique_chain_id = ''.join(last_file[0:-1])
                else:
                    raise JsonFileExtensionValidation(f"Could not load chain in"
                                                      f" '{last_file[-1]}' extension, use 'json' format")
            else:
                raise FileNotFoundError(f"Write path to 'json' format file")

        def find_model_jsons_from_json_chain(json_chain, list_of_model_ids):
            node_objects = []
            for model_id in list_of_model_ids:
                node_objects += list(filter(lambda node_dict: node_dict['model_id'] == str(model_id), json_chain))

            return node_objects

        def roll_chain_structure(model_object: dict) -> Node:
            if model_object['model_id'] in visited_nodes:
                return visited_nodes[model_object['model_id']]
            if model_object['nodes_from']:
                node = SecondaryNode(model_object['model_type'])
            else:
                node = PrimaryNode(model_object['model_type'])

            node.model.params = model_object['params']
            node.nodes_from = [roll_chain_structure(node_from) for node_from
                               in find_model_jsons_from_json_chain(json_object_chain['nodes'],
                                                                   model_object['nodes_from'])]
            if "trained_model_path" in model_object and model_object['trained_model_path']:
                path_to_model = os.path.abspath(model_object['trained_model_path'])
                if not os.path.isfile(path_to_model):
                    raise FileNotFoundError(f"File on the path: {path_to_model} does not exist.")
                node.cache = joblib.load(path_to_model)
            visited_nodes[model_object['model_id']] = node
            return node

        check_for_current_path(path)

        with open(path) as json_file:
            json_object_chain = json.load(json_file)

        visited_nodes = {}
        self._json_to_chain_template(json_object_chain)
        root_node = find_model_jsons_from_json_chain(json_object_chain['nodes'], [0])[0]
        root_node = roll_chain_structure(root_node)
        self.link_to_empty_chain.add_node(root_node)
        self.depth = self.link_to_empty_chain.depth
        self.link_to_empty_chain = None

    def _json_to_chain_template(self, chain_json):
        model_objects = chain_json['nodes']

        for model_object in model_objects:
            model_template = ModelTemplate()
            model_template.json_to_model_template(model_object)
            self._add_model_type_to_state(model_template.model_type)
            self.model_templates.append(model_template)


class ModelTemplate:
    # TODO issues_1: make decision get name of model and full
    #  params from different types of model (sklearns, statmodels)
    def __init__(self, node: Node = None, model_id: str = None, nodes_from: list = None, chain_id: str = None):
        self.model_id = None
        self.model_type = None
        self.model_name = None
        self.custom_params = None
        self.params = None
        self.nodes_from = None
        self.fitted_model_path = None

        if node:
            self._model_to_model_template(node, model_id, nodes_from, chain_id)

    def _model_to_model_template(self, node: Node, model_id: str, nodes_from: list, chain_id: str):
        self.model_id = model_id
        self.model_type = node.model.model_type
        self.custom_params = node.model.params
        self.params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if node.cache.actual_cached_state:
            # TODO issues_1
            absolute_path = os.path.abspath(os.path.join(DEFAULT_FITTED_MODELS_PATH, chain_id))
            if not os.path.exists(absolute_path):
                os.makedirs(absolute_path)
            self.model_name = node.cache.actual_cached_state.model.__class__.__name__
            self.fitted_model_path = os.path.join(DEFAULT_FITTED_MODELS_PATH, chain_id, 'model_'
                                                  + str(self.model_id) + '.pkl')
            joblib.dump(node.cache.actual_cached_state.model, self.fitted_model_path)

    def _create_full_params(self, node: Node) -> dict:
        params = {}
        if node.cache.actual_cached_state:
            # TODO issues_1
            params = node.cache.actual_cached_state.model.get_params()
            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    params[key] = value

        return params

    def export_to_json(self) -> dict:

        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "trained_model_path": self.fitted_model_path,
        }

        return model_object

    def json_to_model_template(self, model_object: dict):
        try:
            self.model_id = model_object['model_id']
            self.model_type = model_object['model_type']
            self.params = model_object['params']
            self.nodes_from = model_object['nodes_from']
            if "trained_model_path" in model_object:
                self.fitted_model_path = model_object['trained_model_path']
            if "custom_params" in model_object:
                self.custom_params = model_object['custom_params']
            if "model_name" in model_object:
                self.model_name = model_object['model_name']
        except Exception:
            raise JsonFileInvalid(f"Required field 'model_id, model_type, params, nodes_from'")
