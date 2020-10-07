import json
import os
from uuid import uuid4

from sklearn.externals import joblib

from core.composer.node import PrimaryNode, SecondaryNode, Node

ROOT_DIR = os.path.dirname(os.path.abspath("core"))
FITTED_DIR = os.path.join(ROOT_DIR, 'fitted_chains')


class ChainTemplate:
    def __init__(self, chain):
        self.total_model_types = {}
        self.depth = None
        self.model_templates = []
        # TODO think about new fields
        # self.number_primary_nodes = None
        # self.number_secondary_nodes = None
        # self.task_type = None
        # self.training_time = None
        # self.input_shape = None
        # self.accuracy = None

        self._chain_to_chain_template(chain)

    def _add_model_type_to_state(self, model_type):
        if model_type in self.total_model_types:
            self.total_model_types[model_type] += 1
        else:
            self.total_model_types[model_type] = 1

    def _chain_to_chain_template(self, chain):

        def extract_chain_structure(node: Node, model_id):
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

            model_template = ModelTemplate(node, model_id, sorted(nodes_from))

            self.model_templates.append(model_template)
            self._add_model_type_to_state(model_template.model_type)

            return model_template

        counter = 0
        visited_nodes = []
        extract_chain_structure(chain.root_node, counter)
        self.depth = chain.depth

    def make_json(self, path: str):
        json_object = {
            "total_model_types": self.total_model_types,
            "depth": self.depth,
            "nodes": list(map(lambda model_template: model_template.export_to_json(path), self.model_templates))
        }

        return json.dumps(json_object)

    def export_to_json(self, file_name: str):
        if not file_name:
            file_name = str(uuid4())

        path = os.path.join(FITTED_DIR, file_name)

        if os.path.exists(path):
            raise FileExistsError(f"Chain with name {file_name} exist")

        os.makedirs(path)

        data = self.make_json(path)
        full_path = os.path.join(path, file_name + '.json')
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(data), indent=4))
            print(f"The chain saved in the path: {full_path}")

        return self

    # def _json_to_chain_template(self, chain_json):
    #     nodes_objects = chain_json['nodes']
    #
    #     for node_object in nodes_objects:
    #         model_template = ModelTemplate(node_object)
    #         self._add_model_type_to_state(model_template.model_type)
    #         self.nodes.append(model_template)
    #
    #     self.depth = self._find_depth_chain_template()

    # def import_from_json(self, path):
    #    if not os.path.exists(path):
    #        raise FileNotFoundError
    #
    #     def roll_chain_structure(node_object: dict) -> Node:
    #         if node_object['nodes_from']:
    #             if node_object['trained_model_path']:
    #                 pass
    #                 # TODO import fitted model
    #             else:
    #                 secondary_node = SecondaryNode(node_object['model_type'])
    #                 secondary_node.custom_params = node_object['params']
    #                 secondary_node.nodes_from = [roll_chain_structure(node_from) for node_from
    #                                              in node_object['nodes_from']]
    #                 return secondary_node
    #         else:
    #             if node_object['trained_model_path']:
    #                 pass
    #                 # TODO import fitted model
    #             else:
    #                 primary_node = PrimaryNode(node_object['model_type'])
    #                 primary_node.custom_params = node_object['params']
    #                 primary_node.nodes_from = []
    #                 return primary_node
    #
    #     chain = Chain()
    #     # nodes = json.loads(chain_json)['nodes']
    #     root_node = None

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
    def __init__(self, node: Node, model_id: str, nodes_from: list = None):
        self.model_id = None
        self.model_type = None
        self.model_name = None
        self.custom_params = None
        self.full_params = None
        self.nodes_from = None
        self.fitted_model_path = None
        # TODO We need to store the trained model in the model template until we export the fitted model in the h5 file.
        #  In fact, then the whole meaning of templates is lost. We wanted them to be light, but before saving them,
        #  they must store the object of the trained model.
        #  Even after saving, template_model should also store the fitted model, since the user will want to save the
        #  chain again, but with a different name for example.
        #  The problem is solved using links for fitted models. I don't know yet how to implement links to fitted
        #  models in Python.
        self.fitted_model = None

        self._model_to_model_template(node, model_id, nodes_from)

    def _model_to_model_template(self, node: Node, model_id: str, nodes_from: list):
        self.model_id = model_id
        self.model_type = node.model.model_type
        self.custom_params = node.model.params
        self.full_params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if node.cache.actual_cached_state:
            # TODO issues_1
            self.model_name = node.cache.actual_cached_state.model.__class__.__name__
            self.fitted_model = node.cache.actual_cached_state.model

    def _create_full_params(self, node: Node) -> dict:
        full_params = {}
        if node.cache.actual_cached_state:
            # TODO issues_1
            full_params = node.cache.actual_cached_state.model.get_params()
            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    full_params[key] = value

        return full_params

    def export_to_json(self, path: str) -> object:
        if self.fitted_model:
            self.fitted_model_path = os.path.join(path + '/model_' + str(self.model_id)) + '.pkl'
            joblib.dump(self.fitted_model, self.fitted_model_path)

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
