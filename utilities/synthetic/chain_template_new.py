import json

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode, Node


class ChainTemplate:
    def __init__(self, instance):
        self.total_model_types = {}
        self.depth = None
        self.nodes = []
        # TODO think about new fields
        # self.number_primary_nodes = None
        # self.number_secondary_nodes = None
        # self.task_type = None
        # self.training_time = training_time
        # self.input_shape = input_shape

        if isinstance(instance, Chain):
            self._chain_to_chain_template(instance)
        elif isinstance(instance, object):
            self._json_to_chain_template(instance)

    def add_model_type_to_state(self, model_type):
        if model_type in self.total_model_types:
            self.total_model_types[model_type] += 1
        else:
            self.total_model_types[model_type] = 1

    def _chain_to_chain_template(self, chain: Chain):

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

            self.nodes.append(model_template)
            self.add_model_type_to_state(model_template.model_type)

            return model_template

        counter = 0
        visited_nodes = []
        extract_chain_structure(chain.root_node, counter)
        self.depth = chain.depth

    def export_to_json(self):
        json_object = {
            "total_model_types": self.total_model_types,
            "depth": self.depth,
            "nodes": list(map(lambda model_template: model_template.export_to_json(), self.nodes))
        }

        return json.dumps(json_object)

    def _json_to_chain_template(self, chain_json):
        nodes_objects = chain_json['nodes']

        for node_object in nodes_objects:
            model_template = ModelTemplate(node_object)
            self.add_model_type_to_state(model_template.model_type)
            self.nodes.append(model_template)

        self.depth = self.find_depth_chain_template()

    def json_to_chain(self):

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

        chain = Chain()
        # nodes = json.loads(chain_json)['nodes']
        root_node = None

    def find_depth_chain_template(self):
        def recursive_traversal(node, counter=0):
            if node.nodes_from:
                for node_from in node.nodes_from:
                    return recursive_traversal(node_from, counter + 1)
            return counter

        return max([recursive_traversal(node) for node in self.nodes])


class ModelTemplate:
    def __init__(self, instance, model_id: str = None, nodes_from: list = None):
        self.model_id = None
        self.model_type = None
        self.custom_params = None
        self.full_params = None
        self.nodes_from = None
        self.trained_model_path = None

        if isinstance(instance, Node):
            self._model_to_model_template(instance, model_id, nodes_from)
        elif isinstance(instance, object):
            self._json_to_model_template(instance)

    def _model_to_model_template(self, node: Node, model_id: str, nodes_from: list):
        self.model_id = model_id
        self.model_type = node.model.model_type
        self.custom_params = node.model.params
        self.full_params = self._create_full_params(node)
        self.nodes_from = nodes_from
        # TODO where located trained_model_path (node.cache.actual_cached_state)
        # self.trained_model_path = None

    def _create_full_params(self, node) -> dict:
        cached_state = node.cache.actual_cached_state

        if cached_state and isinstance(self.custom_params, dict):
            full_params = cached_state.model.get_params()
            for key, value in self.custom_params.items():
                full_params[key] = value
        else:
            full_params = "chain not fitted"

        return full_params

    def export_to_json(self) -> object:
        model_object = {
            "model_id": self.model_id,
            # TODO full name of model_type
            "model_type": self.model_type,
            "custom_params": self.custom_params,
            "full_params": self.full_params,
            "nodes_from": self.nodes_from,
            # "trained_model_path": self.trained_model_path,
        }

        return model_object

    def _json_to_model_template(self, model_object: object):
        self.model_id = model_object['model_id']
        self.model_type = model_object['model_type']
        self.custom_params = model_object['custom_params']
        self.nodes_from = model_object['nodes_from']

        return 0
