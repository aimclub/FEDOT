import uuid
import json

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode, Node
from core.models.model import DEFAULT_PARAMS_STUB


class ChainTemplate:
    def __init__(self, chain: Chain = None):
        self.total_model_types = {}
        self.depth = None
        self.nodes = []
        # self.number_primary_nodes = None
        # self.number_secondary_nodes = None
        # self.task_type = None
        # self.training_time = training_time
        # self.input_shape = input_shape

        if chain:
            self._chain_to_chain_template(chain)

    def _chain_to_chain_template(self, chain: Chain):
        def add_model_type_to_state(model_type):
            if model_type in self.total_model_types:
                self.total_model_types[model_type] += 1
            else:
                self.total_model_types[model_type] = 1

        def extract_chain_structure(node: Node):
            model_template = ModelTemplate(node)
            add_model_type_to_state(model_template.model_type)

            if node.nodes_from:
                nodes_from = [extract_chain_structure(node) for node in node.nodes_from]
            else:
                nodes_from = []

            model_template.nodes_from = list(map(lambda node_from: node_from.model_id, nodes_from))

            if model_template not in self.nodes:
                self.nodes.append(model_template)

            return model_template

        extract_chain_structure(chain.root_node)
        self.depth = chain.depth

    def to_json(self):
        json_object = {
            "total_model_types": self.total_model_types,
            "depth": self.depth,
            "nodes": list(map(lambda model_template: model_template.prepare_object(), self.nodes))
        }

        return json.dumps(json_object)

    def json_to_chain_template(self, chain_json):

        def roll_chain_structure(node_object: dict) -> Node:
            if node_object['nodes_from']:
                if node_object['trained_model_path']:
                    pass
                    # TODO import fitted model
                else:
                    secondary_node = SecondaryNode(node_object['model_type'])
                    if node_object['params']:
                        secondary_node.custom_params = node_object['params']
                    else:
                        secondary_node.custom_params = DEFAULT_PARAMS_STUB
                    secondary_node.nodes_from = [roll_chain_structure(node_from) for node_from
                                                 in node_object['nodes_from']]
                    return secondary_node
            else:
                if node_object['trained_model_path']:
                    pass
                    # TODO import fitted model
                else:
                    primary_node = PrimaryNode(node_object['model_type'])
                    if node_object['params']:
                        primary_node.custom_params = node_object['params']
                    else:
                        primary_node.custom_params = DEFAULT_PARAMS_STUB
                    primary_node.nodes_from = []
                    return primary_node

        chain = Chain()
        nodes = json.loads(chain_json)['nodes']
        root_node = None


class ModelTemplate:
    def __init__(self, node: Node):
        self.model_id = str(uuid.uuid4())
        self.model_type = node.model.model_type
        self.params = node.model.params
        # TODO where located trained_model_path
        # self.trained_model_path = node.cache.actual_cached_state
        self.nodes_from = None

    def prepare_object(self):
        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "params": self.params,
            "nodes_from": self.nodes_from,
            # "trained_model_path": self.trained_model_path,
        }

        return model_object

