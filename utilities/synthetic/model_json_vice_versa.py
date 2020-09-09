import json
from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode, Node
from core.models.model import DEFAULT_PARAMS_STUB


def serializing_json_from_chain(chain: Chain):

    def recursive_traversal(node: Node) -> dict:
        model_id = None
        model_type = node.model.model_type

        if node.model.params == DEFAULT_PARAMS_STUB:
            params = None
        else:
            params = node.model.params

        if node.nodes_from:
            nodes_from = [recursive_traversal(node) for node in node.nodes_from]
        else:
            nodes_from = []

        node_object = {'model_id': model_id, 'model_type': model_type, 'params': params, 'nodes_from': nodes_from}

        return node_object

    json_object = {'root_node': recursive_traversal(chain.root_node), 'depth': chain.depth}

    return json.dumps(json_object)


def deserializing_chain_from_json(json_object) -> Chain:

    def recursive_traversal(node_object: dict) -> Node:
        if node_object['nodes_from']:
            if node_object['model_id']:
                pass
                # TODO import fitted model
            else:
                secondary_node = SecondaryNode(node_object['model_type'])
                if node_object['params']:
                    secondary_node.custom_params = node_object['params']
                else:
                    secondary_node.custom_params = DEFAULT_PARAMS_STUB
                secondary_node.nodes_from = [recursive_traversal(node) for node in node_object['nodes_from']]
                return secondary_node
        else:
            if node_object['model_id']:
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
    root_node = json.loads(json_object)['root_node']
    chain.add_node(recursive_traversal(root_node))

    return chain
