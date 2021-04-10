import os
from typing import Tuple

from fedot.core.chains.node import Node
from fedot.core.operations.operation_template import OperationTemplateAbstract, _check_existing_path


class AtomizedModelTemplate(OperationTemplateAbstract):
    def __init__(self, node: Node = None, operation_id: int = None, nodes_from: list = None, path: str = None):
        # Need use the imports inside the class because of the problem of circular imports.
        from fedot.core.chains.chain import Chain
        from fedot.core.chains.chain_template import ChainTemplate
        from fedot.core.operations.atomized_model import AtomizedModel

        super().__init__()
        self.atomized_model_json_path = None
        self.next_chain_template = None
        self.chain_template = None

        if path:
            chain = Chain()
            chain.load(path)
            self.next_chain_template = AtomizedModel(chain)
            self.chain_template = ChainTemplate(chain)

        if node:
            self._operation_to_template(node, operation_id, nodes_from)

    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list):
        from fedot.core.chains.chain_template import ChainTemplate

        self.operation_id = operation_id
        self.operation_type = node.operation.operation_type
        self.nodes_from = nodes_from
        self.chain_template = ChainTemplate(node.operation.chain)
        self.atomized_model_json_path = 'nested_' + str(self.operation_id)

    def convert_to_dict(self) -> dict:

        operation_object = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "nodes_from": self.nodes_from,
            "atomized_model_json_path": self.atomized_model_json_path
        }

        return operation_object

    def _create_nested_path(self, path: str) -> Tuple[str, str]:
        """
        Create folder for nested JSON operation and prepared path to save JSON's.
        :params path: path where to save parent JSON operation
        :return: absolute and relative paths to save nested JSON operation
        """

        relative_path = os.path.join('fitted_operations', 'nested_' + str(self.operation_id))
        absolute_path = os.path.join(path, relative_path)

        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        return absolute_path, relative_path

    def export_operation(self, path: str):
        absolute_path = os.path.join(path, self.atomized_model_json_path)
        _check_existing_path(absolute_path)
        self.chain_template.export_chain(absolute_path)

    def import_json(self, operation_object: dict):
        required_fields = ['operation_id', 'operation_type', 'nodes_from', 'atomized_model_json_path']
        self._validate_json_operation_template(operation_object, required_fields)

        self.operation_id = operation_object['operation_id']
        self.operation_type = operation_object['operation_type']
        self.nodes_from = operation_object['nodes_from']
        self.atomized_model_json_path = operation_object['atomized_model_json_path']
