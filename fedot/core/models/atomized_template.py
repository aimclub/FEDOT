import os
from typing import Tuple

from fedot.core.chains.node import Node
from fedot.core.models.model_template import ModelTemplateAbstract, _check_existing_path


class AtomizedModelTemplate(ModelTemplateAbstract):
    def __init__(self, node: Node = None, model_id: int = None, nodes_from: list = None, path: str = None):
        # Need use the imports inside the class because of the problem of circular imports.
        from fedot.core.chains.chain import Chain
        from fedot.core.chains.chain_template import ChainTemplate
        from fedot.core.models.atomized_model import AtomizedModel

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
            self._model_to_template(node, model_id, nodes_from)

    def _model_to_template(self, node: Node, model_id: int, nodes_from: list):
        from fedot.core.chains.chain_template import ChainTemplate

        self.model_id = model_id
        self.model_type = node.model.model_type
        self.nodes_from = nodes_from
        self.chain_template = ChainTemplate(node.model.chain)
        self.atomized_model_json_path = 'nested_' + str(self.model_id)

    def convert_to_dict(self) -> dict:

        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "nodes_from": self.nodes_from,
            "atomized_model_json_path": self.atomized_model_json_path
        }

        return model_object

    def _create_nested_path(self, path: str) -> Tuple[str, str]:
        """
        Create folder for nested JSON model and prepared path to save JSON's.
        :params path: path where to save parent JSON model
        :return: absolute and relative paths to save nested JSON model
        """

        relative_path = os.path.join('fitted_models', 'nested_' + str(self.model_id))
        absolute_path = os.path.join(path, relative_path)

        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        return absolute_path, relative_path

    def export_model(self, path: str):
        absolute_path = os.path.join(path, self.atomized_model_json_path)
        _check_existing_path(absolute_path)
        self.chain_template.export_chain(absolute_path)

    def import_json(self, model_object: dict):
        required_fields = ['model_id', 'model_type', 'nodes_from', 'atomized_model_json_path']
        self._validate_json_model_template(model_object, required_fields)

        self.model_id = model_object['model_id']
        self.model_type = model_object['model_type']
        self.nodes_from = model_object['nodes_from']
        self.atomized_model_json_path = model_object['atomized_model_json_path']
