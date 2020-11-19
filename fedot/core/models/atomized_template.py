import os

from fedot.core.chains.node import Node
from fedot.core.log import default_log, Log
from fedot.core.models.model_template import ModelTemplateAbstract


class AtomizedModelTemplate(ModelTemplateAbstract):
    def __init__(self, node: Node = None, model_id: int = None, nodes_from: list = None,
                 log: Log = default_log(__name__), path: str =None):
        from fedot.core.chains.chain import Chain
        from fedot.core.chains.chain_template import ChainTemplate
        from fedot.core.models.atomized_model import AtomizedModel

        super().__init__(log)
        self.atomized_model_json_path = None
        self.next_chain_template = None
        self.chain_template = None

        if path:
            chain = Chain()
            chain.load_chain(path)
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

    def convert_to_dict(self, path: str = None) -> dict:
        path_to_save = self._create_nested_path(path)

        self.chain_template.export_chain(path_to_save)
        self.atomized_model_json_path = path_to_save

        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "nodes_from": self.nodes_from,
            "atomized_model_json_path": self.atomized_model_json_path
        }

        return model_object

    def _create_nested_path(self, path: str) -> str:
        """
        Create folder for nested JSON model and prepared path to save JSON's.
        :params path: path where to save parent JSON model
        :return: absolute path to save nested JSON model
        """
        # prepare path
        split_path = path.split('/')
        name_of_parent_json = '.'.join(split_path[-1].split('.')[:-1])

        # create nested folder
        absolute_path_to_parent_dir = os.path.abspath(os.path.join('/'.join(split_path[:-1]), name_of_parent_json))
        if not os.path.exists(absolute_path_to_parent_dir):
            os.makedirs(absolute_path_to_parent_dir)

        # create name for JSON
        full_name_of_parent_json = 'nested_' + str(self.model_id) + '.json'
        absolute_path_to_parent_dir = os.path.join(absolute_path_to_parent_dir, full_name_of_parent_json)
        return absolute_path_to_parent_dir

    def import_json(self, model_object: dict):
        required_fields = ['model_id', 'model_type', 'nodes_from', 'atomized_model_json_path']
        self._validate_json_model_template(model_object, required_fields)

        self.model_id = model_object['model_id']
        self.model_type = model_object['model_type']
        self.nodes_from = model_object['nodes_from']
        self.atomized_model_json_path = model_object['atomized_model_json_path']
