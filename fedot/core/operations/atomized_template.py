import os
from typing import Tuple

from fedot.core.operations.operation_template import OperationTemplateAbstract, check_existing_path
from fedot.core.pipelines.node import Node


class AtomizedModelTemplate(OperationTemplateAbstract):
    def __init__(self, node: Node = None, operation_id: int = None, nodes_from: list = None, path: str = None):
        # Need use the imports inside the class because of the problem of circular imports.
        from fedot.core.pipelines.pipeline import Pipeline
        from fedot.core.pipelines.template import PipelineTemplate
        from fedot.core.operations.atomized_model import AtomizedModel

        super().__init__()
        self.atomized_model_json_path = None
        self.next_pipeline_template = None
        self.pipeline_template = None

        if path:
            pipeline = Pipeline.from_serialized(path)
            self.next_pipeline_template = AtomizedModel(pipeline)
            self.pipeline_template = PipelineTemplate(pipeline)

        if node:
            self._operation_to_template(node, operation_id, nodes_from)

    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list):
        from fedot.core.pipelines.template import PipelineTemplate

        self.operation_id = operation_id
        self.operation_type = node.operation.operation_type
        self.nodes_from = nodes_from
        self.pipeline_template = PipelineTemplate(node.operation.pipeline)
        self.atomized_model_json_path = 'nested_' + str(self.operation_id)

    def convert_to_dict(self) -> dict:

        operation_object = {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type,
            'nodes_from': self.nodes_from,
            'atomized_model_json_path': self.atomized_model_json_path
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
        check_existing_path(absolute_path)
        self.pipeline_template.export_pipeline(absolute_path, create_subdir=False)

    def import_json(self, operation_object: dict):
        required_fields = ['operation_id', 'operation_type', 'nodes_from', 'atomized_model_json_path']
        self._validate_json_operation_template(operation_object, required_fields)

        self.operation_id = operation_object['operation_id']
        self.operation_type = operation_object['operation_type']
        self.nodes_from = operation_object['nodes_from']
        self.atomized_model_json_path = operation_object['atomized_model_json_path']
