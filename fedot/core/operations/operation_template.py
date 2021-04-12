import os
from abc import ABC, abstractmethod

import joblib

from fedot.core.chains.node import Node
from fedot.core.log import Log, default_log


class OperationTemplateAbstract(ABC):
    """
    Base class used for create different types of operation ("atomized_operation"
    or others like ("knn", "xgboost")).
    Atomized_operation is chain which can be used like general operation.
    """

    def __init__(self, log: Log = None):
        self.operation_id = None
        self.operation_type = None
        self.nodes_from = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @abstractmethod
    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list):
        """
        Preprocessing for local fields
        :param node: current node
        :param operation_id: operation id in chain
        :param nodes_from: parents operation's id
        """

    @abstractmethod
    def import_json(self, operation_object: dict):
        """
        Parse JSON like object and fill local fields
        :param operation_object: JSON like object to parse
        """

    @abstractmethod
    def convert_to_dict(self) -> dict:
        """
        Transform all object's parameters to dictionary.
        :params path: string path to save.
        :return dict: dictionary with object parameters.
        """

    def _validate_json_operation_template(self, operation_object: dict, required_fields: list):
        """
        Check whether there are fields in the dictionary.
        :params operation_object: dictionary to check
        :params required_fields: list of fields name
        """

        for field in required_fields:
            if field not in operation_object:
                message = f"Required field '{field}' is expected, but not found."
                self.log.error(message)
                raise RuntimeError(message)


class OperationTemplate(OperationTemplateAbstract):
    def __init__(self, node: Node = None, operation_id: int = None,
                 nodes_from: list = None):
        super().__init__()
        self.operation_name = None
        self.custom_params = None
        self.params = None
        self.fitted_operation = None
        self.fitted_operation_path = None

        if node:
            self._operation_to_template(node, operation_id, nodes_from)

    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list):
        self.operation_id = operation_id
        self.operation_type = node.operation.operation_type
        self.custom_params = node.operation.params
        self.params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if _is_node_fitted(node):
            self.operation_name = _extract_operation_name(node)
            self._extract_fields_of_fitted_operation(node)

    def _create_full_params(self, node: Node) -> dict:
        wrapped_operations = ['base_estimator', 'estimator']

        params = {}
        if _is_node_fitted(node):
            params = extract_operation_params(node)

            # Check if it is needed to process "model in model" cases
            # such strategy is needed for RANSAC or RFE algorithms
            for wrapped_operation in wrapped_operations:
                if wrapped_operation in params:
                    del params[wrapped_operation]

            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    params[key] = value

        return params

    def _extract_fields_of_fitted_operation(self, node: Node):
        operation_name = f'operation_{str(self.operation_id)}.pkl'
        self.fitted_operation_path = os.path.join('fitted_operations', operation_name)
        self.fitted_operation = node.fitted_operation

    def convert_to_dict(self) -> dict:

        operation_object = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_name": self.operation_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "fitted_operation_path": self.fitted_operation_path
        }

        return operation_object

    def export_operation(self, path: str):
        _check_existing_path(path)

        if self.fitted_operation:
            path_fitted_operations = os.path.join(path, 'fitted_operations')
            _check_existing_path(path_fitted_operations)
            joblib.dump(self.fitted_operation, os.path.join(path, self.fitted_operation_path))

    def import_json(self, operation_object: dict):
        required_fields = ['operation_id', 'operation_type', 'params', 'nodes_from']
        self._validate_json_operation_template(operation_object, required_fields)

        self.operation_id = operation_object['operation_id']
        self.operation_type = operation_object['operation_type']
        self.params = operation_object['params']
        self.nodes_from = operation_object['nodes_from']
        if "fitted_operation_path" in operation_object:
            self.fitted_operation_path = operation_object['fitted_operation_path']
        if "custom_params" in operation_object:
            self.custom_params = operation_object['custom_params']
        if "operation_name" in operation_object:
            self.operation_name = operation_object['operation_name']


def _check_existing_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_operation_params(node: Node):
    return node.fitted_operation.get_params()


def _extract_operation_name(node: Node):
    return node.fitted_operation.__class__.__name__


def _is_node_fitted(node: Node) -> bool:
    return bool(node.fitted_operation)
