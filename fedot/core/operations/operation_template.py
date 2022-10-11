import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Optional, Union

import joblib
import numpy as np

from fedot.core.log import default_log
from fedot.core.pipelines.node import Node


class OperationTemplateAbstract(ABC):
    """
    Base class used for create different types of operation ("atomized_operation"
    or others like ("knn", "rf")).
    Atomized_operation is pipeline which can be used like general operation.
    """

    def __init__(self):
        self.operation_id = None
        self.operation_type = None
        self.nodes_from = None

        self.log = default_log(self)

    @abstractmethod
    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list):
        """
        Preprocessing for local fields
        :param node: current node
        :param operation_id: operation id in pipeline
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
        self.rating = None

        if node:
            self._operation_to_template(node, operation_id, nodes_from)

    def _operation_to_template(self, node: Node, operation_id: int, nodes_from: list):
        self.operation_id = operation_id
        if not isinstance(node.operation, str):
            # for model-based operations
            self.operation_type = node.operation.operation_type
            self.custom_params = node.parameters
            self.params = self._create_full_params(node)

            if _is_node_fitted(node):
                self.operation_name = _extract_operation_name(node)
                self._extract_fields_of_fitted_operation(node)
        else:
            # for custom operations without implementation
            self.operation_type = 'custom'
            self.custom_params = {}
            self.params = {}
            self.operation_name = node.operation
        self.nodes_from = nodes_from
        self.rating = node.rating

    def _create_full_params(self, node: Node) -> dict:
        wrapped_operations = ['base_estimator', 'estimator']

        params = {}
        if _is_node_fitted(node):
            params = extract_operation_params(node)

            # Check if it is needed to process "model in model" cases
            # such strategy is needed for RANSAC or RFE algorithms
            for wrapped_operation in wrapped_operations:
                if params is not None and wrapped_operation in params:
                    del params[wrapped_operation]

            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    params[key] = value

        return params

    def _extract_fields_of_fitted_operation(self, node: Node):
        if 'h2o' in self.operation_type:
            self.fitted_operation_path = os.path.join('fitted_operations', f"h2o_{self.operation_id}")
        else:
            operation_name = f'operation_{str(self.operation_id)}.pkl'
            self.fitted_operation_path = os.path.join('fitted_operations', operation_name)
        self.fitted_operation = node.fitted_operation

    def convert_to_dict(self) -> dict:
        # Convert compound path into separate files and directories
        if self.fitted_operation_path is not None and os.sep in self.fitted_operation_path:
            fitted_path = os.path.split(self.fitted_operation_path)
        else:
            fitted_path = self.fitted_operation_path

        operation_object = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_name": self.operation_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "fitted_operation_path": fitted_path,
            "rating": self.rating,
        }

        return operation_object

    def export_operation(self, return_path: str = None) -> Optional[Union[str, bytes, Any]]:
        """Export fitted operation: either by path or directly as bytes-like object.
        Returns None for not fitted operation.

        :param return_path: path to directory for exporting the operation.
        If None, then bytes-like object is returned directly.
        """
        if not self.fitted_operation:
            return None

        if return_path:
            check_existing_path(return_path)

            # dictionary with paths to saved fitted operations
            if 'h2o' in self.operation_type:
                self.fitted_operation_path = self.fitted_operation.save_operation(
                    os.path.join(return_path, 'fitted_operations'),
                    self.operation_id
                )
                return self.fitted_operation_path
            else:
                path_fitted_operations = os.path.join(return_path, 'fitted_operations')
                check_existing_path(path_fitted_operations)
                joblib.dump(self.fitted_operation, os.path.join(return_path, self.fitted_operation_path))
                return os.path.join(path_fitted_operations, f'operation_{self.operation_id}.pkl')
        else:
            # dictionary with bytes of fitted operations
            bytes_container = BytesIO()
            joblib.dump(self.fitted_operation, bytes_container)
            return bytes_container

    def import_json(self, operation_object: dict):
        required_fields = ['operation_id', 'operation_type', 'params', 'nodes_from']
        self._validate_json_operation_template(operation_object, required_fields)

        self.operation_id = operation_object['operation_id']
        self.operation_type = operation_object['operation_type']
        self.params = operation_object['params']
        if 'dtype' in self.params and isinstance(self.params['dtype'], str):
            self.params['dtype'] = np.dtype(self.params['dtype'])
        self.nodes_from = operation_object['nodes_from']
        if 'fitted_operation_path' in operation_object:
            fitted_operation_path = operation_object['fitted_operation_path']
            if isinstance(fitted_operation_path, str):
                self.fitted_operation_path = fitted_operation_path
            elif isinstance(fitted_operation_path, list):
                # Path were set as folders and files in tuple or list
                self.fitted_operation_path = os.path.join(*fitted_operation_path[:2])

        if 'custom_params' in operation_object:
            self.custom_params = operation_object['custom_params']
        if 'operation_name' in operation_object:
            self.operation_name = operation_object['operation_name']
        if 'rating' in operation_object:
            self.rating = operation_object['rating']


def check_existing_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_operation_params(node: Node):
    params = node.parameters

    if 'dtype' in params:
        params['dtype'] = params['dtype'].__name__

    return params


def _extract_operation_name(node: Node):
    return node.fitted_operation.__class__.__name__


def _is_node_fitted(node: Node) -> bool:
    return bool(node.fitted_operation)
