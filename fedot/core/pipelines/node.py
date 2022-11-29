from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from fedot.core.dag.linked_graph_node import LinkedGraphNode
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.log import default_log
from fedot.core.operations.factory import OperationFactory
from fedot.core.operations.operation import Operation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.optimisers.timer import Timer
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.serializers.serializer import register_serializable
from fedot.core.utils import DEFAULT_PARAMS_STUB


@register_serializable
@dataclass
class NodeMetadata:
    """Dataclass. :class:`Node` metadata

    Args:
        metric: quality score
    """

    metric: Optional[float] = None


class Node(LinkedGraphNode):
    """Base class for node definition in :class:`Pipeline` structure

    Args:
        nodes_from: parent nodes which information comes from
        operation_type: type of the operation defined in operation repository
            the custom prefix can be added after ``/`` (to highlight the specific node)\n
            **The prefix will be ignored at Implementation stage**
    """

    def __init__(self, nodes_from: Optional[List['Node']],
                 operation_type: Optional[Union[str, 'Operation']] = None, **kwargs):
        passed_content = kwargs.get('content')
        if passed_content:
            # Define operation, based on content dictionary
            operation = self._process_content_init(passed_content)
            params = passed_content.get('params', {})
            # The check for "default_params" is needed for backward compatibility.
            if params == DEFAULT_PARAMS_STUB:
                params = {}
            self.metadata = passed_content.get('metadata', NodeMetadata())
        else:
            # There is no content for node
            operation = self._process_direct_init(operation_type)

            # Define operation with default parameters
            params = {}
            self.metadata = NodeMetadata()

        self.fit_time_in_seconds = 0
        self.inference_time_in_seconds = 0

        self._parameters = OperationParameters.from_operation_type(operation.operation_type, **params)
        super().__init__(content={'name': operation,
                                  'params': self._parameters.to_dict()}, nodes_from=nodes_from)
        self.log = default_log(self)
        self._fitted_operation = None
        self.rating = None

    def _process_content_init(self, passed_content: dict) -> Operation:
        """ Updating content in the node """
        if isinstance(passed_content['name'], str):
            # Need to convert name of operation into operation class object
            operation_factory = OperationFactory(operation_name=passed_content['name'])
            operation = operation_factory.get_operation()

            passed_content.update({'name': operation})
        else:
            operation = passed_content['name']
        self.content = passed_content

        return operation

    @staticmethod
    def _process_direct_init(operation_type: Optional[Union[str, Operation]]) -> Operation:
        """Define operation based on the direct ``operation_type`` without defining content in the node

        Args:
            operation_type: node type representation

        Returns:
            Operation: operation class object
        """
        if not operation_type:
            raise ValueError('Operation is not defined in the node')

        if not isinstance(operation_type, str):
            # AtomizedModel
            operation = operation_type
        else:
            # Define appropriate operation or data operation
            operation_factory = OperationFactory(operation_name=operation_type)
            operation = operation_factory.get_operation()

        return operation

    def update_params(self):
        """Updates :attr:`custom_params` with changed parameters"""
        new_params = self.fitted_operation.get_params()
        changed_parameters = new_params.changed_parameters
        updated_parameters = {**self.parameters, **changed_parameters}
        self.parameters = updated_parameters

    @property
    def name(self) -> str:
        """ Returns str name of operation """
        return self.operation.operation_type

    @property
    def operation(self) -> Operation:
        """Returns node operation object

        Returns:
            Operation: operation object
        """
        return self.content['name']

    @operation.setter
    def operation(self, value: Operation):
        """Updates ``operation`` property with the provided ``value``

        Args:
            value: new operation object
        """

        self.content.update({'name': value})

    @property
    def fitted_operation(self) -> Optional[Any]:
        """Returns already fitted operation if exists or ``None`` instead

        Returns:
            node fitted operation or ``None``
        """

        return getattr(self, '_fitted_operation', None)

    @fitted_operation.setter
    def fitted_operation(self, value: Any):
        """Sets node fitted operation with the provided ``value``

        Args:
            value: any model from the ``list`` of acceptable nodes for the chosen task and problem
        """

        if value is None:
            if hasattr(self, '_fitted_operation'):
                del self._fitted_operation
        else:
            self._fitted_operation = value

    def unfit(self):
        """Sets :obj:`fitted_operation` to ``None``

        Todo:
            check how it would be rendered
        """

        self.fitted_operation = None

    def fit(self, input_data: InputData) -> OutputData:
        """Runs training process in the node

        Args:
            input_data: data used for operation training

        Returns:
            OutputData: values predicted on the provided ``input_data``
        """
        if self.fitted_operation is None:
            with Timer() as t:
                self.fitted_operation, operation_predict = self.operation.fit(params=self._parameters,
                                                                              data=input_data)
                self.fit_time_in_seconds = round(t.seconds_from_start, 3)
        else:

            operation_predict = self.operation.predict_for_fit(fitted_operation=self.fitted_operation,
                                                               data=input_data,
                                                               params=self._parameters)

        # Update parameters after operation fitting (they can be corrected)
        not_atomized_operation = 'atomized' not in self.operation.operation_type

        if not_atomized_operation and 'correct_params' in self.operation.metadata.tags:
            self.update_params()
        return operation_predict

    def predict(self, input_data: InputData, output_mode: str = 'default') -> OutputData:
        """Runs prediction process in the node

        Args:
            input_data: data used for prediction
            output_mode: desired output for operations (e.g. ``'labels'``, ``'probs'``, ``'full_probs'``)

        Returns:
            OutputData: values predicted on the provided ``input_data``
        """

        with Timer() as t:
            operation_predict = self.operation.predict(fitted_operation=self.fitted_operation,
                                                       params=self._parameters,
                                                       data=input_data,
                                                       output_mode=output_mode)
            self.inference_time_in_seconds = round(t.seconds_from_start, 3)
        return operation_predict

    @property
    def parameters(self) -> dict:
        """Returns node custom parameters

        Returns:
            dict: of custom parameters
        """
        return self.content.get('params')

    @parameters.setter
    def parameters(self, params: dict):
        """Sets custom parameters of the node

        Args:
            params: new parameters to be placed instead of existing
        """
        if params:
            # The check for "default_params" is needed for backward compatibility.
            if params == DEFAULT_PARAMS_STUB:
                params = {}
            # take nested composer params if they appeared
            if 'nested_space' in params:
                params = params['nested_space']
            self._parameters = OperationParameters.from_operation_type(self.operation.operation_type, **params)
            self.content['params'] = self._parameters.to_dict()

    def __str__(self) -> str:
        """Returns ``str`` representation of the node

        Returns:
            str: stringified node operation type
        """

        return str(self.operation.operation_type)

    @property
    def tags(self) -> Optional[List[str]]:
        """Returns tags of operation in the node or empty list

        Returns:
            Optional[List[str]]: ``empty list`` if node is of atomized type and ``list of tags`` otherwise
        """

        if 'atomized' in self.operation.operation_type:
            # There are no tags for atomized operation
            return []

        info = OperationTypesRepository(operation_type='all').operation_info_by_id(self.operation.operation_type)
        if info is not None:
            return info.tags


class PrimaryNode(Node):
    """Defines the interface of primary nodes where initial task data is located

    Args:
        operation_type: operation defined in the operation repository
        node_data: ``dict`` with :class:`InputData` for fit and predict stage
        kwargs: optional arguments (i.e. logger); ``nodes_from`` will be deleted if exists
    """

    def __init__(self, operation_type: Optional[Union[str, 'Operation']] = None, node_data: Optional[dict] = None,
                 **kwargs):
        if 'nodes_from' in kwargs:
            del kwargs['nodes_from']

        super().__init__(nodes_from=None, operation_type=operation_type, **kwargs)

        if node_data is None:
            self._node_data = {}
            self.direct_set = False
        else:
            self._node_data = node_data
            # Was the data passed directly to the node or not
            self.direct_set = True

    def fit(self, input_data: InputData, **kwargs) -> OutputData:
        """Fits the operation located in the primary node

        Args:
            input_data: data used for operation training

        Returns:
            OutputData: values predicted on the provided ``input_data``
        """

        self.log.debug(f'Trying to fit primary node with operation: {self.operation}')

        if self.direct_set:
            input_data = self.node_data
        else:
            self.node_data = input_data
        return super().fit(input_data)

    def unfit(self):
        """Sets ``node_data`` (if exists) and ``fitted_operation`` to ``None``
        """

        self.fitted_operation = None
        if hasattr(self, 'node_data'):
            self.node_data = None

    def predict(self, input_data: InputData, output_mode: str = 'default') -> OutputData:
        """Predicts using the operation located in the primary node

        Args:
            input_data: data used for prediction
            output_mode: desired output for operations (e.g. ``'labels'``, ``'probs'``, ``'full_probs'``)

        Returns:
            OutputData: values predicted on the provided ``input_data``
        """

        self.log.debug(f'Predict in primary node by operation: {self.operation}')

        if self.direct_set:
            input_data = self.node_data
        else:
            self.node_data = input_data
        return super().predict(input_data, output_mode)

    def get_data_from_node(self) -> dict:
        """Returns data if it was set to the nodes directly

        Returns:
            dict: ``dict`` with :class:`InputData` for fit and predict stage
        """

        return self.node_data

    @property
    def node_data(self) -> dict:
        """Returns directly set :attr:`node_data`

        Returns:
            dict: ``dict`` with :class:`InputData` for fit and predict stage
        """

        return getattr(self, '_node_data', {})

    @node_data.setter
    def node_data(self, value: dict):
        """Sets :attr:`node_data`

        Args:
            value: ``dict`` with :class:`InputData` for fit and predict stage
        """

        if value is None:
            if hasattr(self, '_node_data'):
                del self._node_data
        else:
            self._node_data = value


class SecondaryNode(Node):
    """The class defines the interface of ``Secondary`` nodes modifying tha data flow in the :class:`Pipeline`

    Args:
        operation_type: operation defined in the operation repository
        nodes_from: parent nodes where data comes from
        kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, operation_type: Optional[Union[str, 'Operation']] = None,
                 nodes_from: Optional[List['Node']] = None, **kwargs):
        super().__init__(nodes_from=nodes_from, operation_type=operation_type, **kwargs)

    def fit(self, input_data: InputData, **kwargs) -> OutputData:
        """Fits the operation located in the secondary node

        Args:
            input_data: data used for operation training

        Returns:
            OutputData: values predicted on the provided ``input_data``
        """

        self.log.debug(f'Trying to fit secondary node with operation: {self.operation}')

        secondary_input = self._input_from_parents(input_data=input_data, parent_operation='fit')
        return super().fit(input_data=secondary_input)

    def predict(self, input_data: InputData, output_mode: str = 'default') -> OutputData:
        """Predicts using the operation located in the secondary node

        Args:
            input_data: data used for prediction
            output_mode: desired output for operations (e.g. ``'labels'``, ``'probs'``, ``'full_probs'``)

        Returns:
            OutputData: values predicted on the provided ``input_data``
        """

        self.log.debug(f'Obtain prediction in secondary node with operation: {self.operation}')

        secondary_input = self._input_from_parents(input_data=input_data,
                                                   parent_operation='predict')

        return super().predict(input_data=secondary_input, output_mode=output_mode)

    def _input_from_parents(self, input_data: InputData, parent_operation: str) -> InputData:
        """Processes all the parent nodes via the current operation using ``input_data``

        Args:
            input_data: input data from pipeline abstraction (source input data)
            parent_operation: name of parent operation (``'fit'`` or ``'predict'``)

        Returns:
            InputData: predictions from the secondary nodes
        """

        if len(self.nodes_from) == 0:
            raise ValueError('No parent nodes found')

        self.log.debug(f'Fit all parent nodes in secondary node with operation: {self.operation}')

        parent_nodes = self._nodes_from_with_fixed_order()

        parent_results, _ = _combine_parents(parent_nodes, input_data,
                                             parent_operation)
        secondary_input = DataMerger.get(parent_results).merge()
        # Update info about visited nodes
        parent_operations = [node.operation.operation_type for node in parent_nodes]
        secondary_input.supplementary_data.previous_operations = parent_operations
        return secondary_input

    def _nodes_from_with_fixed_order(self):
        """Sorts :attr:`nodes_from` (if exists) by the nodes unique id

        Returns:
            sorted :attr:`nodes_from` by :obj:`GraphNode.descriptive_id` or ``None``
        """
        return sorted(self.nodes_from, key=lambda node: node.descriptive_id)


def _combine_parents(parent_nodes: List[Node],
                     input_data: Optional[InputData], parent_operation: str) -> Tuple[List[OutputData], np.array]:
    """Сombines predictions from the ``parent_nodes``

    Args:
        parent_nodes: list of parent nodes, from which predictions will be combined
        input_data: input data from pipeline abstraction (source input data)
        parent_operation: name of parent operation (``'fit'`` or ``'predict'``)

    Returns:
        Tuple[List[OutputData], np.array]: :obj:`output data list from parent nodes`,
        :obj:`target for final pipeline prediction`
    """

    if input_data is not None:
        # InputData was set to pipeline
        target = input_data.target
    parent_results = []
    for parent in parent_nodes:

        if parent_operation == 'predict':
            prediction = parent.predict(input_data=input_data)
            parent_results.append(prediction)
        elif parent_operation == 'fit':
            prediction = parent.fit(input_data=input_data)
            parent_results.append(prediction)
        else:
            raise NotImplementedError()

        if input_data is None:
            # InputData was set to primary nodes
            target = prediction.target

    return parent_results, target
