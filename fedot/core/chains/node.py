from abc import ABC
from copy import copy
from typing import List, Optional

from fedot.core.chains.node_operator import NodeOperator
from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.factory import OperationFactory
from fedot.core.operations.operation import Operation


class Node(ABC):
    """
    Base class for Node definition in Chain structure

    :param nodes_from: parent nodes which information comes from
    :param operation_type: str type of the operation defined in operation repository
                            the custom prefix can be added after / (to highlight the specific node)
                            The prefix will be ignored at Implementation stage
    :param log: Log object to record messages
    """

    def __init__(self, nodes_from: Optional[List['Node']],
                 operation_type: [str, 'Operation'],
                 log=None):
        self.nodes_from = nodes_from
        self.log = log
        self._fitted_operation = None
        self._operator = NodeOperator(self)

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if not isinstance(operation_type, str):
            # AtomizedModel
            self.operation = operation_type
        else:
            # Define appropriate operation or data operation
            self.operation_factory = OperationFactory(operation_name=operation_type)
            self.operation = self.operation_factory.get_operation()

    @property
    def descriptive_id(self):
        return self._descriptive_id_recursive(visited_nodes=[])

    def _descriptive_id_recursive(self, visited_nodes):
        """
        Method returns verbal description of the operation in the node
        and its parameters
        """

        node_label = self.operation.description
        full_path = ''
        if self in visited_nodes:
            return 'ID_CYCLED'
        visited_nodes.append(self)
        if self.nodes_from:
            previous_items = []
            for parent_node in self.nodes_from:
                previous_items.append(f'{parent_node._descriptive_id_recursive(copy(visited_nodes))};')
            previous_items.sort()
            previous_items_str = ';'.join(previous_items)

            full_path += f'({previous_items_str})'
        full_path += f'/{node_label}'
        return full_path

    @property
    def fitted_operation(self):
        if hasattr(self, '_fitted_operation'):
            return self._fitted_operation
        else:
            return None

    @fitted_operation.setter
    def fitted_operation(self, value):
        if value is None:
            if hasattr(self, '_fitted_operation'):
                del self._fitted_operation
        else:
            self._fitted_operation = value

    def unfit(self):
        self.fitted_operation = None

    def fit(self, input_data: InputData) -> OutputData:
        """
        Run training process in the node

        :param input_data: data used for operation training
        """

        if self.fitted_operation is None:
            self.fitted_operation, operation_predict = self.operation.fit(data=input_data,
                                                                          is_fit_chain_stage=True)
        else:
            operation_predict = self.operation.predict(fitted_operation=self.fitted_operation,
                                                       data=input_data,
                                                       is_fit_chain_stage=True)

        return operation_predict

    def predict(self, input_data: InputData, output_mode: str = 'default') -> OutputData:
        """
        Run prediction process in the node

        :param input_data: data used for prediction
        :param output_mode: desired output for operations (e.g. labels, probs, full_probs)
        """
        operation_predict = self.operation.predict(fitted_operation=self.fitted_operation,
                                                   data=input_data,
                                                   output_mode=output_mode,
                                                   is_fit_chain_stage=False)
        return operation_predict

    def __str__(self):
        operation = f'{self.operation}'
        return operation

    def __repr__(self):
        return self.__str__()

    def ordered_subnodes_hierarchy(self, visited=None) -> List['Node']:
        return self._operator.ordered_subnodes_hierarchy(visited)

    @property
    def distance_to_primary_level(self):
        return self._operator.distance_to_primary_level()

    @property
    def custom_params(self) -> dict:
        return self.operation.params

    @custom_params.setter
    def custom_params(self, params):
        if params:
            self.operation.params = params


class PrimaryNode(Node):
    """
    The class defines the interface of Primary nodes where initial task data is located

    :param operation_type: str type of the operation defined in operation repository
    :param node_data: dictionary with InputData for fit and predict stage
    :param kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, operation_type: [str, 'Operation'], node_data: dict = None, **kwargs):
        super().__init__(nodes_from=None, operation_type=operation_type, **kwargs)

        if node_data is None:
            self._node_data = {}
            self.direct_set = False
        else:
            self._node_data = node_data
            # Was the data passed directly to the node or not
            self.direct_set = True

    def fit(self, input_data: InputData) -> OutputData:
        """
        Fit the operation located in the primary node

        :param input_data: data used for operation training
        """
        self.log.ext_debug(f'Trying to fit primary node with operation: {self.operation}')

        if self.direct_set:
            input_data = self.node_data
        else:
            self.node_data = input_data
        return super().fit(input_data)

    def unfit(self):
        self.fitted_operation = None
        if hasattr(self, 'node_data'):
            self.node_data = None

    def predict(self, input_data: InputData,
                output_mode: str = 'default') -> OutputData:
        """
        Predict using the operation located in the primary node

        :param input_data: data used for prediction
        :param output_mode: desired output for operations (e.g. labels, probs, full_probs)
        """
        self.log.ext_debug(f'Predict in primary node by operation: {self.operation}')

        if self.direct_set:
            input_data = self.node_data
        else:
            self.node_data = input_data
        return super().predict(input_data, output_mode)

    def get_data_from_node(self):
        """ Method returns data if the data was set to the nodes directly """
        return self.node_data

    @property
    def node_data(self):
        if hasattr(self, '_node_data'):
            return self._node_data
        else:
            return {}

    @node_data.setter
    def node_data(self, value):
        if value is None:
            if hasattr(self, '_node_data'):
                del self._node_data
        else:
            self._node_data = value


class SecondaryNode(Node):
    """
    The class defines the interface of Secondary nodes modifying tha data flow in Chain

    :param operation_type: str type of the operation defined in operation repository
    :param nodes_from: parent nodes where data comes from
    :param operation: optional custom atomized_operation
    :param kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, operation_type: [str, 'Operation'], nodes_from: Optional[List['Node']] = None,
                 **kwargs):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, operation_type=operation_type,
                         **kwargs)

    def fit(self, input_data: InputData) -> OutputData:
        """
        Fit the operation located in the secondary node

        :param input_data: data used for operation training
        """
        self.log.ext_debug(f'Trying to fit secondary node with operation: {self.operation}')

        secondary_input = self._input_from_parents(input_data=input_data, parent_operation='fit')

        return super().fit(input_data=secondary_input)

    def predict(self, input_data: InputData, output_mode: str = 'default') -> OutputData:
        """
        Predict using the operation located in the secondary node

        :param input_data: data used for prediction
        :param output_mode: desired output for operations (e.g. labels, probs, full_probs)
        """
        self.log.ext_debug(f'Obtain prediction in secondary node with operation: {self.operation}')

        secondary_input = self._input_from_parents(input_data=input_data,
                                                   parent_operation='predict')

        return super().predict(input_data=secondary_input, output_mode=output_mode)

    def _nodes_from_with_fixed_order(self):
        if self.nodes_from is not None:
            return sorted(self.nodes_from, key=lambda node: node.descriptive_id)

    def _input_from_parents(self, input_data: InputData,
                            parent_operation: str) -> InputData:
        if len(self.nodes_from) == 0:
            raise ValueError()

        self.log.ext_debug(f'Fit all parent nodes in secondary node with operation: {self.operation}')

        parent_nodes = self._nodes_from_with_fixed_order()

        parent_results, target = _combine_parents(parent_nodes, input_data,
                                                  parent_operation)

        secondary_input = InputData.from_predictions(outputs=parent_results)

        return secondary_input


def _combine_parents(parent_nodes: List[Node],
                     input_data: InputData,
                     parent_operation: str):
    """
    Method for combining predictions from parent node or nodes

    :param parent_nodes: list of parent nodes, from which predictions will
    be combined
    :param input_data: input data from chain abstraction (source input data)
    :param parent_operation: name of parent operation (fit or predict)
    :return parent_results: list with OutputData from parent nodes
    :return target: target for final chain prediction
    """

    if input_data is not None:
        # InputData was set to chain
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
