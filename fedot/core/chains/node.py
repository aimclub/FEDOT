from abc import ABC
from collections import namedtuple
from copy import copy
from datetime import timedelta
from typing import List, Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.strategy import StrategyOperator

CachedState = namedtuple('CachedState', 'operation')


class Node(ABC):
    """
    Base class for Node definition in Chain structure

    :param nodes_from: parent nodes which information comes from
    :param operation_type: str type of the operation defined in operation repository
    :param log: Log object to record messages
    """

    def __init__(self, nodes_from: Optional[List['Node']], operation_type: str,
                 log=None):
        self.nodes_from = nodes_from
        self.cache = FittedOperationCache(self)

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        # Define appropriate model or data operation
        self.strategy_operator = StrategyOperator(operation_name=operation_type)
        self.operation = self.strategy_operator.get_operation()

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

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        """
        Run training process in the node

        :param input_data: data used for operation training
        :param verbose: flag used for status printing to console, default False
        """
        copied_input_data = copy(input_data)

        if not self.cache.actual_cached_state:
            if verbose:
                print('Cache is not actual')

            cached_operation, operation_predict = self.operation.fit(data=copied_input_data,
                                                                     is_fit_chain_stage=True)
            self.cache.append(CachedState(operation=cached_operation))
        else:
            if verbose:
                print('Operation were obtained from cache')

            fitted = self.cache.actual_cached_state.operation
            operation_predict = self.operation.predict(fitted_operation=fitted,
                                                       data=copied_input_data,
                                                       is_fit_chain_stage=True)

        return operation_predict

    def predict(self, input_data: InputData, output_mode, verbose=False) -> OutputData:
        """
        Run prediction process in the node

        :param input_data: data used for prediction
        :param output_mode: desired output for operations (e.g. labels, probs, full_probs)
        :param verbose: flag used for status printing to console, default False
        """
        copied_input_data = copy(input_data)

        if not self.cache:
            raise ValueError('Operation must be fitted before predict')

        fitted = self.cache.actual_cached_state.operation
        operation_predict = self.operation.predict(fitted_operation=fitted,
                                                   data=copied_input_data,
                                                   is_fit_chain_stage=False,
                                                   output_mode=output_mode,)

        return operation_predict

    def fine_tune(self, input_data: InputData,
                  max_lead_time: timedelta = timedelta(minutes=5),
                  iterations: int = 30):
        # TODO remove
        """
        Run the process of hyperparameter optimization for the node

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        """
        copied_input_data = copy(input_data)

        fitted_operation, _ = self.operation.fine_tune(copied_input_data,
                                                       max_lead_time=max_lead_time,
                                                       iterations=iterations)

        self.cache.append(CachedState(operation=fitted_operation))

    def __str__(self):
        operation = f'{self.operation}'
        return operation

    @property
    def ordered_subnodes_hierarchy(self) -> List['Node']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.ordered_subnodes_hierarchy
        return nodes

    @property
    def custom_params(self) -> dict:
        return self.operation.params

    @custom_params.setter
    def custom_params(self, params):
        if params:
            self.operation.params = params


class FittedOperationCache:
    def __init__(self, related_node: Node):
        self._local_cached_operations = {}
        self._related_node_ref = related_node

    def append(self, fitted_operation):
        self._local_cached_operations[self._related_node_ref.descriptive_id] = fitted_operation

    def import_from_other_cache(self, other_cache: 'FittedOperationCache'):
        for entry_key in other_cache._local_cached_operations.keys():
            self._local_cached_operations[entry_key] = other_cache._local_cached_operations[entry_key]

    def clear(self):
        self._local_cached_operations = {}

    @property
    def actual_cached_state(self):
        found_operation = self._local_cached_operations.get(self._related_node_ref.descriptive_id, None)
        return found_operation


class SharedCache(FittedOperationCache):
    def __init__(self, related_node: Node, global_cached_operations: dict):
        super().__init__(related_node)
        self._global_cached_operations = global_cached_operations

    def append(self, fitted_operation):
        super().append(fitted_operation)
        if self._global_cached_operations is not None:
            self._global_cached_operations[self._related_node_ref.descriptive_id] = fitted_operation

    @property
    def actual_cached_state(self):
        found_operation = super().actual_cached_state

        if not found_operation and self._global_cached_operations:
            found_operation = self._global_cached_operations.get(self._related_node_ref.descriptive_id, None)
        return found_operation


class PrimaryNode(Node):
    """
    The class defines the interface of Primary nodes where initial task data is located

    :param operation_type: str type of the operation defined in operation repository
    :param node_data: dictionary with InputData for fit and predict stage
    :param kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, operation_type: str, node_data: dict = None,
                 **kwargs):
        super().__init__(nodes_from=None, operation_type=operation_type, **kwargs)

        if node_data is None:
            self.node_data = {}
            self.direct_set = False
        else:
            self.node_data = node_data
            # Was the data passed directly to the node or not
            self.direct_set = True

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        """
        Fit the operation located in the primary node

        :param input_data: data used for operation training
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Trying to fit primary node with operation: {self.operation}')

        if self.direct_set is True:
            input_data = self.node_data.get('fit')
        else:
            self.node_data.update({'fit': input_data})
        return super().fit(input_data, verbose)

    def predict(self, input_data: InputData,
                output_mode: str = 'default', verbose=False) -> OutputData:
        """
        Predict using the operation located in the primary node

        :param input_data: data used for prediction
        :param output_mode: desired output for operations (e.g. labels, probs, full_probs)
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Predict in primary node by operation: {self.operation}')

        if self.direct_set is True:
            input_data = self.node_data.get('predict')
        else:
            self.node_data.update({'predict': input_data})
        return super().predict(input_data, output_mode, verbose)

    def get_data_from_node(self):
        """ Method returns data if the data was set to the nodes directly """
        return self.node_data


class SecondaryNode(Node):
    """
    The class defines the interface of Secondary nodes modifying tha data flow in Chain

    :param operation_type: str type of the operation defined in operation repository
    :param nodes_from: parent nodes where data comes from
    :param kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, operation_type: str, nodes_from: Optional[List['Node']] = None,
                 **kwargs):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, operation_type=operation_type,
                         **kwargs)

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        """
        Fit the operation located in the secondary node

        :param input_data: data used for operation training
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Trying to fit secondary node with operation: {self.operation}')

        secondary_input = self._input_from_parents(input_data=input_data,
                                                   parent_operation='fit',
                                                   verbose=verbose)
        return super().fit(input_data=secondary_input)

    def predict(self, input_data: InputData, output_mode: str = 'default', verbose=False) -> OutputData:
        """
        Predict using the operation located in the secondary node

        :param input_data: data used for prediction
        :param output_mode: desired output for operations (e.g. labels, probs, full_probs)
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Obtain prediction in secondary node with operation: {self.operation}')

        secondary_input = self._input_from_parents(input_data=input_data,
                                                   parent_operation='predict',
                                                   verbose=verbose)

        return super().predict(input_data=secondary_input, output_mode=output_mode, verbose=verbose)

    def fine_tune(self, input_data: InputData, recursive: bool = True,
                  max_lead_time: timedelta = timedelta(minutes=5), iterations: int = 30,
                  verbose: bool = False):
        # TODO remove
        """
        Run the process of hyperparameter optimization for the node

        :param recursive: flag to initiate the tuning in the parent nodes or not, default: True
        :param input_data: data used for tuning
        :param max_lead_time: max time available for tuning process
        :param iterations: max number of iterations
        :param verbose: flag used for status printing to console, default True
        """
        if verbose:
            self.log.info(f'Tune all parent nodes in secondary node with operation: {self.operation}')

        if recursive:
            secondary_input = self._input_from_parents(input_data=input_data,
                                                       parent_operation='fine_tune',
                                                       max_tune_time=max_lead_time, verbose=verbose)
        else:
            secondary_input = self._input_from_parents(input_data=input_data,
                                                       parent_operation='fit',
                                                       max_tune_time=max_lead_time, verbose=verbose)

        return super().fine_tune(input_data=secondary_input)

    def _nodes_from_with_fixed_order(self):
        if self.nodes_from is not None:
            return sorted(self.nodes_from, key=lambda node: node.descriptive_id)

    def _input_from_parents(self, input_data: InputData,
                            parent_operation: str,
                            max_tune_time: Optional[timedelta] = None,
                            verbose=False) -> InputData:
        if len(self.nodes_from) == 0:
            raise ValueError()

        if verbose:
            self.log.info(f'Fit all parent nodes in secondary node with operation: {self.operation}')

        parent_nodes = self._nodes_from_with_fixed_order()

        parent_results, target = _combine_parents(parent_nodes, input_data,
                                                  parent_operation, max_tune_time)

        secondary_input = InputData.from_predictions(outputs=parent_results,
                                                     target=target)

        return secondary_input


def _combine_parents(parent_nodes: List[Node],
                     input_data: InputData,
                     parent_operation: str,
                     max_tune_time: Optional[timedelta]):
    """
    Method for combining predictions from parent node or nodes

    :param parent_nodes: list of parent nodes, from which predictions will
    be combined
    :param input_data: input data from chain abstraction (source input data)
    :param parent_operation: name of parent operation (fit, predict or fine_tune)
    :param max_tune_time: max time for tuning hyperparameters in nodes
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
        elif parent_operation == 'fine_tune':
            parent.fine_tune(input_data=input_data, max_lead_time=max_tune_time)
            prediction = parent.predict(input_data=input_data)
            parent_results.append(prediction)
        else:
            raise NotImplementedError()

        if input_data is None:
            # InputData was set to primary nodes
            target = prediction.target

    return parent_results, target
