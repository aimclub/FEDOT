from copy import copy
from copy import deepcopy
from datetime import timedelta
from multiprocessing import Manager, Process
from typing import Callable, List, Optional, Union
from uuid import uuid4

from fedot.core.chains.chain_template import ChainTemplate
from fedot.core.chains.graph_operator import GraphOperator
from fedot.core.chains.node import (Node, PrimaryNode)
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.composer.optimisers.utils.population_utils import input_data_characteristics
from fedot.core.composer.timer import Timer
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log

ERROR_PREFIX = 'Invalid chain configuration:'


class Chain:
    """
    Base class used for composite model structure definition

    :param nodes: Node object(s)
    :param log: Log object to record messages

    .. note::
        fitted_on_data stores the data which were used in last chain fitting (equals None if chain hasn't been
        fitted yet)
    """

    def __init__(self, nodes: Optional[Union[Node, List[Node]]] = None,
                 log: Log = None):
        self.uid = str(uuid4())
        self.nodes = []
        self.log = log
        self.template = None
        self.computation_time = None
        self.operator = GraphOperator(self)

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if nodes:
            if isinstance(nodes, list):
                for node in nodes:
                    self.add_node(node)
            else:
                self.add_node(nodes)
        self.fitted_on_data = {}

    def fit_from_scratch(self, input_data: InputData = None):
        """
        Method used for training the chain without using cached information

        :param input_data: data used for operation training
        """
        # Clean all cache and fit all operations
        self.log.info('Fit chain from scratch')
        self.unfit()
        self.fit(input_data, use_cache=False)

    def update_fitted_on_data(self, data: InputData):
        characteristics = input_data_characteristics(data=data, log=self.log)
        self.fitted_on_data['data_type'] = characteristics[0]
        self.fitted_on_data['features_hash'] = characteristics[1]
        self.fitted_on_data['target_hash'] = characteristics[2]

    def _cache_status_if_new_data(self, new_input_data: InputData, cache_status: bool):
        new_data_params = input_data_characteristics(new_input_data, log=self.log)
        if cache_status and self.fitted_on_data:
            params_names = ('data_type', 'features_hash', 'target_hash')
            are_data_params_different = any(
                [new_data_param != self.fitted_on_data[param_name] for new_data_param, param_name in
                 zip(new_data_params, params_names)])
            if are_data_params_different:
                info = 'Trained operation cache is not actual because you are using new dataset for training. ' \
                       'Parameter use_cache value changed to False'
                self.log.info(info)
                cache_status = False
        return cache_status

    def _fit_with_time_limit(self, input_data: Optional[InputData] = None, use_cache=False,
                             time: timedelta = timedelta(minutes=3)) -> Manager:
        """
        Run training process with time limit. Create

        :param input_data: data used for operation training
        :param use_cache: flag defining whether use cache information about previous executions or not, default True
        :param time: time constraint for operation fitting process (seconds)
        """
        time = int(time.total_seconds())
        manager = Manager()
        process_state_dict = manager.dict()
        fitted_operations = manager.list()
        p = Process(target=self._fit,
                    args=(input_data, use_cache, process_state_dict, fitted_operations),
                    kwargs={})
        p.start()
        p.join(time)
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f'Chain fitness evaluation time limit is expired')

        self.fitted_on_data = process_state_dict['fitted_on_data']
        self.computation_time = process_state_dict['computation_time']
        for node_num, node in enumerate(self.nodes):
            self.nodes[node_num].fitted_operation = fitted_operations[node_num]
        return process_state_dict['train_predicted']

    def _fit(self, input_data: InputData, use_cache=False, process_state_dict: Manager = None,
             fitted_operations: Manager = None):
        """
        Run training process in all nodes in chain starting with root.

        :param input_data: data used for operation training
        :param use_cache: flag defining whether use cache information about previous executions or not, default True
        :param process_state_dict: this dictionary is used for saving required chain parameters (which were changed
        inside the process) in a case of operation fit time control (when process created)
        :param fitted_operations: this list is used for saving fitted operations of chain nodes
        """

        # InputData was set directly to the primary nodes
        if input_data is None:
            use_cache = False
        else:
            use_cache = self._cache_status_if_new_data(new_input_data=input_data, cache_status=use_cache)

            if not use_cache or not self.fitted_on_data:
                # Don't use previous information
                self.unfit()
                self.update_fitted_on_data(input_data)

        with Timer(log=self.log) as t:
            computation_time_update = not use_cache or not self.root_node.fitted_operation or \
                                      self.computation_time is None

            train_predicted = self.root_node.fit(input_data=input_data)
            if computation_time_update:
                self.computation_time = round(t.minutes_from_start, 3)

        if process_state_dict is None:
            return train_predicted
        else:
            process_state_dict['train_predicted'] = train_predicted
            process_state_dict['computation_time'] = self.computation_time
            process_state_dict['fitted_on_data'] = self.fitted_on_data
            for node in self.nodes:
                fitted_operations.append(node.fitted_operation)

    def fit(self, input_data: Union[InputData, MultiModalData], use_cache=True,
            time_constraint: Optional[timedelta] = None):
        """
        Run training process in all nodes in chain starting with root.

        :param input_data: data used for operation training
        :param use_cache: flag defining whether use cache information about previous executions or not, default True
        :param time_constraint: time constraint for operation fitting (seconds)
        """

        if not use_cache:
            self.unfit()

        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = copy(input_data)
        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        if time_constraint is None:
            train_predicted = self._fit(input_data=copied_input_data, use_cache=use_cache)
        else:
            train_predicted = self._fit_with_time_limit(input_data=copied_input_data, use_cache=use_cache,
                                                        time=time_constraint)
        return train_predicted

    def predict(self, input_data: Union[InputData, MultiModalData], output_mode: str = 'default'):
        """
        Run the predict process in all nodes in chain starting with root.

        :param input_data: data for prediction
        :param output_mode: desired form of output for operations. Available options are:
                'default' (as is),
                'labels' (numbers of classes - for classification) ,
                'probs' (probabilities - for classification =='default'),
                'full_probs' (return all probabilities - for binary classification).
        :return: OutputData with prediction
        """

        if not self.is_fitted:
            ex = 'Trained operation cache is not actual or empty'
            self.log.error(ex)
            raise ValueError(ex)

        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = copy(input_data)
        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        result = self.root_node.predict(input_data=copied_input_data, output_mode=output_mode)
        return result

    def fine_tune_all_nodes(self, loss_function: Callable,
                            loss_params: Callable = None,
                            input_data: Union[InputData, MultiModalData] = None,
                            iterations=50, max_lead_time: int = 5) -> 'Chain':
        """ Tune all hyperparameters of nodes simultaneously via black-box
            optimization using ChainTuner. For details, see
        :meth:`~fedot.core.chains.tuning.unified.ChainTuner.tune_chain`
        """
        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = copy(input_data)
        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        max_lead_time = timedelta(minutes=max_lead_time)
        chain_tuner = ChainTuner(chain=self,
                                 task=copied_input_data.task,
                                 iterations=iterations,
                                 max_lead_time=max_lead_time)
        self.log.info('Start tuning of primary nodes')
        tuned_chain = chain_tuner.tune_chain(input_data=copied_input_data,
                                             loss_function=loss_function,
                                             loss_params=loss_params)
        self.log.info('Tuning was finished')

        return tuned_chain

    def add_node(self, new_node: Node):
        """
        Add new node to the Chain

        :param node: new Node object
        """
        self.operator.add_node(new_node)

    def update_node(self, old_node: Node, new_node: Node):
        """
        Replace old_node with new one.

        :param old_node: Node object to replace
        :param new_node: Node object to replace
        """

        self.operator.update_node(old_node, new_node)

    def update_subtree(self, old_subroot: Node, new_subroot: Node):
        """
        Replace the subtrees with old and new nodes as subroots

        :param old_subroot: Node object to replace
        :param new_subroot: Node object to replace
        """
        self.operator.update_subtree(old_subroot, new_subroot)

    def delete_node(self, node: Node):
        """
        Delete chosen node redirecting all its parents to the child.

        :param node: Node object to delete
        """

        self.operator.delete_node(node)

    def delete_subtree(self, subroot: Node):
        """
        Delete the subtree with node as subroot.

        :param subroot:
        """
        self.operator.delete_subtree(subroot)

    @property
    def is_fitted(self):
        return all([(node.fitted_operation is not None) for node in self.nodes])

    def unfit(self):
        """
        Remove fitted operations for all nodes.
        """
        for node in self.nodes:
            node.unfit()

    def fit_from_cache(self, cache):
        for node in self.nodes:
            cached_state = cache.get(node)
            if cached_state:
                node.fitted_operation = cached_state.operation
            else:
                node.fitted_operation = None

    def save(self, path: str):
        """
        Save the chain to the json representation with pickled fitted operations.

        :param path to json file with operation
        :return: json containing a composite operation description
        """
        if not self.template:
            self.template = ChainTemplate(self, self.log)
        json_object = self.template.export_chain(path)
        return json_object

    def load(self, path: str):
        """
        Load the chain the json representation with pickled fitted operations.

        :param path to json file with operation
        """
        self.nodes = []
        self.template = ChainTemplate(self, self.log)
        self.template.import_chain(path)

    def show(self, path: str = None):
        ChainVisualiser().visualise(self, path)

    def __eq__(self, other) -> bool:
        return self.root_node.descriptive_id == other.root_node.descriptive_id

    def __str__(self):
        description = {
            'depth': self.depth,
            'length': self.length,
            'nodes': self.nodes,
        }
        return f'{description}'

    def __repr__(self):
        return self.__str__()

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not any(self.operator.node_children(node))]
        if len(root) > 1:
            raise ValueError(f'{ERROR_PREFIX} More than 1 root_nodes in chain')
        return root[0]

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        def _depth_recursive(node):
            if node is None:
                return 0
            if isinstance(node, PrimaryNode):
                return 1
            else:
                return 1 + max([_depth_recursive(next_node) for next_node in node.nodes_from])

        return _depth_recursive(self.root_node)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.uid = uuid4()
        return result

    def __deepcopy__(self, memo=None):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        result.uid = uuid4()
        return result

    def _assign_data_to_nodes(self, input_data) -> Optional[InputData]:
        if isinstance(input_data, MultiModalData):
            for node in self.nodes:
                if node.operation.operation_type in input_data.keys():
                    node.node_data = input_data[node.operation.operation_type]
                    node.direct_set = True
            return None
        return input_data


def nodes_with_operation(chain: Chain, operation_name: str) -> list:
    """ The function return list with nodes with the needed operation

    :param chain: chain to process
    :param operation_name: name of operation to search
    :return : list with nodes, None if there are no nodes
    """

    # Check if model has decompose operations
    appropriate_nodes = filter(lambda x: x.operation.operation_type == operation_name, chain.nodes)

    return list(appropriate_nodes)
