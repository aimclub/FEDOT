from copy import deepcopy
from datetime import timedelta
from multiprocessing import Manager, Process
from typing import List, Optional, Union

from fedot.core.chains.chain_template import ChainTemplate
from fedot.core.chains.node import (Node, PrimaryNode, SecondaryNode)
from fedot.core.composer.timer import Timer
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.composer.optimisers.utils.population_utils import input_data_characteristics

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
        self.nodes = []
        self.log = log
        self.template = None
        self.computation_time = None
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

    def fit_from_scratch(self, input_data: InputData):
        """
        Method used for training the chain without using cached information

        :param input_data: data used for model training
        """
        # Clean all cache and fit all models
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
                info = 'Trained model cache is not actual because you are using new dataset for training. ' \
                       'Parameter use_cache value changed to False'
                self.log.info(info)
                cache_status = False
        return cache_status

    def _fit_with_time_limit(self, input_data: InputData, use_cache=False,
                             time: timedelta = timedelta(minutes=3)) -> Manager:
        """
            Run training process with time limit. Create

            :param input_data: data used for model training
            :param use_cache: flag defining whether use cache information about previous executions or not, default True
            :param time: time constraint for model fitting process (seconds)
        """
        time = int(time.total_seconds())
        manager = Manager()
        process_state_dict = manager.dict()
        fitted_models = manager.list()
        fitted_preprocessors = manager.list()
        p = Process(target=self._fit,
                    args=(input_data, use_cache, process_state_dict, fitted_models, fitted_preprocessors),
                    kwargs={})
        p.start()
        p.join(time)
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f'Chain fitness evaluation time limit is expired')

        self.fitted_on_data = process_state_dict['fitted_on_data']
        self.computation_time = process_state_dict['computation_time']
        for node_num, node in enumerate(self.nodes):
            self.nodes[node_num].fitted_model = fitted_models[node_num]
            self.nodes[node_num].fitted_preprocessor = fitted_preprocessors[node_num]
        return process_state_dict['train_predicted']

    def _fit(self, input_data: InputData, use_cache=False, process_state_dict: Manager = None,
             fitted_models: Manager = None, fitted_preprocessors: Manager = None):
        """
        Run training process in all nodes in chain starting with root.

        :param input_data: data used for model training
        :param use_cache: flag defining whether use cache information about previous executions or not, default True
        :param process_state_dict: this dictionary is used for saving required chain parameters (which were changed
        inside the process) in a case of model fit time control (when process created)
        :param fitted_models: this list is used for saving fitted models of chain nodes
        :param fitted_preprocessors: this list is used for saving fitted preprocessors
        """
        use_cache = self._cache_status_if_new_data(new_input_data=input_data, cache_status=use_cache)

        if not use_cache:
            self.unfit()

        if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
            if input_data.task.task_params.make_future_prediction:
                input_data.task.task_params.return_all_steps = True
            # the make_future_prediction is useless for the fit stage
            input_data.task.task_params.make_future_prediction = False
            check_data_appropriate_for_task(input_data)

        if not use_cache or not self.fitted_on_data:
            self.update_fitted_on_data(input_data)

        with Timer(log=self.log) as t:
            computation_time_update = not use_cache or not self.root_node.fitted_model or \
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
                fitted_models.append(node.fitted_model)
                fitted_preprocessors.append(node.fitted_preprocessor)

    def fit(self, input_data: InputData, use_cache=True, time_constraint: Optional[timedelta] = None):
        """
        Run training process in all nodes in chain starting with root.

        :param input_data: data used for model training
        :param use_cache: flag defining whether use cache information about previous executions or not, default True
        :param time_constraint: time constraint for model fitting (seconds)
        """
        if not use_cache:
            self.unfit()

        if time_constraint is None:
            train_predicted = self._fit(input_data=input_data, use_cache=use_cache)
        else:
            train_predicted = self._fit_with_time_limit(input_data=input_data, use_cache=use_cache,
                                                        time=time_constraint)
        return train_predicted

    def predict(self, input_data: InputData, output_mode: str = 'default'):
        """
        Run the predict process in all nodes in chain starting with root.

        :param input_data: data for prediction
        :param output_mode: desired form of output for models. Available options are:
                'default' (as is),
                'labels' (numbers of classes - for classification) ,
                'probs' (probabilities - for classification =='default'),
                'full_probs' (return all probabilities - for binary classification).
        :return: array of prediction target values
        """

        if not self.is_fitted():
            ex = 'Trained model cache is not actual or empty'
            self.log.error(ex)
            raise ValueError(ex)

        result = self.root_node.predict(input_data=input_data, output_mode=output_mode)
        return result

    def fine_tune_primary_nodes(self, input_data: InputData, iterations: int = 30,
                                max_lead_time: timedelta = timedelta(minutes=3)):
        """
        Optimize hyperparameters in primary nodes models

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        """
        # Select all primary nodes
        # Perform fine-tuning for each model in node
        self.log.info('Start tuning of primary nodes')

        all_primary_nodes = [node for node in self.nodes if isinstance(node, PrimaryNode)]
        for node in all_primary_nodes:
            node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations)

        self.log.info('End tuning')

    def fine_tune_all_nodes(self, input_data: InputData, iterations: int = 30,
                            max_lead_time: timedelta = timedelta(minutes=5)):
        """
        Optimize hyperparameters in all nodes models

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        """

        self.log.info('Start tuning of chain')

        node = self.root_node
        node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations)

        self.log.info('End tuning')

    def add_node(self, new_node: Node):
        """
        Add new node to the Chain

        :param new_node: new Node object
        """
        if new_node not in self.nodes:
            self.nodes.append(new_node)
            if new_node.nodes_from:
                for new_parent_node in new_node.nodes_from:
                    if new_parent_node not in self.nodes:
                        self.add_node(new_parent_node)

    def _actualise_old_node_childs(self, old_node: Node, new_node: Node):
        old_node_offspring = self.node_childs(old_node)
        for old_node_child in old_node_offspring:
            old_node_child.nodes_from[old_node_child.nodes_from.index(old_node)] = new_node

    def replace_node_with_parents(self, old_node: Node, new_node: Node):
        """Exchange subtrees with old and new nodes as roots of subtrees"""
        new_node = deepcopy(new_node)
        self._actualise_old_node_childs(old_node, new_node)
        self.delete_subtree(old_node)
        self.add_node(new_node)
        self._sort_nodes()

    def update_node(self, old_node: Node, new_node: Node):
        if type(new_node) is not type(old_node):
            raise ValueError(f"Can't update {old_node.__class__.__name__} "
                             f"with {new_node.__class__.__name__}")

        self._actualise_old_node_childs(old_node, new_node)
        new_node.nodes_from = old_node.nodes_from
        self.nodes.remove(old_node)
        self.nodes.append(new_node)
        self._sort_nodes()

    def delete_subtree(self, subtree_root_node: Node):
        """Delete node with all the parents it has"""
        for node_child in self.node_childs(subtree_root_node):
            node_child.nodes_from.remove(subtree_root_node)
        for subtree_node in subtree_root_node.ordered_subnodes_hierarchy():
            self.nodes.remove(subtree_node)

    def delete_node(self, node: Node):
        """ This method redirects edges of parents to
        all the childs old node had.
        PNode    PNode              PNode    PNode
            \  /                      |  \   / |
            SNode <- delete this      |   \/   |
            / \                       |   /\   |
        SNode   SNode               SNode   SNode
        """

        def make_secondary_node_as_primary(node_child):
            extracted_type = node_child.model.model_type
            new_primary_node = PrimaryNode(extracted_type)
            this_node_children = self.node_childs(node_child)
            for node in this_node_children:
                index = node.nodes_from.index(node_child)
                node.nodes_from.remove(node_child)
                node.nodes_from.insert(index, new_primary_node)

        node_children_cached = self.node_childs(node)
        self_root_node_cached = self.root_node

        for node_child in self.node_childs(node):
            node_child.nodes_from.remove(node)

        if isinstance(node, SecondaryNode) and len(node.nodes_from) > 1 \
                and len(node_children_cached) > 1:

            for child in node_children_cached:
                for node_from in node.nodes_from:
                    child.nodes_from.append(node_from)

        else:
            if isinstance(node, SecondaryNode):
                for node_from in node.nodes_from:
                    node_children_cached[0].nodes_from.append(node_from)
            elif isinstance(node, PrimaryNode):
                for node_child in node_children_cached:
                    if not node_child.nodes_from:
                        make_secondary_node_as_primary(node_child)
        self.nodes.clear()
        self.add_node(self_root_node_cached)

    def is_fitted(self):
        return all([(node.fitted_model is not None and
                     node.fitted_preprocessor is not None) for node in self.nodes])

    def unfit(self):
        for node in self.nodes:
            node.fitted_model = None
            node.fitted_preprocessor = None

    def node_childs(self, node) -> List[Optional[Node]]:
        return [other_node for other_node in self.nodes if isinstance(other_node, SecondaryNode) and
                node in other_node.nodes_from]

    def _is_node_has_child(self, node) -> bool:
        return any(self.node_childs(node))

    def fit_from_cache(self, cache):
        for node in self.nodes:
            cached_state = cache.get(node)
            if cached_state:
                node.fitted_model = cached_state.model
                node.fitted_preprocessor = cached_state.preprocessor
            else:
                node.fitted_model = None
                node.fitted_preprocessor = None

    def _sort_nodes(self):
        """layer by layer sorting"""
        nodes = self.root_node.ordered_subnodes_hierarchy()
        self.nodes = nodes

    def save(self, path: str):
        """
        :param path to json file with model
        :return: json containing a composite model description
        """
        if not self.template:
            self.template = ChainTemplate(self, self.log)
        json_object = self.template.export_chain(path)
        return json_object

    def load(self, path: str):
        """
        :param path to json file with model
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
                if not self._is_node_has_child(node)]
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


def check_data_appropriate_for_task(data: InputData):
    if (data.task.task_type == TaskTypesEnum.ts_forecasting and
            data.target is not None and
            data.task.task_params.max_window_size > data.target.shape[0]):
        raise ValueError(f'Window size {data.task.task_params.max_window_size} is '
                         f'more then data length {data.target.shape[0]}')
