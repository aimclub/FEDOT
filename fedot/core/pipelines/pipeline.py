from copy import deepcopy
from datetime import timedelta
from multiprocessing import Manager, Process
from typing import Callable, List, Optional, Tuple, Union

from fedot.core.composer.cache import OperationsCache
from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.operations.model import Model
from fedot.core.optimisers.timer import Timer
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.preprocessing.preprocessing import DataPreprocessor, update_indices_for_time_series

ERROR_PREFIX = 'Invalid pipeline configuration:'


class Pipeline(Graph):
    """
    Base class used for composite model structure definition

    :param nodes: Node object(s)
    :param log: Log object to record messages
    """

    def __init__(self, nodes: Optional[Union[Node, List[Node]]] = None,
                 log: Log = None):
        self.computation_time = None
        self.template = None
        self.log = log
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        # Define data preprocessor
        self.preprocessor = DataPreprocessor(self.log)
        super().__init__(nodes)
        self.operator = GraphOperator(self, self._graph_nodes_to_pipeline_nodes)

    def _graph_nodes_to_pipeline_nodes(self, nodes: List[Node] = None):
        """Method to update nodes types after performing some action on the pipeline
        via GraphOperator, if any of them are GraphNode type"""

        if not nodes:
            nodes = self.nodes

        for node in nodes:
            if node.nodes_from and not isinstance(node, SecondaryNode):
                self.operator.update_node(old_node=node,
                                          new_node=SecondaryNode(nodes_from=node.nodes_from,
                                                                 content=node.content))
            elif not node.nodes_from and not self.operator.node_children(node):
                self.nodes.remove(node)
            elif not node.nodes_from and not isinstance(node, PrimaryNode):
                self.operator.update_node(old_node=node,
                                          new_node=PrimaryNode(nodes_from=node.nodes_from,
                                                               content=node.content))

    def fit_from_scratch(self, input_data: Union[InputData, MultiModalData] = None):
        """
        [Obsolete] Method used for training the pipeline without using saved information

        :param input_data: data used for operation training
        """
        # Clean all saved states and fit all operations
        self.unfit(unfit_preprocessor=True)
        self.fit(input_data, use_fitted=False)

    def _fit_with_time_limit(self, input_data: Optional[InputData] = None, use_fitted_operations=False,
                             time: timedelta = timedelta(minutes=3)) -> Manager:
        """
        Run training process with time limit. Create

        :param input_data: data used for operation training
        :param use_fitted_operations: flag defining whether use saved information about previous executions or not,
        default True
        :param time: time constraint for operation fitting process (seconds)
        """
        time = int(time.total_seconds())
        manager = Manager()
        process_state_dict = manager.dict()
        fitted_operations = manager.list()
        p = Process(target=self._fit,
                    args=(input_data, use_fitted_operations, process_state_dict, fitted_operations),
                    kwargs={})
        p.start()
        p.join(time)
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f'Pipeline fitness evaluation time limit is expired')

        self.computation_time = process_state_dict['computation_time']
        for node_num, node in enumerate(self.nodes):
            self.nodes[node_num].fitted_operation = fitted_operations[node_num]
        return process_state_dict['train_predicted']

    def _fit(self, input_data: InputData, use_fitted_operations=False, process_state_dict: Manager = None,
             fitted_operations: Manager = None):
        """
        Run training process in all nodes in pipeline starting with root.

        :param input_data: data used for operation training
        :param use_fitted_operations: flag defining whether use saved information about previous executions or not,
        default True
        :param process_state_dict: this dictionary is used for saving required pipeline parameters (which were changed
        inside the process) in a case of operation fit time control (when process created)
        :param fitted_operations: this list is used for saving fitted operations of pipeline nodes
        """

        with Timer(log=self.log) as t:
            computation_time_update = not use_fitted_operations or not self.root_node.fitted_operation or \
                                      self.computation_time is None
            train_predicted = self.root_node.fit(input_data=input_data)
            if computation_time_update:
                self.computation_time = round(t.minutes_from_start, 3)

        if process_state_dict is None:
            return train_predicted
        else:
            process_state_dict['train_predicted'] = train_predicted
            process_state_dict['computation_time'] = self.computation_time
            for node in self.nodes:
                fitted_operations.append(node.fitted_operation)

    def fit(self, input_data: Union[InputData, MultiModalData], use_fitted=False,
            time_constraint: Optional[timedelta] = None):
        """
        Run training process in all nodes in pipeline starting with root.

        :param input_data: data used for operation training
        :param use_fitted: flag defining whether use saved information about previous fits or not
        :param time_constraint: time constraint for operation fitting (seconds)
        """
        if not use_fitted:
            self.unfit(unfit_preprocessor=True)

        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = deepcopy(input_data)
        copied_input_data = self.preprocessor.obligatory_prepare_for_fit(copied_input_data)
        # Make additional preprocessing if it is needed
        copied_input_data = self.preprocessor.optional_prepare_for_fit(pipeline=self,
                                                                       data=copied_input_data)

        copied_input_data = self.preprocessor.convert_indexes_for_fit(pipeline=self,
                                                                      data=copied_input_data)

        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        if time_constraint is None:
            train_predicted = self._fit(input_data=copied_input_data,
                                        use_fitted_operations=use_fitted)
        else:
            train_predicted = self._fit_with_time_limit(input_data=copied_input_data,
                                                        use_fitted_operations=use_fitted,
                                                        time=time_constraint)
        return train_predicted

    @property
    def is_fitted(self):
        return all([(node.fitted_operation is not None) for node in self.nodes])

    def unfit(self, unfit_preprocessor: bool = True):
        """
        Remove fitted operations for all nodes.
        """
        for node in self.nodes:
            node.unfit()

        if unfit_preprocessor:
            self.preprocessor = DataPreprocessor(self.log)

    def fit_from_cache(self, cache: OperationsCache, fold_num: int = 0) -> bool:
        is_cache_used = False
        for node in self.nodes:
            cached_state = cache.get(node, fold_num)
            if cached_state:
                node.fitted_operation = cached_state.operation
                is_cache_used = True
            else:
                node.fitted_operation = None
        return is_cache_used

    def predict(self, input_data: Union[InputData, MultiModalData], output_mode: str = 'default'):
        """
        Run the predict process in all nodes in pipeline starting with root.

        :param input_data: data for prediction
        :param output_mode: desired form of output for operations. Available options are:
                'default' (as is),
                'labels' (numbers of classes - for classification) ,
                'probs' (probabilities - for classification == 'default'),
                'full_probs' (return all probabilities - for binary classification).
        :return: OutputData with prediction
        """

        if not self.is_fitted:
            ex = 'Pipeline is not fitted yet'
            self.log.error(ex)
            raise ValueError(ex)

        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = deepcopy(input_data)
        copied_input_data = self.preprocessor.obligatory_prepare_for_predict(copied_input_data)
        # Make additional preprocessing if it is needed
        copied_input_data = self.preprocessor.optional_prepare_for_predict(pipeline=self,
                                                                           data=copied_input_data)
        copied_input_data = self.preprocessor.convert_indexes_for_predict(pipeline=self,
                                                                          data=copied_input_data)
        copied_input_data = update_indices_for_time_series(copied_input_data)

        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        result = self.root_node.predict(input_data=copied_input_data, output_mode=output_mode)

        result = self.preprocessor.restore_index(copied_input_data, result)
        # Prediction should be converted into source labels (if it is needed)
        if output_mode == 'labels':
            result.predict = self.preprocessor.apply_inverse_target_encoding(result.predict)
        return result

    def fine_tune_all_nodes(self, loss_function: Callable,
                            loss_params: dict = None,
                            input_data: Union[InputData, MultiModalData] = None,
                            iterations=50, timeout: Optional[int] = 5,
                            cv_folds: int = None,
                            validation_blocks: int = 3) -> 'Pipeline':
        """ Tune all hyperparameters of nodes simultaneously via black-box
            optimization using PipelineTuner. For details, see
        :meth:`~fedot.core.pipelines.tuning.unified.PipelineTuner.tune_pipeline`
        """
        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = deepcopy(input_data)

        if timeout is not None:
            timeout = timedelta(minutes=timeout)
        pipeline_tuner = PipelineTuner(pipeline=self,
                                       task=copied_input_data.task,
                                       iterations=iterations,
                                       timeout=timeout)
        self.log.info('Start pipeline tuning')

        tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=copied_input_data,
                                                      loss_function=loss_function,
                                                      loss_params=loss_params,
                                                      cv_folds=cv_folds,
                                                      validation_blocks=validation_blocks)
        self.log.info('Tuning was finished')

        return tuned_pipeline

    def save(self, path: str = None, datetime_in_path: bool = True) -> Tuple[str, dict]:
        """
        Save the pipeline to the json representation with pickled fitted operations.

        :param path to json file with operation
        :param datetime_in_path flag for addition of the datetime stamp to saving path
        :return: json containing a composite operation description
        """
        self.template = PipelineTemplate(self, self.log)
        json_object, dict_fitted_operations = self.template.export_pipeline(path, root_node=self.root_node,
                                                                            datetime_in_path=datetime_in_path)
        return json_object, dict_fitted_operations

    def load(self, source: Union[str, dict], dict_fitted_operations: dict = None):
        """
        Load the pipeline the json representation with pickled fitted operations.

        :param source path to json file with operation
        :param dict_fitted_operations dictionary of the fitted operations
        """
        self.nodes = []
        self.template = PipelineTemplate(self, self.log)
        self.template.import_pipeline(source, dict_fitted_operations)

    def __eq__(self, other) -> bool:
        return self.root_node.descriptive_id == other.root_node.descriptive_id

    def __str__(self):
        description = {
            'depth': self.depth,
            'length': self.length,
            'nodes': self.nodes,
        }
        return f'{description}'

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not any(self.operator.node_children(node))]
        if len(root) > 1:
            raise ValueError(f'{ERROR_PREFIX} More than 1 root_nodes in pipeline')
        return root[0]

    def pipeline_for_side_task(self, task_type: TaskTypesEnum) -> 'Pipeline':
        """
        Method returns pipeline formed from the last node solving the given problem and all its parents

        :param task_type: task type last node to search for
        :returns: pipeline formed from the last node solving the given problem and all its parents
        """

        max_distance = 0
        side_root_node = None
        for node in self.nodes:
            if task_type in node.operation.acceptable_task_types \
                    and isinstance(node.operation, Model) \
                    and node.distance_to_primary_level >= max_distance:
                side_root_node = node
                max_distance = node.distance_to_primary_level

        pipeline = Pipeline(side_root_node)
        pipeline.preprocessor = self.preprocessor
        return pipeline

    def _assign_data_to_nodes(self, input_data) -> Optional[InputData]:
        if isinstance(input_data, MultiModalData):
            for node in [n for n in self.nodes if isinstance(n, PrimaryNode)]:
                if node.operation.operation_type in input_data.keys():
                    node.node_data = input_data[node.operation.operation_type]
                    node.direct_set = True
                else:
                    raise ValueError(f'No data for primary node {node}')
            return None
        return input_data

    def print_structure(self):
        """ Method print information about pipeline """
        print('Pipeline structure:')
        print(self.__str__())
        for node in self.nodes:
            print(f"{node.operation.operation_type} - {node.custom_params}")


def nodes_with_operation(pipeline: Pipeline, operation_name: str) -> list:
    """ The function return list with nodes with the needed operation

    :param pipeline: pipeline to process
    :param operation_name: name of operation to search
    :return : list with nodes, None if there are no nodes
    """

    # Check if model has decompose operations
    appropriate_nodes = filter(lambda x: x.operation.operation_type == operation_name, pipeline.nodes)

    return list(appropriate_nodes)
