from copy import deepcopy
from datetime import timedelta
from typing import Callable, List, Optional, Tuple, Union, Sequence

import func_timeout

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.dag.graph_delegate import GraphDelegate
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.model import Model
from fedot.core.optimisers.timer import Timer
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.utilities.serializable import Serializable
from fedot.preprocessing.preprocessing import DataPreprocessor, update_indices_for_time_series

ERROR_PREFIX = 'Invalid pipeline configuration:'


class Pipeline(GraphDelegate, Serializable):
    """
    Base class used for composite model structure definition

    :param nodes: Node object(s)
    """

    def __init__(self, nodes: Union[Node, Sequence[Node]] = ()):
        super().__init__(nodes, _graph_nodes_to_pipeline_nodes)

        self.computation_time = None
        self.log = default_log(self)

        # Define data preprocessor
        self.preprocessor = DataPreprocessor()

    def fit_from_scratch(self, input_data: Union[InputData, MultiModalData] = None):
        """
        [Obsolete] Method used for training the pipeline without using saved information

        :param input_data: data used for operation training
        """
        # Clean all saved states and fit all operations
        self.unfit()
        self.fit(input_data)

    def _fit_with_time_limit(self, input_data: Optional[InputData] = None,
                             time: int = 3) -> OutputData:
        """
        Runs training process in all of the pipeline nodes starting with root with time limit.
            Create TODO: unresolved sentence

        :param input_data: data used for operations training
        :param time: time constraint for operations fitting process (in seconds)

        :return: values predicted on the provided ``input_data``
        """
        time = int(timedelta(minutes=time).total_seconds())
        process_state_dict = {}
        fitted_operations = []
        try:
            func_timeout.func_timeout(
                time, self._fit,
                args=(input_data, process_state_dict, fitted_operations)
            )
        except func_timeout.FunctionTimedOut:
            raise TimeoutError(f'Pipeline fitness evaluation time limit is expired')

        self.computation_time = process_state_dict['computation_time_in_seconds']
        for node_num, _ in enumerate(self.nodes):
            self.nodes[node_num].fitted_operation = fitted_operations[node_num]
        return process_state_dict['train_predicted']

    def _fit(self, input_data: Optional[InputData] = None,
             process_state_dict: dict = None, fitted_operations: list = None) -> Optional[OutputData]:
        """
        Runs training process in all the pipeline nodes starting with root

        :param input_data: data used for operation training
        :param process_state_dict: dictionary used for saving required pipeline parameters
            (which were changed inside the process) in case of operations fit time control (when process created)
        :param fitted_operations: list used for saving fitted operations of pipeline nodes

        :return: values predicted on the provided ``input_data`` or nothing in case of the time controlled call
        """

        with Timer() as t:
            computation_time_update = (
                    not self.root_node.fitted_operation or self.computation_time is None
            )
            train_predicted = self.root_node.fit(input_data=input_data)
            if computation_time_update:
                self.computation_time = round(t.minutes_from_start, 3)

        if process_state_dict is None:
            return train_predicted
        else:
            process_state_dict['train_predicted'] = train_predicted
            process_state_dict['computation_time_in_seconds'] = self.computation_time
            for node in self.nodes:
                fitted_operations.append(node.fitted_operation)

    def fit(self, input_data: Union[InputData, MultiModalData],
            time_constraint: Optional[timedelta] = None, n_jobs: int = 1) -> OutputData:
        """
        Runs training process in all of the pipeline nodes starting with root

        :param input_data: data used for operations training
        :param time_constraint: time constraint for operations fitting (in seconds)
        :param n_jobs: number of threads for nodes fitting

        :return: values predicted on the provided ``input_data``
        """
        self.replace_n_jobs_in_nodes(n_jobs)

        copied_input_data = deepcopy(input_data)
        copied_input_data = self.preprocessor.obligatory_prepare_for_fit(copied_input_data)
        # Make additional preprocessing if it is needed
        copied_input_data = self.preprocessor.optional_prepare_for_fit(pipeline=self,
                                                                       data=copied_input_data)
        copied_input_data = self.preprocessor.convert_indexes_for_fit(pipeline=self,
                                                                      data=copied_input_data)
        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        if time_constraint is None:
            train_predicted = self._fit(input_data=copied_input_data)
        else:
            train_predicted = self._fit_with_time_limit(input_data=copied_input_data, time=time_constraint)
        return train_predicted

    @property
    def is_fitted(self) -> bool:
        """
        Property showing whether pipeline is fitted

        :return: flag showing if all of the pipeline nodes are fitted already
        """
        return all(node.fitted_operation is not None for node in self.nodes)

    def unfit(self, mode='all', unfit_preprocessor: bool = True):
        """
        Removes fitted operations for chosen type of nodes.

        :param mode: - 'all' (default) - All models will be unfitted
                     - 'data_operations' - All data operations will be unfitted
        :param unfit_preprocessor: should we unfit preprocessor
        """
        for node in self.nodes:
            if mode == 'all' or (mode == 'data_operations' and isinstance(node.content['name'], DataOperation)):
                node.unfit()

        if unfit_preprocessor:
            self.unfit_preprocessor()

    def unfit_preprocessor(self):
        self.preprocessor = DataPreprocessor()

    def try_load_from_cache(self, cache: Optional[OperationsCache], preprocessing_cache: Optional[PreprocessingCache],
                            fold_id: Optional[int] = None):
        """
        Tries to load pipeline nodes if ``cache`` is provided

        :param cache: pipeline nodes cacher
        :param preprocessing_cache: preprocessing cacher
        :param fold_id: optional part of the cache item UID
                            (can be used to specify the number of CV fold)
        """
        if cache is not None:
            cache.try_load_into_pipeline(self, fold_id)
        if preprocessing_cache is not None:
            preprocessing_cache.try_load_preprocessor(self, fold_id)

    def predict(self, input_data: Union[InputData, MultiModalData], output_mode: str = 'default') -> OutputData:
        """
        Runs the predict process in all of the pipeline nodes starting with root

        :param input_data: data for prediction
        :param output_mode: desired form of output for operations. Available options are:
                - 'default' (as is, default)
                - 'labels' (numbers of classes - for classification)
                - 'probs' (probabilities - for classification == 'default')
                - 'full_probs' (return all probabilities - for binary classification)

        :return: values predicted on the provided ``input_data``
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
                            input_data: Union[InputData, MultiModalData] = None,
                            iterations: int = 50, timeout: Optional[float] = 5.,
                            cv_folds: Optional[int] = None, validation_blocks: int = 3,
                            n_jobs: int = -1) -> 'Pipeline':
        """
        Tunes all nodes hyperparameters simultaneously via black-box
            optimization using PipelineTuner. For details, see
                :meth:`~fedot.core.pipelines.tuning.unified.PipelineTuner.tune_pipeline`

        :param loss_function: desired loss function
        :param loss_params: ``loss_function`` parameters
        :param input_data: data from which to tune the pipeline upon
        :param iterations: max number of tuning iterations
        :param timeout: max time spent on tuning
        :param cv_folds: number of cross-validation folds
        :param validation_blocks: number of validation blocks for time series forecasting
        :param n_jobs: number of parallel threads for tuner

        :return: pipeline with tuned hyperparameters
        """
        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = deepcopy(input_data)

        if timeout is not None:
            timeout = timedelta(minutes=timeout)
        pipeline_tuner = PipelineTuner(pipeline=self,
                                       task=copied_input_data.task,
                                       iterations=iterations,
                                       timeout=timeout,
                                       n_jobs=n_jobs)
        self.log.info('Start pipeline tuning')

        tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=copied_input_data,
                                                      loss_function=loss_function,
                                                      cv_folds=cv_folds,
                                                      validation_blocks=validation_blocks)
        self.log.info('Tuning was finished')

        return tuned_pipeline

    def save(self, path: Optional[str] = None, datetime_in_path: bool = True) -> Tuple[str, dict]:
        """
        Saves the pipeline to JSON representation with pickled fitted operations

        :param path: custom path to save the JSON to
        :param datetime_in_path: is it required to add the datetime timestamp to the path

        :return: <JSON representation of the pipeline structure>, <dict of paths to fitted models>
        """
        template = PipelineTemplate(self)
        json_object, dict_fitted_operations = template.export_pipeline(path, root_node=self.root_node,
                                                                       datetime_in_path=datetime_in_path)
        return json_object, dict_fitted_operations

    def load(self, source: Union[str, dict], dict_fitted_operations: Optional[dict] = None):
        """
        Loads the pipeline JSON representation with pickled fitted operations.

        :param source: where to load the pipeline from
        :param dict_fitted_operations: dictionary of the fitted operations
        """
        self.nodes = []
        template = PipelineTemplate(self)
        template.import_pipeline(source, dict_fitted_operations)

    def __eq__(self, other: 'Pipeline') -> bool:
        """
        Compares this pipeline with provided ``other``

        :param other: another pipeline

        :return: is it equal to ``other`` in terms of the pipelines
        """
        return self.root_node.descriptive_id == other.root_node.descriptive_id

    def __str__(self) -> str:
        """
        Returns compact information about class instance

        :return: formatted string representation of the class instance
        """
        description = {
            'depth': self.depth,
            'length': self.length,
            'nodes': self.nodes,
        }
        return f'{description}'

    @property
    def root_node(self) -> Optional[Node]:
        """
        Finds pipelines sink-node

        :return: the final predictor-node
        """
        if not self.nodes:
            return None
        root = [node for node in self.nodes
                if not any(self.node_children(node))]
        if len(root) > 1:
            raise ValueError(f'{ERROR_PREFIX} More than 1 root_nodes in pipeline')
        return root[0]

    def pipeline_for_side_task(self, task_type: TaskTypesEnum) -> 'Pipeline':
        """
        Returns pipeline formed from the last node solving the given problem and all its parents

        :param task_type: task type of the last node to search for

        :return: pipeline formed from the last node solving the given problem and all of its parents
        """

        max_distance = 0
        side_root_node = None
        for node in self.nodes:
            if (task_type in node.operation.acceptable_task_types
                    and isinstance(node.operation, Model)
                    and node.distance_to_primary_level >= max_distance):
                side_root_node = node
                max_distance = node.distance_to_primary_level

        pipeline = Pipeline(side_root_node)
        pipeline.preprocessor = self.preprocessor
        return pipeline

    def _assign_data_to_nodes(self, input_data: Union[InputData, MultiModalData]) -> Optional[InputData]:
        """
        In case of provided ``input_data`` is of type :class:`~fedot.core.data.multi_modal.MultiModalData`
            assigns :attr:`~fedot.core.pipelines.node.PrimaryNode.node_data` from the ``input_data``
        Does nothing otherwise

        :param input_data: data to assign to :attr:`~fedot.core.pipelines.node.PrimaryNode.node_data`

        :return: None in case of :class:`~fedot.core.data.multi_modal.MultiModalData` and ``input_data`` otherwise
        """
        if isinstance(input_data, MultiModalData):
            for node in (n for n in self.nodes if isinstance(n, PrimaryNode)):
                if node.operation.operation_type in input_data:
                    node.node_data = input_data[node.operation.operation_type]
                    node.direct_set = True
                else:
                    raise ValueError(f'No data for primary node {node}')
            return None
        return input_data

    def print_structure(self):
        """ Prints structural information about the pipeline """
        print(
            'Pipeline structure:',
            self,
            *(f'{node.operation.operation_type} - {node.custom_params}' for node in self.nodes),
            sep='\n'
        )

    def replace_n_jobs_in_nodes(self, n_jobs: int):
        """
        Changes number of jobs for nodes

        :param n_jobs: required number of the jobs to assign to the nodes
        """
        for node in self.nodes:
            for param in ['n_jobs', 'num_threads']:
                if param in node.content['params']:
                    node.content['params'][param] = n_jobs
                    # workaround for lgbm paramaters
                    if node.content['name'] == 'lgbm':
                        node.content['params']['num_threads'] = n_jobs
                        node.content['params']['n_jobs'] = n_jobs


def nodes_with_operation(pipeline: Pipeline, operation_name: str) -> List[Node]:
    """
    Returns list of nodes with the required ``operation_name``

    :param pipeline: pipeline to process
    :param operation_name: name of the operation to filter by

    :return: list of relevant nodes (empty if there are no such nodes)
    """

    # Check if model has decompose operations
    appropriate_nodes = filter(lambda x: x.operation.operation_type == operation_name, pipeline.nodes)

    return list(appropriate_nodes)


def _graph_nodes_to_pipeline_nodes(operator: GraphOperator, nodes: Sequence[Node]):
    """
    Method to update nodes type after performing some action on the pipeline
        via GraphOperator, if any of them are of GraphNode type

    :param nodes: Node object(s)
    """

    for node in nodes:
        if not isinstance(node, GraphNode):
            continue
        if node.nodes_from and not isinstance(node, SecondaryNode):
            operator.update_node(old_node=node,
                                 new_node=SecondaryNode(nodes_from=node.nodes_from,
                                                        content=node.content))
        # TODO: avoid internal access use operator.delete_node
        elif not node.nodes_from and not operator.node_children(node) and node != operator.root_node:
            operator.nodes.remove(node)
        elif not node.nodes_from and not isinstance(node, PrimaryNode):
            operator.update_node(old_node=node,
                                 new_node=PrimaryNode(nodes_from=node.nodes_from,
                                                      content=node.content))
