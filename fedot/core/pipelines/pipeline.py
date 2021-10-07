from copy import copy
from datetime import timedelta
from multiprocessing import Manager, Process
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.dag.graph import Graph
from fedot.core.data.data import InputData, data_has_categorical_features, data_has_missing_values
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    DataOperationImplementation, ImputationImplementation, OneHotEncodingImplementation
from fedot.core.optimisers.timer import Timer
from fedot.core.optimisers.utils.population_utils import input_data_characteristics
from fedot.core.pipelines.node import Node, PrimaryNode
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.data.data import data_type_is_table

# The allowed empirical partition limit of the number of rows to delete.
# Rows that have 'string' type, instead of other 'integer' observes.
# Example: 90% objects in column are 'integer', other are 'string'. Then
# we will try to convert 'string' data to 'integer', otherwise delete it.
EMPIRICAL_PARTITION = 0.1
ERROR_PREFIX = 'Invalid pipeline configuration:'


class Pipeline(Graph):
    """
    Base class used for composite model structure definition

    :param nodes: Node object(s)
    :param log: Log object to record messages
    :param tag: uniq part of the repository filename

    .. note::
        fitted_on_data stores the data which were used in last pipeline fitting (equals None if pipeline hasn't been
        fitted yet)
    """

    def __init__(self, nodes: Optional[Union[Node, List[Node]]] = None,
                 log: Log = None):

        self.computation_time = None
        self.template = None
        self.fitted_on_data = {}
        self.pre_proc_encoders = {}

        self.log = log
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log
        super().__init__(nodes)

    def fit_from_scratch(self, input_data: Union[InputData, MultiModalData] = None):
        """
        Method used for training the pipeline without using saved information

        :param input_data: data used for operation training
        """
        # Clean all saved states and fit all operations
        self.log.debug('Fit pipeline from scratch')
        self.unfit()
        self.fit(input_data, use_fitted=False)

    def update_fitted_on_data(self, data: InputData):
        characteristics = input_data_characteristics(data=data, log=self.log)
        self.fitted_on_data['data_type'] = characteristics[0]
        self.fitted_on_data['features_hash'] = characteristics[1]
        self.fitted_on_data['target_hash'] = characteristics[2]

    def _fitted_status_if_new_data(self, new_input_data: InputData, fitted_status: bool):
        new_data_params = input_data_characteristics(new_input_data, log=self.log)
        if fitted_status and self.fitted_on_data:
            params_names = ('data_type', 'features_hash', 'target_hash')
            are_data_params_different = any(
                [new_data_param != self.fitted_on_data[param_name] for new_data_param, param_name in
                 zip(new_data_params, params_names)])
            if are_data_params_different:
                info = 'Trained operation is not actual because you are using new dataset for training. ' \
                       'Parameter use_fitted value changed to False'
                self.log.info(info)
                fitted_status = False
        return fitted_status

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

        self.fitted_on_data = process_state_dict['fitted_on_data']
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

        # InputData was set directly to the primary nodes
        if input_data is None:
            use_fitted_operations = False
        else:
            use_fitted_operations = self._fitted_status_if_new_data(new_input_data=input_data,
                                                                    fitted_status=use_fitted_operations)

            if not use_fitted_operations or not self.fitted_on_data:
                # Don't use previous information
                self.unfit()
                self.update_fitted_on_data(input_data)

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
            process_state_dict['fitted_on_data'] = self.fitted_on_data
            for node in self.nodes:
                fitted_operations.append(node.fitted_operation)

    def fit(self, input_data: Union[InputData, MultiModalData], use_fitted=True,
            time_constraint: Optional[timedelta] = None):
        """
        Run training process in all nodes in pipeline starting with root.

        :param input_data: data used for operation training
        :param use_fitted: flag defining whether use saved information about previous executions or not,
            default True
        :param time_constraint: time constraint for operation fitting (seconds)
        """
        if not use_fitted:
            self.unfit()

        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = copy(input_data)
        copied_input_data = self._preprocessing_fit_data(copied_input_data)
        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        if time_constraint is None:
            train_predicted = self._fit(input_data=copied_input_data,
                                        use_fitted_operations=use_fitted)
        else:
            train_predicted = self._fit_with_time_limit(input_data=copied_input_data,
                                                        use_fitted_operations=use_fitted,
                                                        time=time_constraint)
        return train_predicted

    def _preprocessing_fit_data(self, data: Union[InputData, MultiModalData]):
        has_imputation_operation, has_encoder_operation = pipeline_encoders_validation(self)

        data = _custom_preprocessing(data)

        if data_has_missing_values(data) and not has_imputation_operation:
            data = _imputation_implementation(data)

        if data_has_categorical_features(data) and not has_encoder_operation:
            self.pre_proc_encoders = _encode_data_for_fit(data)
        return data

    @property
    def is_fitted(self):
        return all([(node.fitted_operation is not None) for node in self.nodes])

    def unfit(self):
        """
        Remove fitted operations for all nodes.
        """
        for node in self.nodes:
            node.unfit()

    def fit_from_cache(self, cache: OperationsCache):
        for node in self.nodes:
            cached_state = cache.get(node)
            if cached_state:
                node.fitted_operation = cached_state.operation
            else:
                node.fitted_operation = None

    def predict(self, input_data: Union[InputData, MultiModalData], output_mode: str = 'default'):
        """
        Run the predict process in all nodes in pipeline starting with root.

        :param input_data: data for prediction
        :param output_mode: desired form of output for operations. Available options are:
                'default' (as is),
                'labels' (numbers of classes - for classification) ,
                'probs' (probabilities - for classification =='default'),
                'full_probs' (return all probabilities - for binary classification).
        :return: OutputData with prediction
        """

        if not self.is_fitted:
            ex = 'Pipeline is not fitted yet'
            self.log.error(ex)
            raise ValueError(ex)

        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = copy(input_data)
        has_imputation_operation, has_encoder_operation = pipeline_encoders_validation(self)

        copied_input_data = _custom_preprocessing(copied_input_data)

        if data_has_missing_values(copied_input_data) and not has_imputation_operation:
            copied_input_data = _imputation_implementation(copied_input_data)

        if data_has_categorical_features(copied_input_data) and not has_encoder_operation:
            _encode_data_for_prediction(copied_input_data, self.pre_proc_encoders)

        copied_input_data = self._assign_data_to_nodes(copied_input_data)

        result = self.root_node.predict(input_data=copied_input_data, output_mode=output_mode)
        return result

    def fine_tune_all_nodes(self, loss_function: Callable,
                            loss_params: dict = None,
                            input_data: Union[InputData, MultiModalData] = None,
                            iterations=50, timeout: int = 5,
                            cv_folds: int = None,
                            validation_blocks: int = 3) -> 'Pipeline':
        """ Tune all hyperparameters of nodes simultaneously via black-box
            optimization using PipelineTuner. For details, see
        :meth:`~fedot.core.pipelines.tuning.unified.PipelineTuner.tune_pipeline`
        """
        # Make copy of the input data to avoid performing inplace operations
        copied_input_data = copy(input_data)

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

    def save(self, path: str = None) -> Tuple[str, dict]:
        """
        Save the pipeline to the json representation with pickled fitted operations.

        :param path to json file with operation
        :return: json containing a composite operation description
        """
        self.template = PipelineTemplate(self, self.log)
        json_object, dict_fitted_operations = self.template.export_pipeline(path, root_node=self.root_node)
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


def pipeline_encoders_validation(pipeline: Pipeline) -> (bool, bool):
    """ Check whether Imputation and OneHotEncoder operation exist in pipeline.

    :param data Pipeline: data to check
    :return (bool, bool): has Imputation and OneHotEncoder in pipeline
    """

    has_imputers, has_encoders = [], []

    def _check_imputer_encoder_recursion(root: Optional[Node], has_imputer: bool = False, has_encoder: bool = False):
        node_type = root.operation.operation_type
        if node_type == 'simple_imputation':
            has_imputer = True
        if node_type == 'one_hot_encoding':
            has_encoder = True

        if has_imputer and has_encoder:
            return has_imputer, has_encoder
        elif root.nodes_from is None:
            return has_imputer, has_encoder

        for node in root.nodes_from:
            answer = _check_imputer_encoder_recursion(node, has_imputer, has_encoder)
            if answer is not None:
                imputer, encoder = answer
                has_imputers.append(imputer)
                has_encoders.append(encoder)

    _check_imputer_encoder_recursion(pipeline.root_node)

    if not has_imputers and not has_encoders:
        return False, False

    has_imputer = len([_ for _ in has_imputers if not _]) == 0
    has_encoder = len([_ for _ in has_encoders if not _]) == 0
    return has_imputer, has_encoder


def _is_numeric(s) -> bool:
    """ Check if variable converted to float.

    :param s: any type variable
    :return: is variable convertable to float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_np_array_has_nan(array):
    for x in array:
        if x is np.nan:
            return True
    return False


def _try_convert_to_numeric(values):
    try:
        values = pd.to_numeric(values)
        values = values.astype(np.number)
    except ValueError:
        pass
    return values


def _custom_preprocessing(data: Union[InputData, MultiModalData]):
    if isinstance(data, InputData):
        if data_type_is_table(data):
            data = _preprocessing_input_data(data)
    elif isinstance(data, MultiModalData):
        for data_source_name, values in data.items():
            if data_type_is_table(values):
                data[data_source_name] = _preprocessing_input_data(values)

    return data


def _preprocessing_input_data(data: InputData) -> InputData:
    features = data.features
    target = data.target

    # delete rows with equal target None
    if target is not None and len(target.shape) != 0 and is_np_array_has_nan(target):
        target_index_with_nan = np.hstack(np.argwhere(np.isnan(target)))
        data.features = np.delete(features, target_index_with_nan, 0)
        data.target = np.delete(data.target, target_index_with_nan, 0)
        data.idx = np.delete(data.idx, target_index_with_nan, 0)

    source_shape = features.shape
    columns_amount = source_shape[1] if len(source_shape) > 1 else 1

    for i in range(columns_amount):
        values = pd.Series(features[:, i])
        # check for each column, if values converted to numeric. remember index of rows that not converted
        if any(list(map(lambda x: isinstance(x, str), values))):
            not_numeric = list(map(lambda x: not _is_numeric(x), values))
            rows_to_nan = list(values.index[not_numeric])
            partition_not_numeric = len(rows_to_nan) / source_shape[0]

            # if partition of numerical rows less then EMPIRICAL_PARTITION,
            # then convert to numerical and others to Nan
            if partition_not_numeric < EMPIRICAL_PARTITION:
                values[rows_to_nan] = np.nan
                data.features[:, i] = _try_convert_to_numeric(values)
            # if EMPIRICAL_PARTITION < partition < 1, then some data in column are
            # integer and some data are string, can not handle this case
            elif partition_not_numeric < 0.9:
                raise ValueError("The data in the column has a different type. Need to preprocessing data manually.")

    return data


def nodes_with_operation(pipeline: Pipeline, operation_name: str) -> list:
    """ The function return list with nodes with the needed operation

    :param pipeline: pipeline to process
    :param operation_name: name of operation to search
    :return : list with nodes, None if there are no nodes
    """

    # Check if model has decompose operations
    appropriate_nodes = filter(lambda x: x.operation.operation_type == operation_name, pipeline.nodes)

    return list(appropriate_nodes)


def _encode_data_for_fit(data: Union[InputData, MultiModalData]) -> \
        Union[List[DataOperationImplementation], DataOperationImplementation]:
    """ Encode categorical features to numerical. In additional,
    save encoders to use later for prediction data.

    :param data: data to transform
    :return encoders: operation preprocessing categorical features or list of it
    """

    encoders = None
    if isinstance(data, InputData):
        transformed, encoder = _create_encoder(data)
        encoders = encoder
        data.features = transformed
    elif isinstance(data, MultiModalData):
        encoders = {}
        for data_source_name, values in data.items():
            if data_source_name.startswith('data_source_table'):
                transformed, encoder = _create_encoder(values)
                if encoder is not None:
                    encoders[data_source_name] = encoder
                data[data_source_name].features = transformed

    return encoders


def _encode_data_for_prediction(data: Union[InputData, MultiModalData],
                                encoders: Union[dict, DataOperationImplementation]):
    """ Transformation the prediction data inplace. Use the same transformations as for the training data.

    :param data: data to transformation
    :param encoders: encoders f transformation
    """
    if encoders:
        if isinstance(data, InputData):
            transformed = encoders.transform(data, True).predict
            data.features = transformed
        elif isinstance(data, MultiModalData):
            for data_source_name, encoder in encoders.items():
                transformed = encoder.transform(data[data_source_name], True).predict
                data[data_source_name].features = transformed


def _imputation_implementation(data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
    if isinstance(data, InputData):
        return _imputation_implementation_unidata(data)
    if isinstance(data, MultiModalData):
        for data_source_name, values in data.items():
            if data_source_name.startswith('data_source_table') or data_source_name.startswith('data_source_ts'):
                data[data_source_name].features = _imputation_implementation_unidata(values)
        return data
    raise ValueError(f"Data format is not supported.")


def _imputation_implementation_unidata(data: InputData):
    """ Fill in the gaps in the data inplace.

    :param data: data for fill in the gaps
    """
    imputer = ImputationImplementation()
    output_data = imputer.fit_transform(data)
    transformed = InputData(features=output_data.predict, data_type=output_data.data_type,
                            target=output_data.target, task=output_data.task, idx=output_data.idx)
    return transformed


def _create_encoder(data: InputData):
    """ Fills in the gaps, converts categorical features using OneHotEncoder and create encoder.

    :param data: data to preprocess
    :return tuple(array, Union[OneHotEncodingImplementation, None]): tuple of transformed and [encoder or None]
    """

    encoder = None
    if data_has_categorical_features(data):
        encoder = OneHotEncodingImplementation()
        encoder.fit(data)
        transformed = encoder.transform(data, True).predict
    else:
        transformed = data.features

    return transformed, encoder
