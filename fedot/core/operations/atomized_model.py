from collections import Counter
from datetime import timedelta
from functools import reduce
from operator import and_, or_
from typing import Any, Callable, Dict, List, Optional, Set, Union

from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation import Operation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.metrics_repository import MetricCallable
from fedot.core.repository.operation_types_repository import OperationMetaInfo, atomized_model_type


class AtomizedModel(Operation):
    """ Class which replace Operation class for AtomizedModel object """

    def __init__(self, pipeline: 'Pipeline'):
        if not pipeline.root_node:
            raise ValueError('AtomizedModel could not create instance of empty Pipeline.')

        super().__init__(operation_type=atomized_model_type())
        self.pipeline = pipeline

    def fit(self, params: Optional[Union[OperationParameters, dict]], data: InputData) -> ('Pipeline', OutputData):
        predicted_train = self.pipeline.fit(input_data=data)
        fitted_atomized_operation = self.pipeline
        return fitted_atomized_operation, predicted_train

    def predict(self,
                fitted_operation: 'Pipeline',
                data: InputData,
                params: Optional[Union[OperationParameters, Dict[str, Any]]] = None,
                output_mode: str = 'default') -> OutputData:

        # Preprocessing applied
        prediction = fitted_operation.predict(input_data=data, output_mode=output_mode)
        prediction = self.assign_tabular_column_types(prediction, output_mode)
        return prediction

    def predict_for_fit(self,
                        fitted_operation: 'Pipeline',
                        data: InputData,
                        params: Optional[OperationParameters] = None,
                        output_mode: str = 'default') -> OutputData:
        return self.predict(fitted_operation, data, params, output_mode)

    def fine_tune(self,
                  metric_function: MetricCallable,
                  input_data: Optional[InputData] = None,
                  iterations: int = 50,
                  timeout: int = 5) -> 'AtomizedModel':
        """ Method for tuning hyperparameters """
        tuner = TunerBuilder(input_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(metric_function) \
            .with_iterations(iterations) \
            .with_timeout(timedelta(minutes=timeout)) \
            .build(input_data)
        tuned_pipeline = tuner.tune(self.pipeline)
        tuned_atomized_model = AtomizedModel(tuned_pipeline)
        return tuned_atomized_model

    @property
    def metadata(self) -> OperationMetaInfo:
        root_node = self.pipeline.root_node

        def extract_metadata_from_pipeline(attr_name: str,
                                           node_filter: Optional[Callable[[PipelineNode], bool]] = None,
                                           reduce_function: Optional[Callable[[Set], Set]] = None) -> List[Any]:
            """ Extract metadata from atomized pipeline
                :param attr_name: extracting metadata property
                :param node_filter: return True for nodes with extracting metadata
                :param reduce_function: function is used for combining extracted
                                        metadata in ``reduce`` function
                :return: list with extracted metadata
                """
            nodes_to_extract_metadata = self.pipeline.nodes
            if node_filter is not None:
                nodes_to_extract_metadata = [node for node in nodes_to_extract_metadata if node_filter(node)]
            data = [set(getattr(node.operation.metadata, attr_name)) for node in nodes_to_extract_metadata]
            return list(reduce(reduce_function or or_, data))

        tags = extract_metadata_from_pipeline('tags')
        input_types = extract_metadata_from_pipeline('input_types',
                                                     node_filter=lambda node: node.is_primary,
                                                     reduce_function=and_)
        output_types = root_node.operation.metadata.output_types
        presets = extract_metadata_from_pipeline('presets')

        operation_info = OperationMetaInfo(id=root_node.operation.metadata.id,
                                           input_types=input_types,
                                           output_types=output_types,
                                           task_type=root_node.operation.metadata.task_type,
                                           supported_strategies=None,
                                           allowed_positions=['any'],
                                           tags=tags,
                                           presets=presets)
        return operation_info

    def description(self, operation_params: Optional[dict] = None) -> str:
        operation_type = self.operation_type
        operation_length = self.pipeline.length
        operation_depth = self.pipeline.depth
        operation_id = self.pipeline.root_node.descriptive_id
        operation_types = map(lambda node: node.operation.operation_type,
                              self.pipeline.nodes)
        operation_types_dict = dict(Counter(operation_types))
        return f'{operation_type}_length:{operation_length}_depth:{operation_depth}' \
               f'_types:{operation_types_dict}_id:{operation_id}'

    @staticmethod
    def assign_tabular_column_types(output_data: OutputData,
                                    output_mode: str) -> OutputData:
        """ There is no need to perform any column types determination for nested pipelines """
        return output_data
