from copy import deepcopy
from datetime import timedelta
from typing import Callable, Union, Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation import Operation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.operation_types_repository import OperationMetaInfo, \
    atomized_model_type
from fedot.core.utils import make_pipeline_generator
from fedot.preprocessing.preprocessing import DataPreprocessor


class AtomizedModel(Operation):
    """ Class which replace Operation class for AtomizedModel object """

    def __init__(self, pipeline: 'Pipeline'):
        if not pipeline.root_node:
            raise ValueError('AtomizedModel could not create instance of empty Pipeline.')

        super().__init__(operation_type=atomized_model_type())
        self.pipeline = pipeline
        self.unique_id = self.pipeline.root_node.descriptive_id
        self.atomized_preprocessor = DataPreprocessor()

    def fit(self, params: Optional[Union[OperationParameters, dict]], data: InputData,
            use_cache: bool = True):

        copied_input_data = deepcopy(data)
        copied_input_data = self.atomized_preprocessor.obligatory_prepare_for_fit(copied_input_data)

        predicted_train = self.pipeline.fit(input_data=copied_input_data)
        fitted_atomized_operation = self.pipeline

        return fitted_atomized_operation, predicted_train

    def predict(self, fitted_operation, data: InputData,
                params: Optional[Union[OperationParameters, dict]] = None, output_mode: str = 'default'):

        # Preprocessing applied
        copied_input_data = deepcopy(data)
        copied_input_data = self.atomized_preprocessor.obligatory_prepare_for_predict(copied_input_data)

        prediction = fitted_operation.predict(input_data=copied_input_data, output_mode=output_mode)
        prediction = self.assign_tabular_column_types(prediction, output_mode)
        return prediction

    def predict_for_fit(self, fitted_operation, data: InputData, params: Optional[OperationParameters] = None,
                        output_mode: str = 'default'):
        return self.predict(fitted_operation, data, params, output_mode)

    def fine_tune(self, metric_function: Callable,
                  input_data: InputData = None, iterations: int = 50,
                  timeout: int = 5):
        """ Method for tuning hyperparameters """
        tuner = TunerBuilder(input_data.task)\
            .with_tuner(PipelineTuner)\
            .with_metric(metric_function)\
            .with_iterations(iterations)\
            .with_timeout(timedelta(minutes=timeout))\
            .build(input_data)
        tuned_pipeline = tuner.tune(self.pipeline)
        tuned_atomized_model = AtomizedModel(tuned_pipeline)
        return tuned_atomized_model

    @property
    def metadata(self) -> OperationMetaInfo:
        generator = make_pipeline_generator(self.pipeline)
        tags = set()

        for node in generator:
            tags.update(node.operation_tags)

        root_node = self.pipeline.root_node
        supported_strategies = None
        allowed_positions = ['any']
        tags = list(tags)

        operation_info = OperationMetaInfo(root_node.operation.supplementary_data.id,
                                           root_node.operation.supplementary_data.input_types,
                                           root_node.operation.supplementary_data.output_types,
                                           root_node.operation.supplementary_data.task_type,
                                           supported_strategies, allowed_positions,
                                           tags)
        return operation_info

    def description(self, operation_params: Optional[dict]):
        operation_type = self.operation_type
        operation_length = self.pipeline.length
        operation_depth = self.pipeline.depth
        operation_id = self.unique_id
        operation_types = {}

        for node in self.pipeline.nodes:
            if node.operation.operation_type in operation_types:
                operation_types[node.operation.operation_type] += 1
            else:
                operation_types[node.operation.operation_type] = 1

        return f'{operation_type}_length:{operation_length}_depth:{operation_depth}' \
               f'_types:{operation_types}_id:{operation_id}'

    @staticmethod
    def assign_tabular_column_types(output_data: OutputData,
                                    output_mode: str) -> OutputData:
        """ There is no need to perform any column types determination for nested pipelines """
        return output_data
