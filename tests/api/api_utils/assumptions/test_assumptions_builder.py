from functools import partial

import torch

from fedot.api.api_utils.assumptions.assumptions_builder import (
    AssumptionsBuilder,
    UniModalAssumptionsBuilder,
)
from fedot.api.api_utils.assumptions.task_assumptions import TensorClassificationAssumptions
from fedot.core.data.tensor_data import TensorData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _tensor_classification_data() -> TensorData:
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        features=torch.zeros((4, 2)),
        target=torch.tensor([0, 1, 0, 1]),
    )


def _pipeline_operations(pipeline: Pipeline):
    return [node.operation.operation_type for node in pipeline.nodes]


def _pipeline_contains_one(pipeline: Pipeline, operation_name: str) -> bool:
    return any(node.operation.operation_type == operation_name for node in pipeline.nodes)


def _pipeline_contains_any(pipeline: Pipeline, *operation_names: str) -> bool:
    return any(map(partial(_pipeline_contains_one, pipeline), operation_names))


def _suitable_operations(task_type: TaskTypesEnum, data_type: DataTypesEnum, repo='model'):
    return OperationTypesRepository(repo).suitable_operation(
        task_type=task_type, data_type=data_type
    )


def test_assumptions_builder_get_uses_unimodal_builder_for_tensordata():
    builder = AssumptionsBuilder.get(_tensor_classification_data())

    assert isinstance(builder, UniModalAssumptionsBuilder)
    assert isinstance(builder.assumptions_generator, TensorClassificationAssumptions)


def test_tensordata_classification_assumption_uses_torch_linear():
    pipeline = AssumptionsBuilder.get(_tensor_classification_data()).build()[0]

    assert pipeline.root_node.operation.operation_type == 'torch_linear'
    assert _pipeline_operations(pipeline) == ['torch_linear', 'optional_preprocessing']


def test_tensordata_assumption_does_not_add_legacy_preprocessing_nodes():
    pipeline = AssumptionsBuilder.get(_tensor_classification_data()).build()[0]

    assert 'scaling' not in _pipeline_operations(pipeline)
    assert 'optional_preprocessing' in _pipeline_operations(pipeline)


def test_assumptions_builder_unsuitable_available_operations():
    data = _tensor_classification_data()
    available_operations = ['linear', 'lagged', 'xgboostreg']

    default_builder = UniModalAssumptionsBuilder(data)
    checked_builder = UniModalAssumptionsBuilder(data).from_operations(available_operations)

    assert default_builder.build() == checked_builder.build()


def test_assumptions_builder_suitable_available_operations():
    data = _tensor_classification_data()
    available_operations = _suitable_operations(data.task.task_type, data.data_type)
    assert available_operations

    baseline_pipeline = AssumptionsBuilder.get(data).build()[0]
    baseline_operation = baseline_pipeline.root_node.operation.operation_type
    available_operations.remove(baseline_operation)

    checked_pipeline = (
        AssumptionsBuilder.get(data).from_operations(available_operations).build()[0]
    )

    assert baseline_pipeline != checked_pipeline
    assert _pipeline_contains_any(checked_pipeline, *available_operations)
