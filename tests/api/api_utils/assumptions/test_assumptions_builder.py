import torch

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder, UniModalAssumptionsBuilder
from fedot.api.api_utils.assumptions.task_assumptions import TensorClassificationAssumptions
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _tensor_classification_data() -> TensorData:
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        features=torch.zeros((4, 2)),
        target=torch.tensor([0, 1, 0, 1]),
    )


def _pipeline_operations(pipeline):
    return [node.operation.operation_type for node in pipeline.nodes]


def test_assumptions_builder_get_uses_unimodal_builder_for_tensordata():
    builder = AssumptionsBuilder.get(_tensor_classification_data())

    assert isinstance(builder, UniModalAssumptionsBuilder)
    assert isinstance(builder.assumptions_generator, TensorClassificationAssumptions)


def test_tensordata_classification_assumption_uses_torch_linear():
    pipeline = AssumptionsBuilder.get(_tensor_classification_data()).build()[0]

    assert pipeline.root_node.operation.operation_type == 'torch_linear'
    assert _pipeline_operations(pipeline) == ['torch_linear']


def test_tensordata_assumption_does_not_add_legacy_preprocessing_nodes():
    pipeline = AssumptionsBuilder.get(_tensor_classification_data()).build()[0]

    assert 'scaling' not in _pipeline_operations(pipeline)
    assert pipeline.use_input_preprocessing is False
