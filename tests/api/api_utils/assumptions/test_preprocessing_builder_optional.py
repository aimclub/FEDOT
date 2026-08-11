import numpy as np
import pytest

from fedot.api.api_utils.assumptions.preprocessing_builder import PreprocessingBuilder
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.tasks import TaskTypesEnum


def _tensor_data(features: np.ndarray):
    return TensorDataCreator.create(features, backend_name='cpu')


@pytest.mark.unit
def test_preprocessing_builder_adds_optional_preprocessing_for_tensor_data():
    td = _tensor_data(np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]], dtype=np.float32))
    pipeline = PreprocessingBuilder.builder(TaskTypesEnum.classification, td).build()

    assert [node.name for node in pipeline.nodes] == ['optional_preprocessing']


@pytest.mark.unit
def test_preprocessing_builder_skips_optional_preprocessing_when_disabled():
    td = _tensor_data(np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]], dtype=np.float32))
    builder = PreprocessingBuilder.builder(
        TaskTypesEnum.classification,
        td,
        use_optional_preprocessing=False,
    )

    assert builder.to_nodes() == []
    assert builder.build() is None


@pytest.mark.unit
@pytest.mark.parametrize(
    'task_type, features',
    [
        (TaskTypesEnum.regression, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        (TaskTypesEnum.classification, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        (TaskTypesEnum.ts_forecasting, np.arange(20, dtype=np.float32).reshape(-1, 1)),
    ],
)
def test_preprocessing_builder_for_tensor_data_adds_optional_preprocessing(task_type, features):
    pipeline = PreprocessingBuilder.builder(task_type, _tensor_data(features)).build()
    operations = [node.operation.operation_type for node in pipeline.nodes]

    assert 'optional_preprocessing' in operations
    assert 'scaling' not in operations


@pytest.mark.unit
def test_optional_preprocessing_search_space_contains_imputation_and_scaling():
    space = PipelineSearchSpace()
    params = space.get_parameters_for_operation('optional_preprocessing')

    assert 'use_imputation' in params
    assert 'imputation_method' in params
    assert 'use_scaling' in params
    assert 'scaling_method' in params
