import numpy as np
import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.evaluation.optional_preprocessing import (
    TensorOptionalPreprocessingStrategy,
)
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import (
    ImputationMethodEnum,
    PreprocessingStepEnum,
)


def _train_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 0.0],
            [4.0, np.nan, 1.0],
            [7.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )


def _test_features() -> np.ndarray:
    return np.array(
        [
            [10.0, np.nan, 0.0],
            [11.0, 12.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def train_td() -> TensorData:
    return TensorDataCreator.create(_train_features(), backend_name='cpu')


@pytest.fixture
def test_td() -> TensorData:
    return TensorDataCreator.create(_test_features(), backend_name='cpu')


@pytest.mark.unit
def test_pipeline_node_wires_optional_preprocessing_operation_and_strategy():
    node = PipelineNode('optional_preprocessing')

    assert node.name == 'optional_preprocessing'
    assert isinstance(node.operation, DataOperation)
    assert node.operation.operation_type == 'optional_preprocessing'
    assert node.is_primary

    node.operation._init(
        task=Task(TaskTypesEnum.classification),
        params=node.parameters,
        n_samples_data=3,
    )
    assert isinstance(
        node.operation._eval_strategy,
        TensorOptionalPreprocessingStrategy,
    )


@pytest.mark.unit
def test_pipeline_node_fit_predict_imputes_missing_values(train_td, test_td):
    node = PipelineNode('optional_preprocessing')
    node.parameters = {
        'strategy': {
            PreprocessingStepEnum.imputation: [{
                'method': ImputationMethodEnum.mean,
                'features_idx': [1],
                'step_args': None,
            }]
        }
    }

    fitted_output = node.fit(train_td)

    assert isinstance(node.fitted_operation, OptionalTabularService)
    assert isinstance(fitted_output, TensorData)
    assert not torch.isnan(fitted_output.features[:, 1]).any()
    assert fitted_output.features[1, 1] == 5.0

    predicted = node.predict(test_td)

    assert isinstance(predicted, TensorData)
    assert predicted.features.shape == test_td.features.shape
    assert not torch.isnan(predicted.features[:, 1]).any()
    assert predicted.features[0, 1] == 5.0


@pytest.mark.unit
def test_pipeline_fit_with_optional_preprocessing_node(train_td, test_td):
    node = PipelineNode('optional_preprocessing')
    node.parameters = {
        'strategy': {
            PreprocessingStepEnum.imputation: [{
                'method': ImputationMethodEnum.mean,
                'features_idx': [1],
                'step_args': None,
            }]
        }
    }
    pipeline = Pipeline(node)

    fitted_output = pipeline.fit(train_td)

    assert pipeline.is_fitted
    assert isinstance(pipeline.root_node.fitted_operation, OptionalTabularService)
    assert isinstance(fitted_output, TensorData)
    assert not torch.isnan(fitted_output.features[:, 1]).any()
    assert fitted_output.features[1, 1] == 5.0

    # Avoid Pipeline.predict postprocess (still InputData-oriented); call the node path.
    predicted = pipeline.root_node.predict(test_td)
    assert not torch.isnan(predicted.features[:, 1]).any()
    assert predicted.features[0, 1] == 5.0
