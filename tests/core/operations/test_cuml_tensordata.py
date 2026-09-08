import importlib.util

import numpy as np
import pytest
import torch

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


CUML_INSTALLED = importlib.util.find_spec('cuml') is not None
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not CUML_INSTALLED or not torch.cuda.is_available(), reason='cuML CUDA runtime is unavailable'),
]


def _tensor_data(task_type: TaskTypesEnum, non_negative: bool = False) -> TensorData:
    features = torch.rand(512, 12, generator=torch.Generator().manual_seed(42))
    if not non_negative:
        features = 2 * features - 1
    features = features.cuda()
    if task_type is TaskTypesEnum.classification:
        threshold = 1 if non_negative else 0
        target = (features[:, 0] + features[:, 1] > threshold).float()[:, None]
    elif task_type is TaskTypesEnum.regression:
        target = (2 * features[:, 0] - features[:, 1])[:, None]
    else:
        target = None
    return TensorData(
        task=Task(task_type),
        data_type=DataTypesEnum.tabular,
        idx=list(range(len(features))),
        features=features,
        target=target,
    )


@pytest.mark.parametrize(
    ('operation', 'output_mode', 'non_negative'),
    [
        ('logit', 'full_probs', False),
        ('rf', 'full_probs', False),
        ('svc', 'full_probs', False),
        ('knn', 'full_probs', False),
        ('multinb', 'full_probs', True),
        ('bernb', 'full_probs', False),
        ('minibatchsgd', 'labels', False),
    ],
)
def test_cuml_classifiers_keep_tensordata_predictions_on_cuda(operation, output_mode, non_negative):
    from fedot.core.operations.evaluation.gpu.classification import CuMLClassificationStrategy

    data = _tensor_data(TaskTypesEnum.classification, non_negative=non_negative)
    strategy = CuMLClassificationStrategy(operation, OperationParameters.from_operation_type(operation))
    fitted = strategy.fit(data)
    strategy.output_mode = output_mode

    prediction = strategy.predict(fitted, data)

    assert isinstance(prediction, TensorData)
    assert prediction.predict.device.type == 'cuda'
    assert len(prediction.predict) == len(data.features)


@pytest.mark.parametrize(
    'operation',
    ['linear', 'ridge', 'lasso', 'elasticnet', 'rfr', 'knnreg', 'mbsgdcregr', 'cd'],
)
def test_cuml_regressors_keep_tensordata_predictions_on_cuda(operation):
    from fedot.core.operations.evaluation.gpu.regression import CuMLRegressionStrategy

    data = _tensor_data(TaskTypesEnum.regression)
    strategy = CuMLRegressionStrategy(operation, OperationParameters.from_operation_type(operation))
    fitted = strategy.fit(data)

    prediction = strategy.predict(fitted, data)

    assert isinstance(prediction, TensorData)
    assert prediction.predict.device.type == 'cuda'
    assert prediction.predict.shape == (len(data.features),)


def test_cuml_kmeans_keeps_tensordata_predictions_on_cuda():
    from fedot.core.operations.evaluation.gpu.clustering import CumlClusteringStrategy

    data = _tensor_data(TaskTypesEnum.clustering)
    strategy = CumlClusteringStrategy('kmeans', OperationParameters())
    fitted = strategy.fit(data)

    prediction = strategy.predict(fitted, data)

    assert isinstance(prediction, TensorData)
    assert prediction.predict.device.type == 'cuda'
    assert prediction.predict.shape == (len(data.features),)


def test_cuml_legacy_inputdata_path_still_returns_outputdata():
    from fedot.core.operations.evaluation.gpu.classification import CuMLClassificationStrategy

    features = np.random.default_rng(42).normal(size=(256, 8)).astype(np.float32)
    target = (features[:, 0] > 0).astype(np.float32)
    data = InputData.from_numpy(features, target, task='classification', data_type=DataTypesEnum.table)
    strategy = CuMLClassificationStrategy('logit', OperationParameters())
    fitted = strategy.fit(data)
    strategy.output_mode = 'full_probs'

    prediction = strategy.predict(fitted, data)

    assert isinstance(prediction, OutputData)
    assert isinstance(prediction.predict, np.ndarray)
    assert prediction.predict.shape == (len(features), 2)


def test_cuml_model_runs_in_tensordata_pipeline():
    data = _tensor_data(TaskTypesEnum.classification)
    OperationTypesRepository.assign_repo('model', 'gpu_models_repository.json')
    try:
        pipeline = Pipeline(PipelineNode('logit'))
        pipeline.fit(data)
        prediction = pipeline.predict(data, output_mode='full_probs')
    finally:
        OperationTypesRepository.assign_repo('model', 'model_repository.json')

    assert isinstance(prediction, TensorData)
    assert prediction.predict.device.type == 'cuda'
    assert prediction.predict.shape == (len(data.features), 2)


def test_tensor_data_operation_feeds_cuml_model_without_leaving_cuda():
    data = _tensor_data(TaskTypesEnum.classification)
    OperationTypesRepository.assign_repo('model', 'gpu_models_repository.json')
    try:
        model_node = PipelineNode('logit', nodes_from=[PipelineNode('pca')])
        pipeline = Pipeline(model_node)
        pipeline.fit(data)
        prediction = pipeline.predict(data, output_mode='full_probs')
    finally:
        OperationTypesRepository.assign_repo('model', 'model_repository.json')

    assert isinstance(prediction, TensorData)
    assert prediction.predict.device.type == 'cuda'
    assert prediction.predict.shape == (len(data.features), 2)
