"""Catalogue and pipeline serialization use the canonical operations packages."""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
@pytest.mark.parametrize('catalogue', ['model_repository', 'data_operation_repository'])
def test_catalogue_strategy_references_resolve_to_canonical_classes(catalogue):
    repository = Path(__file__).resolve().parents[3] / 'fedot/core/repository/data'
    metadata = json.loads((repository / f'{catalogue}.json').read_text())['metadata']
    for entry in metadata.values():
        strategies = entry['strategies']
        references = strategies.values() if isinstance(strategies, dict) else [strategies]
        for module, name in references:
            strategy = getattr(importlib.import_module(module), name)
            assert issubclass(strategy, EvaluationStrategy)
            assert strategy.__module__ == module


@pytest.mark.unit
@pytest.mark.parametrize('on_disk', [False, True])
def test_fitted_pipeline_roundtrip_with_canonical_implementations(tmp_path, on_disk):
    features = torch.randn(24, 4, generator=torch.Generator().manual_seed(23))
    data = TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        idx=list(range(len(features))),
        features=features,
        target=(features[:, 0] > 0).float()[:, None],
    )
    projection = PipelineNode('pca')
    projection.parameters = {'n_components': 2}
    model = PipelineNode('torch_linear', nodes_from=[projection])
    model.parameters = {'epochs': 2, 'validation_fraction': 0, 'random_state': 23}
    pipeline = Pipeline(model)
    pipeline.fit(data)
    predict_data = replace(data, target=None)
    expected = pipeline.predict(predict_data, output_mode='full_probs')

    if on_disk:
        source = str(tmp_path / 'pipeline.json')
        pipeline.save(source)
        restored = Pipeline().load(source)
    else:
        source, fitted_operations = pipeline.save()
        for operation in fitted_operations.values():
            operation.seek(0)
        restored = Pipeline().load(json.loads(source), fitted_operations)

    actual = restored.predict(predict_data, output_mode='full_probs')
    torch.testing.assert_close(actual.predict, expected.predict)
    np.testing.assert_array_equal(actual.idx, expected.idx)
    assert actual.target is None
    assert actual.device == data.device
    for node in restored.nodes:
        assert node.fitted_operation.__class__.__module__.startswith('fedot.core.operations.implementations.')
