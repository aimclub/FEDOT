from datetime import timedelta
from unittest.mock import Mock

import func_timeout
import pytest
import torch

from fedot.core.data.tensor_data import TensorData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_timed_worker_cannot_publish_late_state_after_timeout(monkeypatch):
    graph = Pipeline(PrimaryNode('torch_linear'))
    data = TensorData(Task(TaskTypesEnum.regression), DataTypesEnum.table, torch.ones((2, 2)))
    pending = []
    cleaned = []
    original_unfit = Pipeline.unfit

    def late_fit(self, tensor_data, state, fitted, *args):
        self.nodes[0].fitted_operation = object()
        state['train_predicted'] = tensor_data
        state['computation_time_in_seconds'] = 1
        fitted.append(self.nodes[0].fitted_operation)

    def timeout(seconds, function, args):
        pending.append(lambda: function(*args))
        raise func_timeout.FunctionTimedOut()

    def unfit(self, *args, **kwargs):
        cleaned.append(self)
        original_unfit(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, '_fit', late_fit)
    monkeypatch.setattr(Pipeline, 'unfit', unfit)
    monkeypatch.setattr(func_timeout, 'func_timeout', timeout)
    with pytest.raises(TimeoutError):
        graph.fit(data, time_constraint=timedelta(milliseconds=100))
    assert not graph.is_fitted
    pending[0]()
    assert not graph.is_fitted
    assert len(cleaned) == 1 and cleaned[0] is not graph
    assert not cleaned[0].is_fitted


def test_timed_worker_releases_owned_state_on_success(monkeypatch):
    graph = Pipeline(PrimaryNode('torch_linear'))
    data = TensorData(Task(TaskTypesEnum.regression), DataTypesEnum.table, torch.ones((2, 2)))
    worker = []

    def fit(self, tensor_data, state, fitted, *args):
        worker.append(self)
        self.nodes[0].fitted_operation = 'fitted-model'
        state.update(train_predicted=tensor_data, computation_time_in_seconds=1)
        fitted.append(self.nodes[0].fitted_operation)

    monkeypatch.setattr(Pipeline, '_fit', fit)
    assert graph.fit(data, time_constraint=timedelta(seconds=1)) == data
    assert graph.is_fitted and not worker[0].is_fitted
    graph.unfit()


def test_timed_worker_does_not_publish_state_when_cache_commit_fails(monkeypatch):
    graph = Pipeline(PrimaryNode('torch_linear'))
    data = TensorData(Task(TaskTypesEnum.regression), DataTypesEnum.table, torch.ones((2, 2)))
    backend = Mock()
    backend.save_node_prediction.side_effect = RuntimeError('cache commit failed')

    def fit(self, tensor_data, state, fitted, predictions_cache, fold_id):
        self.nodes[0].fitted_operation = 'fitted-model'
        predictions_cache.save_node_prediction('node', 'raw', fold_id, tensor_data)
        state.update(train_predicted=tensor_data, computation_time_in_seconds=1)
        fitted.append(self.nodes[0].fitted_operation)

    monkeypatch.setattr(Pipeline, '_fit', fit)

    with pytest.raises(RuntimeError, match='cache commit failed'):
        graph.fit(
            data,
            time_constraint=timedelta(seconds=1),
            predictions_cache=backend,
            fold_id=0,
        )

    assert not graph.is_fitted
    assert graph.computation_time is None
    backend.save_node_prediction.assert_called_once()


def test_timed_worker_cannot_publish_late_prediction_after_timeout(monkeypatch):
    graph = Pipeline(PrimaryNode('torch_linear'))
    data = TensorData(Task(TaskTypesEnum.regression), DataTypesEnum.table, torch.ones((2, 2)))
    backend = Mock()
    pending = []

    def late_fit(self, tensor_data, state, fitted, predictions_cache, fold_id):
        predictions_cache.save_node_prediction('node', 'raw', fold_id, tensor_data)
        state.update(train_predicted=tensor_data, computation_time_in_seconds=1)
        fitted.append(object())

    def timeout(seconds, function, args):
        pending.append(lambda: function(*args))
        raise func_timeout.FunctionTimedOut()

    monkeypatch.setattr(Pipeline, '_fit', late_fit)
    monkeypatch.setattr(func_timeout, 'func_timeout', timeout)

    with pytest.raises(TimeoutError):
        graph.fit(data, time_constraint=timedelta(milliseconds=100), predictions_cache=backend, fold_id=0)

    with pytest.raises(RuntimeError, match='closed'):
        pending[0]()
    backend.save_node_prediction.assert_not_called()
