from dataclasses import replace
import json

from golem.utilities.singleton_meta import SingletonMeta
import pytest
import torch

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.operations import operation_execution as discovery
from fedot.core.operations import operation as operation_module
from fedot.core.operations.model import Model
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.models.sklearn import SklearnModelStrategy
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_implementation import DeviceType, ExecutionPlatform
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


def data_for(name, device='cpu'):
    features = torch.randn(40, 4, generator=torch.Generator().manual_seed(42))
    target = (features[:, 0] > 0).long() if name == 'rf' else features[:, 0] + features[:, 1]
    return TensorData(features=features, target=target, idx=torch.arange(len(features)),
                      task=Task(TaskTypesEnum.classification if name == 'rf' else TaskTypesEnum.regression),
                      data_type=DataTypesEnum.tabular, numerical_idx=list(range(features.shape[1]))).to(device)


def forbidden(*args, **kwargs):
    pytest.fail('Fitted inference must not choose an implementation or reprocess training parameters')


class RecordingPredictionsCache:
    def __init__(self):
        self.entries = {}

    def load_node_prediction(self, descriptive_id, output_mode, fold_id, is_fit=False):
        return self.entries.get((descriptive_id, output_mode, fold_id, is_fit))

    def save_node_prediction(self, descriptive_id, output_mode, fold_id, outputData, is_fit=False):
        self.entries[(descriptive_id, output_mode, fold_id, is_fit)] = outputData


@pytest.mark.parametrize('name', ['rf', 'rfr'])
def test_cpu_execution_is_bound_and_prediction_reuses_strategy(name, monkeypatch):
    model = Model(name)
    data = data_for(name)
    fitted, _ = model.fit({'n_estimators': 3, 'random_state': 42}, data)
    strategy = model._eval_strategy
    state = fitted._fedot_execution
    assert state.resolution.implementation_id == 'sklearn'
    assert state.resolution.device == torch.device('cpu')

    monkeypatch.setattr(model, 'resolve_execution', forbidden)
    monkeypatch.setattr(model, '_init', forbidden)
    result = model.predict(fitted, replace(data, target=None), output_mode='labels' if name == 'rf' else 'default')
    assert model._eval_strategy is strategy
    assert result.device == data.device


def test_model_execution_parameters_are_prepared_before_estimator_creation(monkeypatch):
    def prepare(self, params, device):
        assert device == torch.device('cpu')
        params.update(n_estimators=2)
        return params

    monkeypatch.setattr(SklearnModelStrategy, 'prepare_execution_params', prepare)
    params = OperationParameters(n_estimators=3)
    fitted, _ = Model('rf').fit(params, data_for('rf'))
    assert fitted.n_estimators == 2
    assert fitted._fedot_execution.params.get('n_estimators') == 2
    assert params.get('n_estimators') == 3


def test_prediction_cache_uses_bound_execution_and_preserves_node_identity():
    cache = RecordingPredictionsCache()
    model = Model('rf')
    data = data_for('rf')
    cache.entries[('node', 'full_probs', 7, False)] = 'prediction from another execution'
    fitted, _ = model.fit({'n_estimators': 3}, data, predictions_cache=cache, fold_id=7, descriptive_id='node')
    result = model.predict(fitted, replace(data, target=None), output_mode='full_probs',
                           predictions_cache=cache, fold_id=7, descriptive_id='node')
    assert result.predict.shape == (len(data.features), 2)
    scoped = [key for key in cache.entries if '|execution=' in key[0]]
    assert len(scoped) == 2
    assert all(key[0].startswith('node|execution=sklearn:cpu:') for key in scoped)
    assert all('|output_device=cpu' in key[0] for key in scoped)


@pytest.mark.parametrize('on_disk', [False, True])
def test_pipeline_roundtrip_restores_bound_execution_without_selection(tmp_path, monkeypatch, on_disk):
    data = data_for('rf')
    node = PipelineNode('rf')
    node.parameters = {'n_estimators': 3, 'random_state': 42}
    pipeline = Pipeline(node)
    pipeline.fit(data)
    expected = pipeline.predict(replace(data, target=None), output_mode='full_probs')

    if on_disk:
        path = str(tmp_path / 'pipeline.json')
        pipeline.save(path)
        restored = Pipeline().load(path)
    else:
        source, operations = pipeline.save()
        for stream in operations.values():
            stream.seek(0)
        restored = Pipeline().load(json.loads(source), operations)

    monkeypatch.setattr(operation_module, 'discover_execution_environment', forbidden)
    monkeypatch.setattr(operation_module.HyperparametersPreprocessor, 'correct', forbidden)
    actual = restored.predict(replace(data, target=None), output_mode='full_probs')
    torch.testing.assert_close(actual.predict, expected.predict)
    assert restored.root_node.fitted_operation._fedot_execution.resolution.implementation_id == 'sklearn'


def test_cache_restores_state_and_replaces_an_incompatible_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(SingletonMeta, '_instances', {})
    cache = OperationsCache(cache_dir=str(tmp_path))
    data = data_for('rf')
    node = PipelineNode('rf')
    node.parameters = {'n_estimators': 3, 'random_state': 42}
    node.fit(data)
    state = node.fitted_operation._fedot_execution
    # Simulate a cached fit from a different variant. It must be replaced before reuse.
    node.fitted_operation._fedot_execution = replace(state, resolution=replace(state.resolution,
                                                                               implementation_id='previous'))
    cache.save_nodes(node)
    restored = PipelineNode('rf')
    restored.parameters = node.parameters
    cache.try_load_nodes(restored)
    assert restored.fitted_operation._fedot_execution.resolution.implementation_id == 'previous'
    restored.fit(data)
    assert restored.fitted_operation._fedot_execution.resolution.implementation_id == 'sklearn'
    cache.save_nodes(restored)

    again = PipelineNode('rf')
    again.parameters = node.parameters
    cache.try_load_nodes(again)
    assert again.fitted_operation._fedot_execution.resolution.implementation_id == 'sklearn'
    cached = again.fitted_operation
    again.fit(data)
    assert again.fitted_operation is cached
    monkeypatch.setattr(again.operation, 'resolve_execution', forbidden)
    expected = restored.predict(replace(data, target=None), output_mode='full_probs')
    actual = again.predict(replace(data, target=None), output_mode='full_probs')
    torch.testing.assert_close(actual.predict, expected.predict)


@pytest.mark.parametrize('system,release,expected', [
    ('Linux', '6.0', ExecutionPlatform.linux),
    ('Linux', '5.15-microsoft-standard-WSL2', ExecutionPlatform.wsl),
    ('Windows', '10', ExecutionPlatform.windows),
])
def test_discovery_imports_only_dependencies_eligible_for_platform_and_storage(monkeypatch, system, release, expected):
    metadata = OperationTypesRepository('model').operation_info_by_id('rf')
    monkeypatch.setattr(discovery.platform, 'system', lambda: system)
    monkeypatch.setattr(discovery.platform, 'release', lambda: release)
    imports = []
    monkeypatch.setattr(discovery, 'import_module', imports.append)
    environment = discovery.discover_execution_environment(metadata.implementations, torch.device('cuda:1'))
    assert environment.platform is expected
    assert set(imports) == ({'sklearn', 'cuml', 'cupy'} if expected is ExecutionPlatform.linux else {'sklearn'})


def test_cpu_discovery_does_not_import_gpu_libraries_and_missing_modules_remain_unavailable(monkeypatch):
    metadata = OperationTypesRepository('model').operation_info_by_id('rf')
    imports = []

    def missing(name):
        imports.append(name)
        raise ImportError(name)

    monkeypatch.setattr(discovery, 'import_module', missing)
    environment = discovery.discover_execution_environment(metadata.implementations, torch.device('cpu'))
    assert imports == ['sklearn']
    assert environment.available_dependencies == frozenset()
    assert environment.available_devices == frozenset((DeviceType.cpu,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('name', ['rf', 'rfr'])
def test_gpu_smoke_fit_roundtrip_and_inference_keep_cuml_on_cpu_input(name, monkeypatch):
    pytest.importorskip('cuml')
    data = data_for(name, 'cuda:0')
    node = PipelineNode(name)
    node.parameters = {'n_estimators': 3, 'max_depth': 3, 'random_state': 42, 'n_streams': 1}
    pipeline = Pipeline(node)
    pipeline.fit(data)
    state = node.fitted_operation._fedot_execution
    assert state.resolution.implementation_id == 'cuml'
    assert state.resolution.device == torch.device('cuda:0')
    assert node.fitted_operation.__class__.__module__.startswith('cuml.')
    expected = pipeline.predict(replace(data, target=None))

    source, operations = pipeline.save()
    for stream in operations.values():
        stream.seek(0)
    restored = Pipeline().load(json.loads(source), operations)
    monkeypatch.setattr(operation_module, 'discover_execution_environment', forbidden)
    actual = restored.predict(replace(data, target=None).to('cpu'))
    torch.testing.assert_close(actual.predict.cpu(), expected.predict.cpu())
    assert actual.device == torch.device('cpu')
    assert restored.root_node.operation._eval_strategy.execution.device == torch.device('cuda:0')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_gpu_smoke_multi_output_forest_selects_cpu_without_losing_targets():
    data = data_for('rfr', 'cuda:0')
    data.target = torch.stack((data.target, data.target * 2), dim=1)
    fitted, output = Model('rfr').fit({'n_estimators': 3, 'random_state': 42}, data)
    assert fitted._fedot_execution.resolution.implementation_id == 'sklearn'
    assert output.predict.shape == data.target.shape
    assert output.device == data.device
