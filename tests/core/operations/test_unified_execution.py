from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from fedot.api.help import _get_implementation_info
from fedot.core.operations.model import Model
from fedot.core.operations.operation_execution import prepare_device_params
from fedot.core.operations.operation_execution_rules import (
    ExecutionEnvironment, find_operation_execution, resolve_operation_execution,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_implementation import DeviceType, ExecutionPlatform, StrategyReference
from fedot.core.repository.operation_types_repository import OperationTypesRepository, create_repository_path
from tests.core.operations.test_operation_execution import data_for


@pytest.fixture(scope='module')
def catalogue():
    return {item.id: item for item in OperationTypesRepository._initialise_repo(
        create_repository_path('model_repository.json'))}


def environment(catalogue, platform=ExecutionPlatform.linux):
    dependencies = frozenset(dependency for item in catalogue.values()
                             for variant in item.implementations for dependency in variant.requires)
    return ExecutionEnvironment(platform, frozenset((DeviceType.cpu, DeviceType.cuda)), dependencies)


def test_all_migrated_models_have_explicit_devices_and_one_catalogue(catalogue):
    migrated = [item for item in catalogue.values() if item.implementations]
    assert len(migrated) == 54
    assert all(variant.devices for item in migrated for variant in item.implementations)
    assert not any(name.endswith('_gpu') for name in catalogue)
    assert not Path(create_repository_path('gpu_models_repository.json')).exists()
    defaults = json.loads(Path(create_repository_path('default_operation_params.json')).read_text())
    assert not any(name.endswith('_gpu') for name in defaults)


def test_help_can_describe_gpu_only_models_without_importing_their_strategy(catalogue, monkeypatch):
    monkeypatch.setattr(StrategyReference, 'load', lambda self: pytest.fail('Help imported a strategy'))
    info = _get_implementation_info(catalogue['cd'], None)
    assert info['cuml']['devices'] == ['cuda']
    assert info['cuml']['strategy'].endswith('CuMLRegressionStrategy')


def test_every_variant_selects_only_supported_devices_without_importing_models(catalogue):
    snapshot = environment(catalogue)
    for item in catalogue.values():
        if not item.implementations:
            continue
        for device in (torch.device('cpu'), torch.device('cuda:2')):
            result = find_operation_execution(item, item.task_type[0], device, snapshot)
            cuda = any(DeviceType.cuda in variant.devices for variant in item.implementations)
            cpu = any(DeviceType.cpu in variant.devices for variant in item.implementations)
            if device.type == 'cpu' and not cpu:
                assert result is None
                continue
            assert result is not None
            assert result.device == (device if cuda and device.type == 'cuda' else torch.device('cpu'))


def test_windows_does_not_select_cuml_but_can_select_other_cuda_models(catalogue):
    snapshot = environment(catalogue, ExecutionPlatform.windows)
    for name in ('rf', 'ridge', 'knn', 'logit'):
        item = catalogue[name]
        result = resolve_operation_execution(item, item.task_type[0], torch.device('cuda:0'), snapshot)
        assert result.device.type == 'cpu'
    for name in ('torch_mlp', 'xgboost', 'catboost', 'tabpfn'):
        item = catalogue[name]
        result = resolve_operation_execution(item, item.task_type[0], torch.device('cuda:0'), snapshot)
        assert result.device.type == 'cuda'


@pytest.mark.parametrize('name', ['lda', 'qda', 'cd'])
def test_single_target_models_are_unavailable_for_multioutput(catalogue, name):
    item = catalogue[name]
    assert find_operation_execution(item, item.task_type[0], torch.device('cuda:0'),
                                    environment(catalogue), multi_output=True) is None


@pytest.mark.parametrize('name,key,cpu_value,cuda_value', [
    ('torch_linear', 'device', 'cpu', 'cuda:2'),
    ('tabm', 'device', 'cpu', 'cuda:2'),
    ('ft_transformer', 'device', 'cpu', 'cuda:2'),
    ('realmlp', 'device', 'cpu', 'cuda:2'),
    ('tabpfn', 'device', 'cpu', 'cuda:2'),
    ('tabiclreg', 'device', 'cpu', 'cuda:2'),
    ('xgboost', 'device', 'cpu', 'cuda:2'),
    ('lgbm', 'device_type', 'cpu', 'gpu'),
    ('catboost', 'task_type', 'CPU', 'GPU'),
])
@pytest.mark.parametrize('device', ['cpu', 'cuda:2'])
def test_family_parameters_follow_the_same_execution_contract(catalogue, name, key, cpu_value, cuda_value, device):
    item = catalogue[name]
    params = OperationParameters.from_operation_type(name)
    before = params.to_dict()
    result = resolve_operation_execution(item, item.task_type[0], torch.device(device), environment(catalogue))
    model = Model(name)
    model._init(data_for('rf').task if item.task_type[0].value == 'classification' else data_for('rfr').task,
                params=params, execution=result, n_samples_data=32)
    actual = model._eval_strategy.params_for_fit
    assert actual.get(key) == (cpu_value if device == 'cpu' else cuda_value)
    assert params.to_dict() == before
    if name == 'lgbm' and device.startswith('cuda'):
        assert actual.get('gpu_device_id') == 2
    if name == 'catboost' and device.startswith('cuda'):
        assert actual.get('devices') == '2'


def test_device_conflict_is_reported_and_generic_cuda_keeps_the_selected_ordinal():
    with pytest.raises(ValueError, match='conflicts'):
        prepare_device_params(OperationParameters(device='cuda:0'), torch.device('cpu'))
    params = prepare_device_params(OperationParameters(device='cuda'), torch.device('cuda:2'))
    assert params.get('device') == 'cuda:2'


def test_foundation_explicit_limits_override_device_defaults(catalogue):
    model = Model('tabpfn')
    resolution = resolve_operation_execution(catalogue['tabpfn'], catalogue['tabpfn'].task_type[0],
                                             torch.device('cuda:0'), environment(catalogue))
    model._init(data_for('rf').task, params={'max_samples': 123, 'max_features': 77}, execution=resolution)
    assert model._eval_strategy.params_for_fit.get('max_samples') == 123
    assert model._eval_strategy.params_for_fit.get('max_features') == 77


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('name,params', [
    ('logit', {'max_iter': 100}), ('ridge', {}), ('knn', {'n_neighbors': 100}),
    ('xgboost', {'n_estimators': 2, 'max_depth': 2, 'use_eval_set': False}),
    ('lgbm', {'n_estimators': 2, 'num_leaves': 4, 'use_eval_set': False, 'verbosity': -1}),
    ('catboost', {'num_trees': 2, 'max_depth': 2, 'use_eval_set': False, 'verbose': False}),
    ('torch_linear', {'epochs': 1, 'validation_fraction': 0}),
])
def test_gpu_smoke_unified_catalogue_and_fixed_inference(name, params, monkeypatch):
    data = data_for('rfr' if name == 'ridge' else 'rf', 'cuda:0')
    node = PipelineNode(name)
    node.parameters = params
    pipeline = Pipeline(node)
    pipeline.fit(data)
    state = node.fitted_operation._fedot_execution
    assert state.resolution.device == torch.device('cuda:0')
    if name in {'logit', 'ridge', 'knn'}:
        assert state.resolution.implementation_id == 'cuml'
    if name == 'knn':
        assert node.fitted_operation.n_neighbors == round(len(data.features) / 2)
    monkeypatch.setattr(node.operation, 'resolve_execution', lambda *args: pytest.fail('Inference reselected'))
    result = pipeline.predict(replace(data, target=None))
    assert result.predict.device == data.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('power,expected', [(1, 5.), (2, 9.)])
def test_gpu_knn_distance_power_changes_the_nearest_neighbour(power, expected):
    data = replace(data_for('rfr', 'cuda:0'),
                   features=torch.tensor([[3., 0.], [2., 2.], [10., 10.], [11., 11.]], device='cuda:0'),
                   target=torch.tensor([5., 9., 20., 30.], device='cuda:0'),
                   idx=torch.arange(4, device='cuda:0'), numerical_idx=[0, 1])
    model = Model('knnreg')
    fitted, _ = model.fit({'n_neighbors': 1, 'p': power}, data)
    query = replace(data.subset_rows(torch.tensor([0], device='cuda:0')),
                    features=torch.zeros((1, 2), device='cuda:0'), target=None)
    prediction = model.predict(fitted, query).predict
    torch.testing.assert_close(prediction.reshape(-1), torch.tensor([expected], device='cuda:0'))
