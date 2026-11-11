"""Small library fits for stopping parameters, multioutput axes and categorical vocabularies."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fedot import create_data
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.models.boosting import (
    BoostingClassificationStrategy, BoostingRegressionStrategy,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum


pytestmark = pytest.mark.unit


def numeric_data(*, multi_output=False, regression=False):
    rng = np.random.RandomState(17)
    features = rng.normal(size=(90, 5)).astype(np.float32)
    target = (features[:, :2] if regression else (features[:, :2] > 0).astype(int))
    target = target if multi_output else target[:, 0]
    return TensorData(Task(TaskTypesEnum.regression if regression else TaskTypesEnum.classification),
                      DataTypesEnum.tabular, torch.from_numpy(features), target=torch.from_numpy(target.copy()),
                      idx=torch.arange(90), numerical_idx=list(range(5)), idx_mapping=dict(enumerate(range(5))),
                      features_names=['f0', 'f1', 'f2', 'f3', 'f4'])


def parameters(operation, **overrides):
    params = dict(n_jobs=1, use_eval_set=False, enable_categorical=False)
    if operation.startswith('catboost'):
        params.update(num_trees=8, max_depth=3, random_seed=17)
    elif operation.startswith('xgboost'):
        params.update(n_estimators=8, max_depth=3, random_state=17, nthread=1)
    else:
        params.update(n_estimators=8, max_depth=3, random_state=17, num_threads=1)
    return OperationParameters.from_operation_type(operation, **{**params, **overrides})


@pytest.mark.parametrize('operation', ['catboost', 'catboostreg'])
def test_catboost_disables_validation_parameters_when_no_eval_set_is_used(operation):
    data = numeric_data(regression=operation.endswith('reg'))
    params = parameters(operation, od_wait=17, od_type='IncToDec', od_pval=0.1)
    original = deepcopy(params.to_dict())
    cls = BoostingRegressionStrategy if operation.endswith('reg') else BoostingClassificationStrategy
    fitted = cls(operation, params).fit(data)
    native = fitted.model.get_params()
    assert native['use_best_model'] is False
    assert not set(native).intersection({'early_stopping_rounds', 'od_wait', 'od_type', 'od_pval'})
    assert fitted.predict(data).shape[0] == len(data.idx)
    assert params.to_dict() == original


@pytest.mark.parametrize('operation', ['catboost', 'catboostreg'])
def test_catboost_default_stopping_configuration_retains_its_effective_meaning(operation):
    data = numeric_data(regression=operation.endswith('reg'))
    cls = BoostingRegressionStrategy if operation.endswith('reg') else BoostingClassificationStrategy
    fitted = cls(operation, parameters(operation, use_eval_set=True)).fit(data)
    native = fitted.model.get_all_params()
    assert native['od_type'] == 'Iter'
    assert native['od_wait'] == 100
    assert native['use_best_model'] is True


def test_catboost_explicit_native_detector_configuration_overrides_defaults():
    params = parameters('catboostreg', use_eval_set=True, od_wait=17, od_type='IncToDec', od_pval=0.1)
    fitted = BoostingRegressionStrategy('catboostreg', params).fit(numeric_data(regression=True))
    native = fitted.model.get_all_params()
    assert native['od_type'] == 'IncToDec'
    assert native['od_wait'] == 17
    assert native['od_pval'] == pytest.approx(0.1)


@pytest.mark.parametrize('operation', ['catboost', 'xgboost', 'lgbm'])
def test_multioutput_probabilities_keep_every_target_and_sample_axis(operation):
    data = numeric_data(multi_output=True)
    strategy = BoostingClassificationStrategy(operation, parameters(operation))
    fitted = strategy.fit(data)
    raw = fitted.predict_proba(data)
    positive = raw[..., 1] if raw.ndim == 3 else raw
    assert fitted.is_multi_target is True
    for mode in ['default', 'probs', 'full_probs', 'labels']:
        strategy.output_mode = mode
        output = strategy.predict(fitted, data)
        assert isinstance(output, TensorData)
        result = output.predict
        assert result.device == data.device
        if mode == 'full_probs':
            assert result.shape == (90, 2, 2)
            torch.testing.assert_close(result[..., 1], positive)
            torch.testing.assert_close(result.sum(dim=-1), torch.ones_like(positive))
        elif mode == 'labels':
            torch.testing.assert_close(result, fitted.predict(data))
        else:
            assert result.shape == (90, 2)
            torch.testing.assert_close(result, positive)


@pytest.mark.parametrize('operation', ['catboost', 'xgboost', 'lgbm'])
def test_fitting_multioutput_then_single_target_does_not_change_strategy_or_implementation_class(operation):
    params = parameters(operation)
    original = params.to_dict()
    strategy = BoostingClassificationStrategy(operation, params)
    multi = strategy.fit(numeric_data(multi_output=True))
    single_data = numeric_data()
    single = strategy.fit(single_data)
    assert multi.is_multi_target is True
    assert single.is_multi_target is False
    assert 'is_multi_target' not in strategy.operation_impl.__dict__
    assert params.to_dict() == original
    strategy.output_mode = 'probs'
    assert strategy.predict(single, single_data).predict.shape == (90,)


def test_catboost_multioutput_without_eval_preserves_default_requested_validation_parameters():
    params = parameters('catboost', use_eval_set=True)
    fitted = BoostingClassificationStrategy('catboost', params).fit(numeric_data(multi_output=True))
    assert fitted.model.get_params()['use_best_model'] is False
    assert 'early_stopping_rounds' not in fitted.model.get_params()
    assert params.get('early_stopping_rounds') == 100
    assert params.get('use_eval_set') is True


@pytest.mark.parametrize('operation', ['catboostreg', 'xgboostreg', 'lgbmreg'])
def test_regression_multioutput_keeps_target_axes_and_does_not_change_requested_loss(operation):
    params = parameters(operation)
    original = params.to_dict()
    data = numeric_data(multi_output=True, regression=True)
    strategy = BoostingRegressionStrategy(operation, params)
    fitted = strategy.fit(data)
    assert strategy.predict(fitted, data).predict.shape == (90, 2)
    assert fitted.is_multi_target is True
    assert params.to_dict() == original


def test_lightgbm_multiclass_multioutput_keeps_each_target_class_axis():
    data = numeric_data(multi_output=True)
    data.target = torch.column_stack([torch.arange(90) % 3, (torch.arange(90) // 3) % 3])
    strategy = BoostingClassificationStrategy('lgbm', parameters('lgbm'))
    fitted = strategy.fit(data)
    raw = fitted.predict_proba(data)
    for mode in ['probs', 'full_probs']:
        strategy.output_mode = mode
        result = strategy.predict(fitted, data).predict
        assert result.shape == (90, 2, 3)
        for target in range(2):
            torch.testing.assert_close(result[:, target], raw[:, target])


@pytest.mark.parametrize('operation', ['xgboost', 'xgboostreg'])
def test_xgboost_native_fit_receives_torch_features_targets_and_eval_set(operation):
    data = numeric_data(regression=operation.endswith('reg'))
    strategy_class = BoostingRegressionStrategy if operation.endswith('reg') else BoostingClassificationStrategy
    strategy = strategy_class(operation, parameters(operation, use_eval_set=True))
    implementation = strategy.operation_impl(strategy.params_for_fit.copy())
    with patch.object(implementation.model, 'fit', wraps=implementation.model.fit) as native_fit:
        implementation.fit(data)
    features, target = native_fit.call_args.args
    evaluation = native_fit.call_args.kwargs['eval_set'][0]
    for values in (features, target, *evaluation):
        assert isinstance(values, torch.Tensor)
        assert values.device == data.device


@pytest.mark.parametrize('operation', ['xgboost', 'xgboostreg'])
@pytest.mark.parametrize('use_eval_set', [False, True])
def test_xgboost_native_categories_keep_training_codes_for_predict_batches(operation, use_eval_set):
    features = np.empty((90, 2), dtype=object)
    features[:, 0] = 0.
    features[:, 1] = ['blue', 'green', 'red'] * 30
    target = np.tile([0, 0, 1], 30)
    task = 'regression' if operation.endswith('reg') else 'classification'
    data = create_data(features, target=target, task=task, categorical_idx=[1],
                       features_names=['value', 'color'], use_cache=False)
    params = parameters(operation, enable_categorical=True, tree_method='hist', use_eval_set=use_eval_set,
                        learning_rate=1., min_child_weight=0, reg_lambda=0)
    cls = BoostingRegressionStrategy if operation.endswith('reg') else BoostingClassificationStrategy
    fitted = cls(operation, params).fit(data)
    assert fitted.model.get_params()['enable_categorical'] is True
    red = create_data(features[2:3], from_data=data, features_names=['value', 'color'], use_cache=False)
    torch.testing.assert_close(fitted.predict(red), fitted.predict(data)[2:3])
    if task == 'classification':
        torch.testing.assert_close(fitted.predict_proba(red), fitted.predict_proba(data)[2:3])
    unknown = create_data(np.array([[0., 'unseen']], dtype=object), from_data=data,
                          features_names=['value', 'color'], use_cache=False)
    missing = create_data(np.array([[0., np.nan]], dtype=object), from_data=data,
                          features_names=['value', 'color'], use_cache=False)
    torch.testing.assert_close(fitted.predict(unknown), fitted.predict(missing))


@pytest.mark.parametrize('operation', ['catboost', 'xgboost', 'lgbm'])
@pytest.mark.parametrize('impute', [False, True])
def test_native_categories_preserve_optional_imputation_choice_and_ignore_predict_target(operation, impute):
    features = np.empty((90, 2), dtype=object)
    features[:, 0] = 0.
    features[:, 1] = ['red'] * 40 + ['blue'] * 20 + ['green'] * 20 + [np.nan] * 10
    data = create_data(features, target=np.array([1] * 40 + [0] * 50), categorical_idx=[1],
                       features_names=['value', 'color'], use_cache=False)
    if impute:
        service = OptionalTabularService(use_cache=False).fit(data, {PreprocessingStepEnum.imputation: None})
        data = service.predict(deepcopy(data))
        assert data.categorical_features[-1, 0] == 'red'
    else:
        assert np.isnan(data.categorical_features[-1, 0])
    before = deepcopy(data)
    strategy = BoostingClassificationStrategy(operation, parameters(operation, enable_categorical=True))
    fitted = strategy.fit(data)
    strategy.output_mode = 'probs'
    predicted = strategy.predict(fitted, replace(data, target=None))
    assert predicted.target is None
    assert predicted.predict.shape == (90,)
    assert predicted.validate() is predicted
    torch.testing.assert_close(predicted.predict, strategy.predict(fitted, data).predict)
    assert data == before


@pytest.mark.parametrize('operation', ['catboost', 'xgboost', 'lgbm'])
def test_boosting_after_optional_preprocessing_node_uses_tensordata(operation):
    data = numeric_data()
    data = create_data(data.features, target=data.target, categorical_idx=[], use_cache=False)
    preprocessing = PipelineNode('optional_preprocessing')
    preprocessing.parameters = {'strategy': {PreprocessingStepEnum.imputation: None}, 'use_cache': False}
    model = PipelineNode(operation, nodes_from=[preprocessing])
    model.parameters = parameters(operation).to_dict()
    pipeline = Pipeline(model)
    output = pipeline.fit(deepcopy(data))
    assert isinstance(output, TensorData)
    assert isinstance(output.predict, torch.Tensor)
    predicted = pipeline.root_node.predict(replace(data, target=None), output_mode='probs')
    assert isinstance(predicted, TensorData)
    assert predicted.predict.shape[0] == 90


@pytest.mark.parametrize('case', ['regression', 'binary', 'multiclass', 'multilabel', 'softmax', 'dart'])
def test_xgboost_cuda_output_matches_sklearn_without_numpy_roundtrip(case):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    regression = case == 'regression'
    data = numeric_data(regression=regression, multi_output=case == 'multilabel').to('cuda:0')
    if case in ['multiclass', 'softmax']:
        data.target = torch.arange(len(data.features), device=data.device) % 3
    overrides = dict(device='cuda:0', tree_method='hist', use_eval_set=True)
    if case == 'softmax':
        overrides['objective'] = 'multi:softmax'
    if case == 'dart':
        overrides['booster'] = 'dart'
    operation = 'xgboostreg' if regression else 'xgboost'
    cls = BoostingRegressionStrategy if regression else BoostingClassificationStrategy
    strategy = cls(operation, parameters(operation, **overrides))
    fitted = strategy.fit(data)
    features = fitted._prepare_features(data)
    labels = torch.as_tensor(fitted.model.predict(features), device=data.device)
    probabilities = None if regression else torch.as_tensor(fitted.model.predict_proba(features), device=data.device)
    # The wrapper must use the native GPU path, not sklearn's host-output methods.
    with patch.object(fitted.model, 'predict', side_effect=AssertionError('sklearn predict was used')):
        with patch.object(fitted.model, 'predict_proba', create=True,
                          side_effect=AssertionError('sklearn predict_proba was used')):
            actual = fitted.predict(data)
            torch.testing.assert_close(actual, labels, rtol=0, atol=0)
            cpu_result = fitted.predict(deepcopy(data).to('cpu'))
            torch.testing.assert_close(cpu_result, labels.cpu(), rtol=0, atol=0)
            if not regression:
                actual = fitted.predict_proba(data)
                torch.testing.assert_close(actual, probabilities, rtol=1e-6, atol=1e-7)
                strategy.output_mode = 'full_probs'
                assert strategy.predict(fitted, data).predict.device == data.device
