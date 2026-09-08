import pytest

from fedot.core.operations.evaluation.gpu.rules import (
    adapt_cuml_parameters,
    build_cuml_precision_plan,
    build_cuml_runtime_plan,
    validate_cuml_tensor_operation,
)


@pytest.mark.unit
def test_cuml_forest_parameters_are_adapted_without_mutating_source():
    source = {'criterion': 'entropy', 'n_jobs': 4, 'n_estimators': 200}

    adapted = adapt_cuml_parameters('rf', source)

    assert source == {'criterion': 'entropy', 'n_jobs': 4, 'n_estimators': 200}
    assert adapted == {
        'split_criterion': 'entropy',
        'n_estimators': 200,
        'n_bins': 256,
        'output_type': 'cupy',
    }


@pytest.mark.unit
def test_cuml_svc_always_fits_probability_model():
    assert adapt_cuml_parameters('svc', {}) == {
        'probability': True,
        'output_type': 'cupy',
    }


@pytest.mark.unit
def test_cuml_logit_gets_conservative_convergence_defaults():
    assert adapt_cuml_parameters('logit', {}) == {
        'max_iter': 10000,
        'tol': 1e-7,
        'linesearch_max_iter': 200,
        'output_type': 'cupy',
    }


@pytest.mark.unit
def test_cuml_logit_uses_stable_precision_without_penalizing_other_models():
    assert build_cuml_precision_plan('logit').dtype_name == 'float64'
    assert build_cuml_precision_plan('rf').dtype_name == 'float32'


@pytest.mark.unit
def test_cuml_kmeans_keeps_historical_two_cluster_default():
    assert adapt_cuml_parameters('kmeans', {}) == {
        'n_clusters': 2,
        'output_type': 'cupy',
    }


@pytest.mark.unit
def test_cuml_runtime_plan_preserves_tensordata_result_device():
    tensor_plan = build_cuml_runtime_plan(is_tensor_data=True, feature_device='cuda:1')
    legacy_plan = build_cuml_runtime_plan(is_tensor_data=False, feature_device='cuda:1')

    assert tensor_plan.returns_tensor is True
    assert tensor_plan.result_device == 'cuda:1'
    assert legacy_plan.returns_tensor is False
    assert legacy_plan.result_device == 'cpu'


@pytest.mark.unit
def test_low_level_sgd_solver_is_not_exposed_as_tensordata_model():
    with pytest.raises(ValueError, match='low-level solver'):
        validate_cuml_tensor_operation('sgd')

    validate_cuml_tensor_operation('logit')
