from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fedot.api.experimental import tabular_portfolio as fedot_exec


def test_logloss_selection_score_prefers_better_probabilities():
    truth = np.array([0, 0, 1, 1])
    good = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    bad = good[:, ::-1]

    assert fedot_exec._selection_score("logloss", truth, None, good) > (
        fedot_exec._selection_score("logloss", truth, None, bad)
    )


def test_temperature_scaling_improves_underconfident_probabilities():
    truth = np.repeat([0, 1], 50)
    probabilities = np.vstack(
        (
            np.tile([0.65, 0.35], (50, 1)),
            np.tile([0.35, 0.65], (50, 1)),
        )
    )

    temperature = fedot_exec._fit_temperature(probabilities, truth)
    calibrated = fedot_exec._apply_temperature(probabilities, temperature)

    assert 0.1 < temperature < 1.0
    assert calibrated.sum(axis=1) == pytest.approx(np.ones(len(truth)))
    assert fedot_exec._selection_score("logloss", truth, None, calibrated) > (
        fedot_exec._selection_score("logloss", truth, None, probabilities)
    )


def test_direct_fixed_xgboost_temperature_is_conservative_smoothing():
    probabilities = np.array([[0.9, 0.1], [0.25, 0.75]])
    smoothed = fedot_exec._apply_temperature(
        probabilities, fedot_exec._DIRECT_FIXED_XGBOOST_TEMPERATURE
    )

    assert fedot_exec._DIRECT_FIXED_XGBOOST_TEMPERATURE == pytest.approx(1.015)
    assert smoothed.sum(axis=1) == pytest.approx(np.ones(2))
    assert np.all(
        np.max(smoothed, axis=1) < np.max(probabilities, axis=1)
    )


def test_direct_xgboost_temperature_is_only_used_at_validated_horizon():
    assert fedot_exec._direct_xgboost_temperature(259, 260) == 1.0
    assert fedot_exec._direct_xgboost_temperature(260, 260) == pytest.approx(
        fedot_exec._DIRECT_FIXED_XGBOOST_TEMPERATURE
    )
    with pytest.raises(ValueError, match="must be positive"):
        fedot_exec._direct_xgboost_temperature(0, 260)
    with pytest.raises(ValueError, match="exceed"):
        fedot_exec._direct_xgboost_temperature(261, 260)


def test_direct_fixed_xgboost_uses_validated_l2_regularisation():
    base_params = {"learning_rate": 0.15, "max_depth": 4}

    direct_params = fedot_exec._fixed_round_xgboost_model_params(
        base_params, seed=42
    )

    assert direct_params == {
        "learning_rate": 0.15,
        "max_depth": 4,
        "tree_method": "hist",
        "random_state": 42,
        "reg_lambda": 3.0,
    }
    assert base_params == {"learning_rate": 0.15, "max_depth": 4}


def test_direct_xgboost_fit_limit_preserves_non_fit_runtime_reserve():
    runtime_seconds = 180

    fit_time_limit = (
        runtime_seconds - fedot_exec._DIRECT_XGBOOST_NON_FIT_RESERVE_SECONDS
    )

    assert fit_time_limit == pytest.approx(145.0)


def test_direct_xgboost_refit_uses_only_safe_remaining_fit_budget():
    fit_time_limit = fedot_exec._direct_xgboost_refit_time_limit(
        model="xgboost",
        direct_rounds=120,
        elapsed_seconds=65.0,
        runtime_seconds=180.0,
        prediction_reserve=8.0,
        refit_start_safety_reserve=14.4,
    )

    assert fit_time_limit == pytest.approx(82.6)


@pytest.mark.parametrize(
    ("model", "direct_rounds", "elapsed_seconds"),
    [("lgbm", 120, 65.0), ("xgboost", None, 65.0), ("xgboost", 120, 125.0)],
)
def test_direct_xgboost_refit_rejects_inapplicable_or_too_short_deadline(
    model, direct_rounds, elapsed_seconds
):
    assert (
        fedot_exec._direct_xgboost_refit_time_limit(
            model=model,
            direct_rounds=direct_rounds,
            elapsed_seconds=elapsed_seconds,
            runtime_seconds=180.0,
            prediction_reserve=8.0,
            refit_start_safety_reserve=14.4,
        )
        is None
    )


def _bounded_medium_wide_geometry(row_count=63_000, feature_count=784):
    return SimpleNamespace(
        shape=(row_count, feature_count), dtype=np.dtype(np.float32)
    )


def test_bounded_medium_wide_xgboost_geometry_accepts_validated_regime(
    monkeypatch,
):
    features = _bounded_medium_wide_geometry()
    target = np.arange(features.shape[0]) % 10
    monkeypatch.setattr(fedot_exec, "_sampled_numeric_density", lambda data: 0.15)

    assert fedot_exec._is_bounded_medium_wide_numeric_multiclass(
        features, target
    )


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count"),
    [
        (19_999, 784, 10),
        (100_001, 784, 10),
        (63_000, 511, 10),
        (63_000, 1_025, 10),
        (100_000, 801, 10),
        (63_000, 784, 4),
        (63_000, 784, 21),
    ],
)
def test_bounded_medium_wide_xgboost_geometry_rejects_outside_boundaries(
    monkeypatch, row_count, feature_count, class_count
):
    features = _bounded_medium_wide_geometry(row_count, feature_count)
    target = np.arange(row_count) % class_count
    monkeypatch.setattr(fedot_exec, "_sampled_numeric_density", lambda data: 0.5)

    assert not fedot_exec._is_bounded_medium_wide_numeric_multiclass(
        features, target
    )


def test_bounded_medium_wide_xgboost_geometry_rejects_low_support_and_density(
    monkeypatch,
):
    features = _bounded_medium_wide_geometry()
    target = np.concatenate(
        (np.zeros(499, dtype=int), 1 + np.arange(features.shape[0] - 499) % 9)
    )
    monkeypatch.setattr(fedot_exec, "_sampled_numeric_density", lambda data: 0.5)
    assert not fedot_exec._is_bounded_medium_wide_numeric_multiclass(
        features, target
    )

    supported_target = np.arange(features.shape[0]) % 10
    monkeypatch.setattr(fedot_exec, "_sampled_numeric_density", lambda data: 0.149)
    assert not fedot_exec._is_bounded_medium_wide_numeric_multiclass(
        features, supported_target
    )


def test_deadline_bounded_xgboost_params_add_only_validated_pair(monkeypatch):
    base = {"max_depth": 4}
    monkeypatch.setattr(
        fedot_exec,
        "_is_bounded_medium_wide_numeric_multiclass",
        lambda features, target: True,
    )

    params = fedot_exec._deadline_bounded_xgboost_model_params(
        base,
        object(),
        np.arange(10),
        metric="logloss",
        bounded_refit_seconds=75.0,
        direct_rounds=100,
    )

    assert params == {"max_depth": 4, "max_bin": 128, "reg_lambda": 3.0}
    assert base == {"max_depth": 4}
    assert fedot_exec._candidate_model_params("xgboost") == {}


@pytest.mark.parametrize(
    "override",
    [
        {"metric": "auc"},
        {"bounded_refit_seconds": None},
        {"bounded_refit_seconds": 74.99},
        {"direct_rounds": None},
        {"direct_rounds": 99},
        {"model_params": {"max_bin": 255}},
        {"model_params": {"reg_lambda": 1.0}},
    ],
)
def test_deadline_bounded_xgboost_params_preserve_other_contracts(
    monkeypatch, override
):
    monkeypatch.setattr(
        fedot_exec,
        "_is_bounded_medium_wide_numeric_multiclass",
        lambda features, target: True,
    )
    kwargs = {
        "model_params": {"max_depth": 4},
        "features": object(),
        "target": np.arange(10),
        "metric": "logloss",
        "bounded_refit_seconds": 85.0,
        "direct_rounds": 120,
    }
    kwargs.update(override)
    original = dict(kwargs["model_params"])

    assert fedot_exec._deadline_bounded_xgboost_model_params(**kwargs) == original
    assert kwargs["model_params"] == original


def test_deadline_bounded_xgboost_params_require_structural_geometry(monkeypatch):
    base = {"max_depth": 4}
    monkeypatch.setattr(
        fedot_exec,
        "_is_bounded_medium_wide_numeric_multiclass",
        lambda features, target: False,
    )

    assert fedot_exec._deadline_bounded_xgboost_model_params(
        base,
        object(),
        np.arange(10),
        metric="logloss",
        bounded_refit_seconds=85.0,
        direct_rounds=120,
    ) == base


@pytest.mark.parametrize(
    ("metric", "bounded_refit_seconds", "selected_models", "expected"),
    [
        ("logloss", 82.6, ["xgboost"], True),
        ("logloss", None, ["xgboost"], False),
        ("logloss", 82.6, ["xgboost", "lgbm"], False),
        ("auc", 82.6, ["xgboost"], False),
    ],
)
def test_only_bounded_singleton_xgboost_uses_raw_calibration(
    metric, bounded_refit_seconds, selected_models, expected
):
    assert (
        fedot_exec._use_raw_bounded_xgboost_calibration(
            metric=metric,
            bounded_refit_seconds=bounded_refit_seconds,
            selected_models=selected_models,
        )
        is expected
    )


def test_temperature_scaling_can_be_enabled_for_sparse_oof_classes():
    truth = np.repeat(np.arange(25), 4)
    probabilities = np.full((len(truth), 25), 0.8 / 24)
    probabilities[np.arange(len(truth)), truth] = 0.2

    assert fedot_exec._fit_temperature(probabilities, truth) == 1.0
    assert fedot_exec._fit_temperature(
        probabilities, truth, allow_sparse_classes=True
    ) < 1.0


def test_temperature_scaling_can_use_singletons_when_reusing_holdout_model():
    truth = np.repeat(np.arange(4), [997, 1, 1, 1])
    probabilities = np.full((len(truth), 4), 0.4 / 3)
    probabilities[np.arange(len(truth)), truth] = 0.6

    assert fedot_exec._fit_temperature(probabilities, truth) == 1.0
    assert fedot_exec._fit_temperature(
        probabilities, truth, allow_singleton_classes=True
    ) < 1.0


def test_prior_exponent_corrects_a_single_frequency_bias():
    truth = np.repeat([0, 1], 50)
    probabilities = np.vstack(
        (
            np.tile([0.8, 0.2], (50, 1)),
            np.tile([0.4, 0.6], (50, 1)),
        )
    )
    reference_target = np.repeat([0, 1], [240, 60])

    exponent = fedot_exec._fit_prior_exponent(
        probabilities,
        truth,
        reference_target=reference_target,
    )
    calibrated = fedot_exec._apply_prior_exponent(
        probabilities,
        reference_target=reference_target,
        exponent=exponent,
    )

    assert -fedot_exec._PRIOR_CALIBRATION_MAX_ABS_EXPONENT < exponent < 0.0
    assert calibrated.sum(axis=1) == pytest.approx(np.ones(len(truth)))
    assert fedot_exec._selection_score("logloss", truth, None, calibrated) > (
        fedot_exec._selection_score("logloss", truth, None, probabilities)
    )


def test_prior_exponent_stays_neutral_without_support_or_prior_contrast():
    sparse_truth = np.repeat([0, 1], [97, 3])
    sparse_probabilities = np.tile([0.7, 0.3], (100, 1))
    balanced_truth = np.repeat([0, 1], 50)
    balanced_probabilities = np.vstack(
        (
            np.tile([0.8, 0.2], (50, 1)),
            np.tile([0.4, 0.6], (50, 1)),
        )
    )

    assert fedot_exec._fit_prior_exponent(
        sparse_probabilities,
        sparse_truth,
        reference_target=sparse_truth,
    ) == 0.0
    assert fedot_exec._fit_prior_exponent(
        balanced_probabilities,
        balanced_truth,
        reference_target=balanced_truth,
    ) == 0.0


def test_prior_exponent_stays_neutral_on_tiny_training_samples():
    truth = np.repeat([0, 1], 50)
    probabilities = np.vstack(
        (
            np.tile([0.8, 0.2], (50, 1)),
            np.tile([0.4, 0.6], (50, 1)),
        )
    )
    reference_target = np.repeat([0, 1], [136, 35])

    assert fedot_exec._fit_prior_exponent(
        probabilities,
        truth,
        reference_target=reference_target,
    ) == 0.0


def test_holdout_caps_selector_training_rows_and_keeps_all_classes():
    features = np.arange(900).reshape(300, 3)
    target = np.repeat([0, 1, 2], 100)

    X_train, X_valid, y_train, y_valid = fedot_exec._classification_holdout(
        features,
        target,
        validation_fraction=0.2,
        max_train_rows=90,
        seed=42,
    )

    assert X_train.shape == (90, 3)
    assert X_valid.shape == (60, 3)
    assert set(y_train) == set(y_valid) == {0, 1, 2}


def test_sparse_encoded_target_and_features_are_normalised_and_sliced():
    features = sparse.csr_matrix(np.arange(900).reshape(300, 3))
    encoded_target = sparse.csr_matrix(np.repeat([0, 1, 2], 100).reshape(-1, 1))
    target = fedot_exec._target_array(encoded_target)

    X_train, X_valid, y_train, y_valid = fedot_exec._classification_holdout(
        features,
        target,
        validation_fraction=0.2,
        max_train_rows=90,
        seed=42,
    )

    assert target.shape == (300,)
    assert sparse.issparse(X_train)
    assert sparse.issparse(X_valid)
    assert X_train.shape == (90, 3)
    assert X_valid.shape == (60, 3)
    assert set(y_train) == set(y_valid) == {0, 1, 2}


def test_non_contiguous_encoded_classes_are_restored_for_declared_output():
    target = np.array([0.0, 2.0, 5.0, 2.0])
    contiguous, observed = fedot_exec._contiguous_classification_target(target)
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )

    predictions, restored = fedot_exec._restore_classification_label_space(
        probabilities, observed, encoded_class_count=6
    )

    assert contiguous.tolist() == [0, 1, 2, 1]
    assert predictions.tolist() == [0, 2, 5]
    assert restored.shape == (3, 6)
    assert restored[:, [0, 2, 5]] == pytest.approx(probabilities)
    assert restored[:, [1, 3, 4]] == pytest.approx(0.0)


def test_capped_stratified_sample_keeps_extremely_rare_classes():
    target = np.repeat(np.arange(4), [10_000, 100, 3, 1])

    indices = fedot_exec._stratified_sample_indices_with_class_coverage(
        target, sample_size=100, seed=42
    )
    sampled_target = target[indices]

    assert len(indices) == 100
    assert len(np.unique(indices)) == 100
    assert set(sampled_target) == {0, 1, 2, 3}
    assert np.count_nonzero(sampled_target == 0) > np.count_nonzero(
        sampled_target == 1
    )


def test_capped_sample_preserves_sklearn_split_when_all_classes_survive():
    target = np.repeat(np.arange(4), 100)
    expected, _ = fedot_exec.train_test_split(
        np.arange(len(target)),
        train_size=100,
        random_state=42,
        stratify=target,
    )

    actual = fedot_exec._stratified_sample_indices_with_class_coverage(
        target, sample_size=100, seed=42
    )

    assert np.array_equal(actual, expected)


def test_holdout_keeps_singleton_classes_only_in_training():
    features = np.arange(63).reshape(21, 3)
    target = np.array([0] * 10 + [1] * 10 + [2])

    X_train, X_valid, y_train, y_valid = fedot_exec._classification_holdout(
        features,
        target,
        validation_fraction=0.2,
        max_train_rows=100,
        seed=42,
    )

    assert set(y_train) == {0, 1, 2}
    assert set(y_valid) == {0, 1}
    assert 2 not in y_valid
    assert set(X_train[:, 0]).isdisjoint(X_valid[:, 0])


def test_logloss_score_supports_a_validation_subset_of_training_classes():
    truth = np.array([0, 1])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    score = fedot_exec._selection_score(
        "logloss", truth, None, probabilities, labels=np.array([0, 1, 2])
    )

    assert np.isfinite(score)
    assert fedot_exec._fit_temperature(
        probabilities, truth, labels=np.array([0, 1, 2])
    ) == 1.0


def test_auto_cost_estimate_accounts_for_all_measured_boosters():
    observations = {"lgbm": 5.0, "xgboost": 4.0}

    assert fedot_exec._estimated_candidate_fit_seconds("auto", observations) == 60.0
    assert fedot_exec._estimated_candidate_fit_seconds("xgboost", {"lgbm": 5.0}) == 5.0


def test_candidate_start_does_not_protect_an_impossible_leader_refit():
    assert fedot_exec._candidate_start_refit_reserve(80.0, 100.0) == 80.0
    assert fedot_exec._candidate_start_refit_reserve(101.0, 100.0) == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        fedot_exec._candidate_start_refit_reserve(-1.0, 100.0)


def test_selector_row_cap_uses_feature_and_class_complexity():
    narrow = np.zeros((10_000, 10))
    wide = np.zeros((10_000, 800))
    target = np.arange(10_000) % 7
    extreme_wide = np.broadcast_to(np.zeros((1, 7_200)), (9_000, 7_200))
    extreme_target = np.arange(9_000) % 10

    assert fedot_exec._adaptive_selector_train_rows(narrow, target, 20_000) == 20_000
    assert fedot_exec._adaptive_selector_train_rows(wide, target, 20_000) == 2_500
    assert (
        fedot_exec._adaptive_selector_train_rows(
            extreme_wide, extreme_target, 20_000, xgboost_max_bin=32
        )
        == 7_200
    )
    assert (
        fedot_exec._adaptive_selector_train_rows(
            extreme_wide, extreme_target, 20_000, xgboost_max_bin=64
        )
        == 3_600
    )


def test_refit_scale_uses_sublinear_row_growth_and_mean_fold_cost():
    assert fedot_exec._portfolio_refit_scale(100, 100, 3) == pytest.approx(0.35)
    assert fedot_exec._portfolio_refit_scale(8_000, 2_500, 1) == pytest.approx(
        3.2**0.75 * 1.05
    )
    assert fedot_exec._portfolio_refit_scale(
        63_000, 2_500, 1, feature_count=784
    ) == pytest.approx(25.2 ** (2 / 3) * 1.15)
    assert fedot_exec._portfolio_refit_scale(
        63_000, 2_500, 1, feature_count=255
    ) == pytest.approx(25.2**0.75 * 1.05)


@pytest.mark.parametrize("forest_name", ["rf", "rf_large_subspace"])
def test_secondary_rf_refit_does_not_inherit_extreme_booster_scale(forest_name):
    forest = {
        "model": forest_name,
        "duration": 2.6,
        "refit_seconds": 11.2,
    }
    booster = {
        "model": "xgboost",
        "duration": 2.6,
        "refit_seconds": 11.2,
    }

    assert fedot_exec._secondary_refit_budget_estimate(
        forest, measured_primary_scale=13.8, fold_count=1
    ) == pytest.approx(22.4)
    assert fedot_exec._secondary_refit_budget_estimate(
        booster, measured_primary_scale=13.8, fold_count=1
    ) == pytest.approx(35.88)


@pytest.mark.parametrize(
    ("scale", "fold_count"),
    [(-0.1, 1), (1.0, 0)],
)
def test_secondary_refit_estimate_rejects_invalid_inputs(scale, fold_count):
    with pytest.raises(ValueError, match="Secondary refit estimate"):
        fedot_exec._secondary_refit_budget_estimate(
            {"model": "rf", "duration": 1.0, "refit_seconds": 1.0},
            measured_primary_scale=scale,
            fold_count=fold_count,
        )


@pytest.mark.parametrize(
    (
        "requested",
        "configured_min_rows",
        "train_rows",
        "fold_count",
        "metric",
        "expected",
    ),
    [
        (True, None, 5_000, 1, "auc", True),
        (True, None, 500, 3, "logloss", True),
        (True, None, 500, 1, "logloss", False),
        (True, None, 500, 3, "auc", False),
        (True, 5_000, 500, 3, "logloss", False),
        (True, 1, 500, 1, "auc", True),
        (False, 1, 5_000, 3, "logloss", False),
    ],
)
def test_all_row_refit_uses_rows_or_robust_logloss_cv(
    requested,
    configured_min_rows,
    train_rows,
    fold_count,
    metric,
    expected,
):
    assert (
        fedot_exec._use_all_row_refit(
            requested,
            configured_min_rows,
            train_rows,
            fold_count,
            metric,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("configured", "fold_count", "adaptive_wide_pair", "primary", "expected"),
    [
        (False, 3, True, "logit", False),
        (True, 1, False, "xgboost", False),
        (True, 3, False, "xgboost", True),
        (None, 3, True, "logit", True),
        (None, 3, True, "xgboost", False),
        (None, 3, False, "logit", False),
    ],
)
def test_reuse_cv_secondary_is_explicit_or_adaptive_wide_only(
    configured, fold_count, adaptive_wide_pair, primary, expected
):
    assert (
        fedot_exec._reuse_cv_secondary_models(
            configured, fold_count, adaptive_wide_pair, primary
        )
        is expected
    )


def test_default_candidates_use_linear_diversity_only_for_feasible_wide_data():
    wide = np.zeros((10_000, 800))
    narrow = wide[:, :20]
    target = np.arange(10_000) % 7

    assert fedot_exec._adaptive_default_portfolio_candidates(wide, target) == [
        "scaled_logit",
        "xgboost",
    ]
    assert fedot_exec._adaptive_logit_c(
        wide, target, ["scaled_logit", "xgboost"]
    ) == pytest.approx(0.01)
    assert fedot_exec._adaptive_default_portfolio_candidates(narrow, target) == [
        "lgbm",
        "xgboost",
    ]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        wide[:2_499], target[:2_499]
    ) == ["lgbm", "xgboost", "auto"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        wide[:2_500], target[:2_500]
    ) == ["logit", "xgboost"]


def test_small_dense_kernel_candidate_has_structural_cost_and_density_guards():
    dense = np.broadcast_to(
        np.ones((1, 256), dtype=np.float32), (1_500, 256)
    )
    target = np.arange(1_500) % 10
    sparse_like_row = np.zeros((1, 256), dtype=np.float32)
    sparse_like_row[:, :63] = 1.0
    sparse_like = np.broadcast_to(sparse_like_row, dense.shape)

    assert fedot_exec._is_small_dense_kernel_multiclass(dense, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(dense, target) == [
        "scaled_svc"
    ]
    assert not fedot_exec._is_small_dense_kernel_multiclass(
        dense[:999], target[:999]
    )
    assert not fedot_exec._is_small_dense_kernel_multiclass(
        dense[:, :127], target
    )
    assert not fedot_exec._is_small_dense_kernel_multiclass(
        np.broadcast_to(dense[:1], (2_501, 256)), np.arange(2_501) % 10
    )
    assert not fedot_exec._is_small_dense_kernel_multiclass(sparse_like, target)
    assert not fedot_exec._is_small_dense_kernel_multiclass(
        sparse.csr_matrix(dense), target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        dense, target, configured="lgbm,xgboost"
    ) == ["lgbm", "xgboost"]


def test_small_narrow_kernel_candidate_has_structural_and_quality_guards():
    dense = np.broadcast_to(
        np.ones((1, 60), dtype=np.float32), (600, 60)
    )
    target = np.arange(600) % 6
    sparse_like_row = np.zeros((1, 60), dtype=np.float32)
    sparse_like_row[:, :14] = 1.0
    sparse_like = np.broadcast_to(sparse_like_row, dense.shape)
    rare_target = np.arange(600) % 5
    rare_target[:49] = 5

    assert fedot_exec._is_small_narrow_dense_kernel_multiclass(dense, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(dense, target) == [
        "lgbm",
        "xgboost",
        "scaled_svc",
        "scaled_svc_strong_half",
        "scaled_svc_strong_scale",
        "auto",
    ]
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        dense[:499], target[:499]
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        np.broadcast_to(dense[:1], (2_501, 60)), np.arange(2_501) % 6
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        dense[:, :15], target
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        np.broadcast_to(dense[:1, :1], (600, 128)), target
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        np.broadcast_to(dense[:1, :1], (2_500, 121)), np.arange(2_500) % 5
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        dense, np.arange(600) % 2
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        dense, np.arange(600) % 11
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        dense, rare_target
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        sparse_like, target
    )
    assert not fedot_exec._is_small_narrow_dense_kernel_multiclass(
        sparse.csr_matrix(dense), target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        dense, target, configured="lgbm,xgboost"
    ) == ["lgbm", "xgboost"]


def test_relational_hist_candidate_has_ordinal_geometry_and_cost_guards():
    row_count = 20_000
    row_index = np.arange(row_count)
    compact_integer = np.column_stack(
        [row_index % cardinality for cardinality in range(2, 8)]
    ).astype(np.uint8)
    target = row_index % 5

    assert fedot_exec._is_large_narrow_discrete_relational_candidate(
        compact_integer, target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        compact_integer, target
    ) == ["lgbm", "xgboost", "relational_hist_second_order"]
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        compact_integer[:19_999], target[:19_999]
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        np.broadcast_to(compact_integer[:1], (100_001, 6)),
        np.arange(100_001) % 5,
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        compact_integer[:, :3], target
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        np.broadcast_to(compact_integer[:, :1], (row_count, 10)), target
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        compact_integer, row_index % 2
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        compact_integer, row_index % 6
    )

    rare_target = row_index % 4
    rare_target[:999] = 4
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        compact_integer, rare_target
    )

    # Nine columns at the maximum row bound expand to 11.7 million cells,
    # immediately below the explicit 12-million-cell budget.
    maximum_bounded = np.broadcast_to(
        np.zeros((1, 9), dtype=np.uint16), (100_000, 9)
    )
    assert fedot_exec._is_large_narrow_discrete_relational_candidate(
        maximum_bounded, np.arange(100_000) % 5
    )


def test_relational_second_order_adaptive_guard_uses_complete_expansion_cost():
    jungle_rows = 40_000
    jungle_index = np.arange(jungle_rows)
    jungle_like = np.column_stack(
        [jungle_index % cardinality for cardinality in range(2, 8)]
    ).astype(np.uint8)
    jungle_target = jungle_index % 3

    assert fedot_exec._is_large_narrow_discrete_relational_candidate(
        jungle_like, jungle_target
    )
    assert fedot_exec._is_second_order_relational_expansion_feasible(jungle_like)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        jungle_like, jungle_target
    ) == ["lgbm", "xgboost", "relational_hist_second_order"]

    bng_rows = 50_000
    bng_index = np.arange(bng_rows)
    bng_like = np.column_stack(
        [bng_index % cardinality for cardinality in range(2, 11)]
    ).astype(np.uint8)
    bng_target = bng_index % 3

    assert fedot_exec._is_large_narrow_discrete_relational_candidate(
        bng_like, bng_target
    )
    assert not fedot_exec._is_second_order_relational_expansion_feasible(bng_like)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        bng_like, bng_target
    ) == ["lgbm", "xgboost", "relational_hist"]


def test_relational_hist_adaptive_guard_requires_compact_integer_ordering():
    row_count = 20_000
    row_index = np.arange(row_count)
    low_cardinality = np.column_stack(
        [row_index % 16 for _ in range(6)]
    ).astype(np.uint8)
    high_cardinality = np.column_stack(
        [row_index % 251 for _ in range(2)]
    ).astype(np.uint8)
    exactly_three_quarters_discrete = np.column_stack(
        (low_cardinality, high_cardinality)
    )
    target = row_index % 5

    assert fedot_exec._is_large_narrow_discrete_relational_candidate(
        exactly_three_quarters_discrete, target
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        np.column_stack((low_cardinality[:, :5], high_cardinality, high_cardinality[:, :1])),
        target,
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        exactly_three_quarters_discrete.astype(np.float32), target
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        exactly_three_quarters_discrete.astype(np.int32), target
    )
    categorical = pd.DataFrame(
        {
            f"feature_{column}": pd.Categorical(
                exactly_three_quarters_discrete[:, column]
            )
            for column in range(exactly_three_quarters_discrete.shape[1])
        }
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        categorical, target
    )
    assert not fedot_exec._is_large_narrow_discrete_relational_candidate(
        sparse.csr_matrix(exactly_three_quarters_discrete), target
    )

    # Explicit candidate lists describe controlled experiments and take priority
    # over every adaptive geometry rule.
    assert fedot_exec._adaptive_default_portfolio_candidates(
        categorical, target, configured="relational_hist"
    ) == ["relational_hist"]


def test_medium_numeric_regression_portfolio_has_resource_and_dtype_guards():
    target = np.linspace(0.0, 1.0, 10_000)
    numeric = np.broadcast_to(
        np.zeros((1, 8), dtype=np.float32), (10_000, 8)
    )

    assert fedot_exec._is_medium_numeric_regression_portfolio(numeric, target)
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        numeric, target
    ) == ["lgbmreg_direct", "catboostreg_direct"]
    assert not fedot_exec._is_medium_numeric_regression_portfolio(
        numeric[:9_999], target[:9_999]
    )
    assert not fedot_exec._is_medium_numeric_regression_portfolio(
        np.broadcast_to(numeric[:1], (30_001, 8)),
        np.linspace(0.0, 1.0, 30_001),
    )
    assert not fedot_exec._is_medium_numeric_regression_portfolio(
        numeric[:, :7], target
    )
    assert not fedot_exec._is_medium_numeric_regression_portfolio(
        np.broadcast_to(numeric[:, :1], (10_000, 65)), target
    )
    maximum_bounded = np.broadcast_to(
        np.zeros((1, 64), dtype=np.float32), (30_000, 64)
    )
    assert fedot_exec._is_medium_numeric_regression_portfolio(
        maximum_bounded, np.linspace(0.0, 1.0, 30_000)
    )
    mixed = pd.DataFrame(numeric)
    mixed[0] = pd.Categorical(np.arange(len(mixed)) % 4)
    assert not fedot_exec._is_medium_numeric_regression_portfolio(mixed, target)
    assert not fedot_exec._is_medium_numeric_regression_portfolio(
        sparse.csr_matrix(numeric), target
    )
    nonfinite_target = target.copy()
    nonfinite_target[0] = np.nan
    assert not fedot_exec._is_medium_numeric_regression_portfolio(
        numeric, nonfinite_target
    )


def test_censored_ordinal_regression_portfolio_has_structural_guards():
    row_count = 1_500
    numeric = np.broadcast_to(
        np.zeros((1, 3), dtype=np.float32), (row_count, 3)
    )
    target = np.arange(row_count) % 12
    target[:400] = 0

    assert fedot_exec._is_censored_ordinal_regression_portfolio(numeric, target)
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        numeric, target
    ) == [
        "ordinal_lgbm31_direct",
        "ordinal_lgbm63_direct",
        "ordinal_xgboost_direct",
    ]
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        numeric[:-1], target[:-1]
    )
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        numeric[:, :2], target
    )
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        np.broadcast_to(numeric[:, :1], (row_count, 65)), target
    )
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        numeric, np.arange(row_count) % 2
    )
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        numeric, np.arange(row_count) % 33
    )
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        numeric, np.arange(row_count) % 12
    )
    mixed = pd.DataFrame(numeric)
    mixed[0] = pd.Categorical(np.arange(row_count) % 4)
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(mixed, target)
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        sparse.csr_matrix(numeric), target
    )
    nonfinite_target = target.astype(float)
    nonfinite_target[0] = np.nan
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        numeric, nonfinite_target
    )
    over_limit_rows = 30_001
    over_limit_target = np.arange(over_limit_rows) % 12
    over_limit_target[:8_000] = 0
    assert not fedot_exec._is_censored_ordinal_regression_portfolio(
        np.broadcast_to(numeric[:1], (over_limit_rows, 3)), over_limit_target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        np.zeros((20, 2)),
        np.arange(20),
        configured="ordinal_lgbm,ordinal_lgbm63,ordinal_xgb",
    ) == [
        "ordinal_lgbm31_direct",
        "ordinal_lgbm63_direct",
        "ordinal_xgboost_direct",
    ]


def test_ordered_ordinal_expansion_requires_compact_integer_histogram_geometry():
    row_count = 1_500
    row_index = np.arange(row_count)
    target = row_index % 12
    target[:400] = 0
    compact_bins = pd.DataFrame(
        {
            f"bin_{column}": (row_index + column).astype(np.uint8)
            for column in range(16)
        }
    )

    assert fedot_exec._is_ordered_histogram_ordinal_regression_portfolio(
        compact_bins, target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        compact_bins, target
    ) == [
        "ordinal_lgbm31_direct",
        "ordinal_ordered_lgbm_direct",
        "ordinal_lgbm63_direct",
        "ordinal_xgboost_direct",
    ]
    assert not fedot_exec._is_ordered_histogram_ordinal_regression_portfolio(
        compact_bins.iloc[:, :15], target
    )
    assert not fedot_exec._is_ordered_histogram_ordinal_regression_portfolio(
        compact_bins.astype(np.int32), target
    )
    assert not fedot_exec._is_ordered_histogram_ordinal_regression_portfolio(
        compact_bins.astype(np.float32), target
    )


def test_ordered_feature_expansion_preserves_rows_and_declared_order():
    features = np.array([[1, 2, 4], [3, 5, 8]], dtype=np.uint8)

    expanded = fedot_exec._ordered_feature_expansion(features)

    np.testing.assert_allclose(
        expanded,
        [
            [1, 2, 4, 1, 3, 7, 7, 6, 4, 0, 1, 2],
            [3, 5, 8, 3, 8, 16, 16, 13, 8, 0, 2, 3],
        ],
    )


def test_cumulative_probability_regressor_respects_ordered_target_range():
    features = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    target = np.where(features[:, 0] < -0.5, 0.0, 1.0)
    target[features[:, 0] > 0.5] = 3.0
    estimator = fedot_exec._CumulativeProbabilityRegressor(
        fedot_exec.LogisticRegression(max_iter=200)
    ).fit(features, target)

    predictions = estimator.predict(features)

    assert len(estimator.classifiers_) == 2
    assert predictions.shape == (len(features),)
    assert np.isfinite(predictions).all()
    assert predictions.min() >= target.min()
    assert predictions.max() <= target.max()


def test_medium_mostly_numeric_regression_requires_a_tiny_numeric_category_fringe():
    row_count = 10_000
    numeric_columns = {
        f"numeric_{column}": np.zeros(row_count, dtype=np.float32)
        for column in range(20)
    }
    mostly_numeric = pd.DataFrame(numeric_columns)
    mostly_numeric["numeric_category"] = pd.Categorical(
        (np.arange(row_count) % 70).astype(str)
    )
    target = np.linspace(0.0, 1.0, row_count)

    assert fedot_exec._is_medium_mostly_numeric_regression_portfolio(
        mostly_numeric, target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        mostly_numeric, target
    ) == ["lgbmreg_direct", "catboostreg_direct"]
    assert not fedot_exec._is_medium_numeric_regression_portfolio(
        mostly_numeric, target
    )
    assert not fedot_exec._is_medium_mostly_numeric_regression_portfolio(
        mostly_numeric.iloc[:9_999], target[:9_999]
    )

    too_many_categories = mostly_numeric.copy()
    too_many_categories["second_category"] = pd.Categorical(
        (np.arange(row_count) % 4).astype(str)
    )
    assert not fedot_exec._is_medium_mostly_numeric_regression_portfolio(
        too_many_categories, target
    )
    nominal = mostly_numeric.copy()
    nominal["numeric_category"] = pd.Categorical(
        np.where(np.arange(row_count) % 2, "left", "right")
    )
    assert not fedot_exec._is_medium_mostly_numeric_regression_portfolio(
        nominal, target
    )
    high_cardinality = mostly_numeric.copy()
    high_cardinality["numeric_category"] = pd.Categorical(
        (np.arange(row_count) % 129).astype(str)
    )
    assert not fedot_exec._is_medium_mostly_numeric_regression_portfolio(
        high_cardinality, target
    )
    nonfinite_target = target.copy()
    nonfinite_target[0] = np.inf
    assert not fedot_exec._is_medium_mostly_numeric_regression_portfolio(
        mostly_numeric, nonfinite_target
    )
    assert not fedot_exec._is_medium_mostly_numeric_regression_portfolio(
        sparse.csr_matrix(np.zeros((row_count, 21))), target
    )


def test_small_wide_regression_portfolio_bounds_dense_and_one_hot_work():
    numeric = np.zeros((200, 64), dtype=np.float32)
    numeric_target = np.linspace(0.0, 1.0, len(numeric))

    assert fedot_exec._is_small_wide_regression_portfolio(
        numeric, numeric_target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        numeric, numeric_target
    ) == ["lgbmreg_direct", "extra_treesreg_direct"]

    row_count = 1_000
    mixed = pd.DataFrame(
        {
            **{
                f"numeric_{column}": np.zeros(row_count, dtype=np.float32)
                for column in range(144)
            },
            "instance": pd.Categorical(np.arange(row_count) % 218),
            "algorithm": pd.Categorical(np.arange(row_count) % 5),
            "status": pd.Categorical(np.arange(row_count) % 2),
        }
    )
    mixed_target = np.linspace(0.0, 1.0, row_count)
    assert fedot_exec._is_small_wide_regression_portfolio(mixed, mixed_target)
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        mixed, mixed_target
    ) == ["lgbmreg_onehot_direct"]

    assert not fedot_exec._is_small_wide_regression_portfolio(
        numeric[:199], numeric_target[:199]
    )
    assert not fedot_exec._is_small_wide_regression_portfolio(
        numeric[:, :63], numeric_target
    )
    assert not fedot_exec._is_small_wide_regression_portfolio(
        np.zeros((200, 257), dtype=np.float32), numeric_target
    )
    assert not fedot_exec._is_small_wide_regression_portfolio(
        sparse.csr_matrix(numeric), numeric_target
    )
    too_many_categories = mixed.copy()
    for column in range(6):
        too_many_categories[f"extra_category_{column}"] = pd.Categorical(
            np.arange(row_count) % 2
        )
    assert not fedot_exec._is_small_wide_regression_portfolio(
        too_many_categories, mixed_target
    )
    excessive_one_hot = pd.DataFrame(
        {
            **{
                f"numeric_{column}": np.zeros(row_count, dtype=np.float32)
                for column in range(63)
            },
            "category": pd.Categorical(np.arange(row_count) % 513),
        }
    )
    assert not fedot_exec._is_small_wide_regression_portfolio(
        excessive_one_hot, mixed_target
    )


def test_small_wide_validation_requires_a_large_normalized_signal():
    truth = np.linspace(-1.0, 1.0, 100)
    strong = {
        "truth": truth,
        "predictions": truth + 0.01,
    }
    weak = {
        "truth": truth,
        "predictions": np.zeros_like(truth),
    }

    assert fedot_exec._best_normalized_validation_rmse([strong]) < 0.02
    assert fedot_exec._best_normalized_validation_rmse([weak]) > 0.9
    assert fedot_exec._best_normalized_validation_rmse([]) == float("inf")
    with pytest.raises(ValueError, match="share validation truth"):
        fedot_exec._best_normalized_validation_rmse(
            [strong, {"truth": truth[::-1], "predictions": truth}]
        )


def test_small_classic_regression_portfolio_has_geometry_and_dtype_guards():
    target = np.linspace(0.0, 1.0, 350)
    numeric = np.zeros((350, 5), dtype=np.float32)

    assert fedot_exec._is_small_classic_regression_portfolio(numeric, target)
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        numeric, target
    ) == [
        "scaled_svrreg_direct",
        "xgboostreg_direct",
        "catboostreg_direct",
    ]
    assert not fedot_exec._is_small_classic_regression_portfolio(
        numeric[:349], target[:349]
    )
    assert not fedot_exec._is_small_classic_regression_portfolio(
        numeric[:, :4], target
    )
    assert not fedot_exec._is_small_classic_regression_portfolio(
        np.zeros((1_001, 5), dtype=np.float32), np.linspace(0.0, 1.0, 1_001)
    )
    assert not fedot_exec._is_small_classic_regression_portfolio(
        sparse.csr_matrix(numeric), target
    )

    mostly_numeric = pd.DataFrame(numeric)
    mostly_numeric[0] = pd.Categorical(np.arange(len(target)) % 4)
    assert fedot_exec._is_small_classic_regression_portfolio(
        mostly_numeric, target
    )
    too_categorical = mostly_numeric.copy()
    too_categorical[1] = pd.Categorical(np.arange(len(target)) % 4)
    assert not fedot_exec._is_small_classic_regression_portfolio(
        too_categorical, target
    )


def test_large_compact_mixed_regression_has_resource_and_cardinality_guards():
    row_count = 30_000
    target = np.linspace(0.0, 1.0, row_count)
    mixed = pd.DataFrame(
        {
            **{
                f"numeric_{column}": np.zeros(row_count, dtype=np.float32)
                for column in range(4)
            },
            "category": pd.Categorical(np.arange(row_count) % 4),
        }
    )

    assert fedot_exec._is_large_compact_mixed_regression_portfolio(
        mixed, target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        mixed, target
    ) == [
        "lgbmreg_compact_direct",
        "xgboostreg_direct",
        "catboostreg_direct",
    ]
    assert not fedot_exec._is_large_compact_mixed_regression_portfolio(
        mixed.iloc[:29_999], target[:29_999]
    )
    assert not fedot_exec._is_large_compact_mixed_regression_portfolio(
        mixed.iloc[:, :4], target
    )
    assert not fedot_exec._is_large_compact_mixed_regression_portfolio(
        sparse.csr_matrix(np.zeros((row_count, 5))), target
    )
    numeric = mixed.copy()
    numeric["category"] = 0.0
    assert not fedot_exec._is_large_compact_mixed_regression_portfolio(
        numeric, target
    )
    high_cardinality = mixed.copy()
    high_cardinality["category"] = pd.Categorical(
        np.arange(row_count) % 10_001
    )
    assert not fedot_exec._is_large_compact_mixed_regression_portfolio(
        high_cardinality, target
    )


def test_very_large_compact_mixed_regression_requires_leaf_support():
    row_count = 255_000
    target = np.linspace(0.0, 1.0, row_count)
    mixed = pd.DataFrame(
        {
            **{
                f"numeric_{column}": np.zeros(row_count, dtype=np.float32)
                for column in range(7)
            },
            "category": pd.Categorical(np.arange(row_count) % 32),
        }
    )

    assert fedot_exec._is_very_large_compact_mixed_regression_portfolio(
        mixed, target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        mixed, target
    ) == [
        "lgbmreg_large_leaf127_direct",
        "lgbmreg_large_leaf255_direct",
    ]
    assert not fedot_exec._is_very_large_compact_mixed_regression_portfolio(
        mixed.iloc[:254_999], target[:254_999]
    )
    assert not fedot_exec._is_very_large_compact_mixed_regression_portfolio(
        mixed.iloc[:, :7], target
    )

    high_cardinality = mixed.copy()
    high_cardinality["category"] = pd.Categorical(
        np.arange(row_count) % 1_025
    )
    assert not fedot_exec._is_very_large_compact_mixed_regression_portfolio(
        high_cardinality, target
    )


def test_large_dense_low_cardinality_regression_has_structural_bounds():
    row_count = 250_000
    numeric = SimpleNamespace(
        shape=(row_count, 64),
        dtypes=[np.dtype("float32")] * 64,
    )
    target = np.arange(row_count, dtype=float) % 89

    assert fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        numeric, target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        numeric, target
    ) == ["lgbmreg_large_dense_direct"]

    too_few_rows = SimpleNamespace(
        shape=(row_count - 1, 64), dtypes=numeric.dtypes
    )
    assert not fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        too_few_rows, target[:-1]
    )
    too_narrow = SimpleNamespace(
        shape=(row_count, 63), dtypes=[np.dtype("float32")] * 63
    )
    assert not fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        too_narrow, target
    )
    too_many_cells = SimpleNamespace(
        shape=(600_000, 101), dtypes=[np.dtype("float32")] * 101
    )
    assert not fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        too_many_cells, np.arange(600_000, dtype=float) % 89
    )
    mixed = SimpleNamespace(
        shape=numeric.shape,
        dtypes=[np.dtype("float32")] * 63 + [np.dtype("object")],
    )
    assert not fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        mixed, target
    )
    assert not fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        sparse.csr_matrix((row_count, 64), dtype=np.float32), target
    )
    assert not fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        numeric, np.arange(row_count, dtype=float) % 31
    )
    assert not fedot_exec._is_large_dense_low_cardinality_regression_portfolio(
        numeric, np.linspace(0.0, 1.0, row_count)
    )


def test_large_grouped_sequence_regression_has_structural_bounds():
    row_count = 250_000
    columns = [
        f"history_{group}_{lag}" for group in range(8) for lag in range(7)
    ]
    numeric = SimpleNamespace(
        shape=(row_count, len(columns)),
        columns=columns,
        dtypes=[np.dtype("float32")] * len(columns),
    )
    target = np.expm1(np.linspace(0.0, 15.0, row_count))

    assert fedot_exec._is_large_grouped_sequence_regression_portfolio(
        numeric, target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        numeric, target
    ) == [
        "grouped_trend_lgbmreg_leaf511_direct",
        "grouped_trend_lgbmreg_leaf255_regularized_direct",
    ]
    assert not fedot_exec._is_large_grouped_sequence_regression_portfolio(
        numeric, np.linspace(0.0, 1.0, row_count)
    )
    assert not fedot_exec._is_large_grouped_sequence_regression_portfolio(
        numeric, target - target.max()
    )

    broken_columns = [f"feature_{column}" for column in range(len(columns))]
    broken = SimpleNamespace(
        shape=numeric.shape,
        columns=broken_columns,
        dtypes=numeric.dtypes,
    )
    assert not fedot_exec._is_large_grouped_sequence_regression_portfolio(
        broken, target
    )


def test_grouped_sequence_expansion_adds_only_train_independent_trends():
    features = pd.DataFrame(
        {
            "alpha_0": [0.0, 3.0],
            "alpha_1": [1.0, 3.0],
            "alpha_2": [4.0, 3.0],
            "alpha_3": [9.0, 3.0],
            "beta_0": [2.0, 1.0],
            "beta_1": [4.0, 2.0],
            "beta_2": [6.0, 3.0],
            "beta_3": [8.0, 4.0],
        }
    )

    expanded = fedot_exec._grouped_sequence_trend_expansion(features)

    assert expanded.shape == (2, 14)
    assert expanded[:, :8] == pytest.approx(features.to_numpy())
    assert expanded[0, 8:] == pytest.approx([5.0, 9.0, 3.0, 2.0, 6.0, 2.0])
    assert expanded[1, 8:] == pytest.approx([0.0, 0.0, 0.0, 1.0, 3.0, 1.0])


def test_train_selected_lgbm_caps_horizon_selection_and_refits_all_rows(
    monkeypatch,
):
    fitted_rows = []

    class FakeLGBM:
        def __init__(self, **params):
            self.params = params

        def fit(self, features, target, **kwargs):
            del target, kwargs
            fitted_rows.append((len(features), self.params["n_estimators"]))
            self.best_iteration_ = 37
            return self

        def predict(self, features):
            return np.zeros(len(features))

    monkeypatch.setattr(fedot_exec, "LGBMRegressor", FakeLGBM)
    model = fedot_exec._TrainSelectedLGBMRegressor(
        selector_max_rows=1_000,
        refit_all_rows=True,
        random_state=7,
    )

    features = np.zeros((2_000, 3), dtype=np.float32)
    model.fit(features, np.arange(len(features), dtype=float))

    assert fitted_rows == [(800, 1_000), (2_000, 37)]
    assert model.selected_n_estimators_ == 37


def test_large_dense_regression_caps_selector_and_refits_all_rows(
    monkeypatch, tmp_path
):
    fitted_rows = []

    class IdentityRegressor:
        def __init__(self, selector):
            self.selector = selector

        def fit(self, features, target):
            del target
            fitted_rows.append((len(features), self.selector))
            return self

        def predict(self, features):
            return np.asarray(features)[:, 0]

    row_count = 250_000
    features = np.arange(row_count, dtype=np.float32).reshape(-1, 1)
    target = features[:, 0].astype(float)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:20], y=target[:20]),
    )
    config = SimpleNamespace(
        framework_params={"_portfolio": True},
        metric="rmse",
        seed=42,
        cores=4,
        max_runtime_seconds=180,
        output_predictions_file=str(tmp_path / "predictions.csv"),
    )

    monkeypatch.setattr(
        fedot_exec,
        "_adaptive_regression_portfolio_candidates",
        lambda *args, **kwargs: ["lgbmreg_large_dense_direct"],
    )
    monkeypatch.setattr(
        fedot_exec,
        "_is_large_dense_low_cardinality_regression_portfolio",
        lambda *args: True,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_make_regression_portfolio_estimator",
        lambda *args, selector=False, **kwargs: IdentityRegressor(selector),
    )
    monkeypatch.setattr(fedot_exec, "save_artifacts", lambda *args: None)
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec._run_regression_portfolio(dataset, config)

    assert fitted_rows == [(80_000, True), (row_count, False)]
    assert output["predictions"] == pytest.approx(target[:20])


def test_large_high_categorical_regression_has_structural_bounds():
    row_count = 100_000
    features = pd.DataFrame(
        {
            **{
                f"numeric_{column}": np.zeros(row_count, dtype=np.float32)
                for column in range(8)
            },
            **{
                f"category_{column}": pd.Categorical(
                    np.arange(row_count, dtype=np.int32) % 4
                )
                for column in range(88)
            },
        }
    )
    coordinate = np.linspace(0.0, 1.0, row_count)
    target = coordinate**10 + coordinate * 1e-9

    assert fedot_exec._is_large_high_categorical_regression_portfolio(
        features, target
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        features, target
    ) == ["catboostreg_direct", "target_frequency_lgbmreg_direct"]
    assert not fedot_exec._is_large_high_categorical_regression_portfolio(
        features.iloc[:99_999], target[:99_999]
    )
    assert not fedot_exec._is_large_high_categorical_regression_portfolio(
        features.iloc[:, :95], target
    )
    assert not fedot_exec._is_large_high_categorical_regression_portfolio(
        sparse.csr_matrix((row_count, 96), dtype=np.float32), target
    )
    assert not fedot_exec._is_large_high_categorical_regression_portfolio(
        features, np.linspace(0.0, 1.0, row_count)
    )
    high_cardinality = features.copy()
    high_cardinality["category_0"] = pd.Categorical(
        np.arange(row_count, dtype=np.int32) % 513
    )
    assert not fedot_exec._is_large_high_categorical_regression_portfolio(
        high_cardinality, target
    )


def test_regression_target_frequency_encoding_is_cross_fitted(monkeypatch):
    captured = {}

    class RecordingRegressor:
        def __init__(self, **parameters):
            captured["parameters"] = parameters

        def fit(self, features, target):
            captured["features"] = np.asarray(features).copy()
            captured["target"] = np.asarray(target).copy()
            return self

        def predict(self, features):
            return np.full(len(features), np.mean(captured["target"]))

    monkeypatch.setattr(fedot_exec, "LGBMRegressor", RecordingRegressor)
    row_count = 90
    features = pd.DataFrame(
        {
            "numeric": np.arange(row_count, dtype=float),
            "category": pd.Categorical([f"id-{index}" for index in range(row_count)]),
        }
    )
    target = np.arange(row_count, dtype=float)
    estimator = fedot_exec._CrossFittedTargetFrequencyRegressor(
        smoothing=10.0,
        n_estimators=50,
        n_jobs=2,
        random_state=7,
    ).fit(features, target)

    assert captured["features"].shape == (row_count, 3)
    assert captured["features"][:, 1] == pytest.approx(np.mean(target))
    assert captured["features"][:, 2] == pytest.approx(1.0 / row_count)
    assert captured["parameters"]["n_estimators"] == 50
    predictions = estimator.predict(
        pd.DataFrame(
            {
                "numeric": [1.0, np.nan],
                "category": pd.Categorical(
                    ["id-1", "unseen"], categories=list(features["category"].cat.categories) + ["unseen"]
                ),
            }
        )
    )
    assert predictions.shape == (2,)


def test_explicit_regression_candidates_bypass_adaptive_small_wide_guards(
    monkeypatch, tmp_path
):
    class ConstantRegressor:
        def fit(self, features, target):
            del features
            self.constant_ = float(np.mean(target))
            return self

        def predict(self, features):
            return np.full(len(features), self.constant_)

    features = np.zeros((200, 64), dtype=np.float32)
    target = np.linspace(0.0, 1.0, len(features))
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:20], y=target[:20]),
    )
    config = SimpleNamespace(
        framework_params={"_portfolio_candidates": "lgbmreg_direct"},
        metric="rmse",
        seed=42,
        cores=2,
        output_predictions_file=str(tmp_path / "predictions.csv"),
    )

    monkeypatch.setattr(
        fedot_exec,
        "_make_regression_portfolio_estimator",
        lambda *args, **kwargs: ConstantRegressor(),
    )
    monkeypatch.setattr(fedot_exec, "save_artifacts", lambda *args: None)
    monkeypatch.setattr(
        fedot_exec,
        "_best_normalized_validation_rmse",
        lambda *args: pytest.fail("explicit candidates entered the adaptive gate"),
    )
    monkeypatch.setattr(
        fedot_exec,
        "_clip_to_observed_target_range",
        lambda *args: pytest.fail("explicit candidates entered adaptive clipping"),
    )
    monkeypatch.setattr(
        fedot_exec,
        "_clip_to_observed_target_quantiles",
        lambda *args, **kwargs: pytest.fail(
            "explicit candidates entered adaptive quantile clipping"
        ),
    )

    benchmark_result = fedot_exec._run_regression_portfolio(dataset, config)

    assert benchmark_result["models_count"] == 1
    assert benchmark_result["predictions"].shape == (20,)

    assert fedot_exec._adaptive_regression_portfolio_candidates(
        np.zeros((20, 2)), np.arange(20), configured="scaled_svr"
    ) == ["scaled_svrreg_direct"]


def test_regression_target_range_projection_uses_reference_labels_only():
    predictions = np.array([-2.0, 0.5, 4.0])

    clipped = fedot_exec._clip_to_observed_target_range(
        predictions, np.array([-1.0, 0.0, 2.0])
    )

    assert clipped == pytest.approx([-1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="finite reference"):
        fedot_exec._clip_to_observed_target_range(predictions, [])
    with pytest.raises(ValueError, match="finite reference"):
        fedot_exec._clip_to_observed_target_range(predictions, [0.0, np.nan])


def test_minimum_inflated_skewed_regression_target_has_structural_guards():
    target = np.r_[np.zeros(80), np.linspace(1.0, 2.0, 319), 100.0]

    assert fedot_exec._is_minimum_inflated_skewed_regression_target(target)
    assert not fedot_exec._is_minimum_inflated_skewed_regression_target(
        target[:399]
    )
    assert not fedot_exec._is_minimum_inflated_skewed_regression_target(
        np.r_[np.zeros(59), np.linspace(1.0, 2.0, 340), 100.0]
    )
    assert not fedot_exec._is_minimum_inflated_skewed_regression_target(
        np.tile(np.arange(32, dtype=float), 13)[:400]
    )
    nonfinite = target.copy()
    nonfinite[0] = np.nan
    assert not fedot_exec._is_minimum_inflated_skewed_regression_target(nonfinite)


def test_regression_quantile_projection_uses_reference_labels_only():
    reference = np.r_[np.zeros(20), np.arange(1.0, 80.0), 1_000.0]
    lower, upper = np.quantile(reference, [0.01, 0.99])

    clipped = fedot_exec._clip_to_observed_target_quantiles(
        [-5.0, 10.0, 2_000.0], reference, 0.01, 0.99
    )

    assert clipped == pytest.approx([lower, 10.0, upper])
    with pytest.raises(ValueError, match="finite reference"):
        fedot_exec._clip_to_observed_target_quantiles(
            [0.0], [0.0, np.inf], 0.01, 0.99
        )
    with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
        fedot_exec._clip_to_observed_target_quantiles(
            [0.0], reference, 0.99, 0.01
        )


def test_ordinal_regression_estimators_have_bounded_fixed_parameters():
    features = np.zeros((20, 3), dtype=np.float32)
    lgbm31 = fedot_exec._make_regression_portfolio_estimator(
        "ordinal_lgbm31_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    lgbm63 = fedot_exec._make_regression_portfolio_estimator(
        "ordinal_lgbm63_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    xgboost = fedot_exec._make_regression_portfolio_estimator(
        "ordinal_xgboost_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    ordered_pipeline = fedot_exec._make_regression_portfolio_estimator(
        "ordinal_ordered_lgbm_direct", features, seed=7, n_jobs=2
    )
    ordered_lgbm = ordered_pipeline.steps[-1][1]
    one_hot_lgbm_pipeline = fedot_exec._make_regression_portfolio_estimator(
        "lgbmreg_onehot_direct",
        pd.DataFrame(
            {
                "numeric": np.zeros(20),
                "category": pd.Categorical(np.arange(20) % 2),
            }
        ),
        seed=7,
        n_jobs=2,
    )
    one_hot_lgbm = one_hot_lgbm_pipeline.steps[-1][1]
    compact_lgbm = fedot_exec._make_regression_portfolio_estimator(
        "lgbmreg_compact_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    large_lgbm127 = fedot_exec._make_regression_portfolio_estimator(
        "lgbmreg_large_leaf127_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    large_lgbm255 = fedot_exec._make_regression_portfolio_estimator(
        "lgbmreg_large_leaf255_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    large_dense_lgbm = fedot_exec._make_regression_portfolio_estimator(
        "lgbmreg_large_dense_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    large_dense_selector = fedot_exec._make_regression_portfolio_estimator(
        "lgbmreg_large_dense_direct",
        features,
        seed=7,
        n_jobs=2,
        selector=True,
    ).steps[-1][1]
    target_frequency_regressor = fedot_exec._make_regression_portfolio_estimator(
        "target_frequency_lgbmreg_direct",
        pd.DataFrame(
            {
                "numeric": np.zeros(20),
                "category": pd.Categorical(np.arange(20) % 2),
            }
        ),
        seed=7,
        n_jobs=2,
    )
    extra_trees = fedot_exec._make_regression_portfolio_estimator(
        "extra_treesreg_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]
    scaled_svr = fedot_exec._make_regression_portfolio_estimator(
        "scaled_svrreg_direct", features, seed=7, n_jobs=2
    ).steps[-1][1]

    assert lgbm31.classifier.get_params()["n_estimators"] == 300
    assert lgbm31.classifier.get_params()["num_leaves"] == 31
    assert lgbm63.classifier.get_params()["num_leaves"] == 63
    assert xgboost.classifier.get_params()["n_estimators"] == 300
    assert xgboost.classifier.get_params()["max_depth"] == 6
    assert ordered_lgbm.classifier.get_params()["n_estimators"] == 300
    assert ordered_lgbm.classifier.get_params()["num_leaves"] == 31
    assert isinstance(ordered_pipeline.steps[-2][1], fedot_exec.FunctionTransformer)
    assert one_hot_lgbm.get_params()["n_estimators"] == 500
    assert one_hot_lgbm.get_params()["num_leaves"] == 31
    assert compact_lgbm.get_params()["n_estimators"] == 500
    assert compact_lgbm.get_params()["num_leaves"] == 31
    assert large_lgbm127.get_params()["n_estimators"] == 2_000
    assert large_lgbm127.get_params()["num_leaves"] == 127
    assert large_lgbm255.get_params()["n_estimators"] == 2_000
    assert large_lgbm255.get_params()["num_leaves"] == 255
    assert large_dense_lgbm.get_params()["n_estimators"] == 2_000
    assert large_dense_lgbm.get_params()["num_leaves"] == 255
    assert large_dense_lgbm.get_params()["min_child_samples"] == 100
    assert large_dense_selector.get_params()["n_estimators"] == 1_000
    assert isinstance(
        target_frequency_regressor,
        fedot_exec._CrossFittedTargetFrequencyRegressor,
    )
    assert target_frequency_regressor.get_params()["smoothing"] == pytest.approx(10.0)
    categorical_pipeline = one_hot_lgbm_pipeline.steps[0][1].transformers[1][1]
    assert isinstance(categorical_pipeline.steps[-1][1], fedot_exec.OneHotEncoder)
    assert extra_trees.get_params()["n_estimators"] == 500
    assert isinstance(scaled_svr, fedot_exec.TransformedTargetRegressor)
    assert scaled_svr.regressor.steps[-1][1].get_params()["C"] == 10.0
    assert scaled_svr.regressor.steps[-1][1].get_params()["epsilon"] == 0.05
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        features,
        np.arange(len(features)),
        configured="lgbmreg_large_leaf127,lgbmreg_large_leaf255",
    ) == [
        "lgbmreg_large_leaf127_direct",
        "lgbmreg_large_leaf255_direct",
    ]
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        features,
        np.arange(len(features)),
        configured="ordinal_ordered_lgbm",
    ) == ["ordinal_ordered_lgbm_direct"]
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        features,
        np.arange(len(features)),
        configured="lgbmreg_large_dense",
    ) == ["lgbmreg_large_dense_direct"]
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        features,
        np.arange(len(features)),
        configured="target_frequency_lgbmreg",
    ) == ["target_frequency_lgbmreg_direct"]


def test_small_mixed_regression_portfolio_has_encoding_cost_guards():
    row_count = 400
    mixed = pd.DataFrame(
        {
            "category_1": pd.Categorical(np.arange(row_count) % 4),
            "category_2": pd.Categorical(np.arange(row_count) % 3),
            "numeric_1": np.arange(row_count, dtype=np.float32),
            "numeric_2": np.zeros(row_count, dtype=np.float32),
        }
    )
    target = np.linspace(0.0, 1.0, row_count)

    assert fedot_exec._is_small_mixed_regression_portfolio(mixed, target)
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        mixed, target
    ) == [
        "lgbmreg_direct",
        "xgboostreg_direct",
        "catboostreg_direct",
        "rfr_direct",
    ]
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        mixed.iloc[:399], target[:399]
    )
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        pd.concat([mixed] * 7, ignore_index=True).iloc[:2_501],
        np.linspace(0.0, 1.0, 2_501),
    )
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        mixed.iloc[:, :3], target
    )
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        pd.DataFrame(np.zeros((row_count, 129), dtype=np.float32)).assign(
            category=pd.Categorical(np.arange(row_count) % 4)
        ),
        target,
    )
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        mixed.drop(columns=["category_1", "category_2"]), target
    )
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        sparse.csr_matrix(np.zeros((row_count, 4))), target
    )

    high_cardinality = pd.DataFrame(
        {
            "category_1": pd.Categorical(np.arange(2_500) % 2_049),
            "category_2": pd.Categorical(np.arange(2_500) % 4),
            "numeric_1": np.arange(2_500, dtype=np.float32),
            "numeric_2": np.zeros(2_500, dtype=np.float32),
        }
    )
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        high_cardinality, np.linspace(0.0, 1.0, 2_500)
    )
    assert fedot_exec._adaptive_regression_portfolio_candidates(
        np.zeros((20, 2)),
        np.arange(20),
        configured="lgbm,catboostreg,rfr_direct",
    ) == ["lgbmreg_direct", "catboostreg_direct", "rfr_direct"]

    minority_categorical = pd.DataFrame(
        {
            **{
                f"category_{column}": pd.Categorical(np.arange(row_count) % 4)
                for column in range(3)
            },
            **{
                f"numeric_{column}": np.zeros(row_count, dtype=np.float32)
                for column in range(4)
            },
        }
    )
    assert not fedot_exec._is_small_mixed_regression_portfolio(
        minority_categorical, target
    )


def test_ridge_oof_portfolio_has_geometry_and_encoding_cost_guards():
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            **{f"numeric_{index}": rng.normal(size=500) for index in range(4)},
            **{
                f"category_{index}": pd.Series(
                    np.resize(["a", "b", "c"], 500), dtype="category"
                )
                for index in range(6)
            },
        }
    )
    target = rng.normal(size=500)

    assert fedot_exec._is_ridge_oof_regression_portfolio(features, target)
    assert fedot_exec._is_relaxed_small_mixed_ridge_portfolio(features, target)
    assert not fedot_exec._is_ridge_oof_regression_portfolio(
        features.to_numpy(), target
    )
    assert not fedot_exec._is_ridge_oof_regression_portfolio(
        pd.concat([features] * 6, ignore_index=True), np.resize(target, 3_000)
    )

    high_cardinality = features.copy()
    high_cardinality["category_0"] = [f"level_{index}" for index in range(500)]
    for index in range(1, 6):
        high_cardinality[f"category_{index}"] = [
            f"level_{index}_{row}" for row in range(500)
        ]
    assert not fedot_exec._is_ridge_oof_regression_portfolio(
        high_cardinality, target
    )


def test_ridge_regression_candidate_handles_mixed_missing_features():
    features = pd.DataFrame(
        {
            "numeric": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
            "category": ["a", "b", None, "a", "c", "b"],
        }
    )
    target = np.array([0.0, 1.0, 1.5, 3.0, 4.0, 5.0])
    estimator = fedot_exec._make_regression_portfolio_estimator(
        "ridge_10_direct",
        features,
        seed=42,
        n_jobs=1,
    )

    estimator.fit(features, target)
    predictions = estimator.predict(features)

    assert predictions.shape == target.shape
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize(
    ("ridge_offset", "expected_selected"),
    [(0.7, True), (0.98, False)],
)
def test_ridge_oof_expansion_requires_five_percent_gain(
    monkeypatch, ridge_offset, expected_selected
):
    class OffsetRegressor:
        def __init__(self, offset):
            self.offset = offset

        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.asarray(features["truth"], dtype=float) + self.offset

    def make_estimator(candidate, *args, **kwargs):
        del args, kwargs
        offset = ridge_offset if candidate.startswith("ridge_") else 1.0
        return OffsetRegressor(offset)

    features = pd.DataFrame(
        {"truth": np.linspace(-2.0, 2.0, 60), "other": np.arange(60)}
    )
    target = features["truth"].to_numpy()
    monkeypatch.setattr(
        fedot_exec, "_make_regression_portfolio_estimator", make_estimator
    )

    selected = fedot_exec._select_ridge_oof_expansion(
        ["baseline"],
        features,
        target,
        seed=42,
        n_jobs=1,
        max_seconds=10,
    )

    if expected_selected:
        assert selected is not None
        assert selected[0][0]["model"].startswith("ridge_")
    else:
        assert selected is None


def test_ridge_oof_runtime_is_added_without_mutating_timer(monkeypatch, tmp_path):
    class IdentityRegressor:
        def fit(self, features, target):
            del features, target
            return self

        def predict(self, features):
            return np.asarray(features)[:, 0]

    timer_durations = iter([3.0, 4.0])
    monotonic_times = iter([10.0, 12.0, 20.0, 21.0])

    class ReadOnlyTimer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._duration = next(timer_durations)

        @property
        def duration(self):
            return self._duration

    features = np.column_stack(
        (np.linspace(-1.0, 1.0, 100), np.ones(100, dtype=float))
    )
    target = features[:, 0]
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:10], y=target[:10]),
    )
    config = SimpleNamespace(
        framework_params={},
        metric="rmse",
        seed=42,
        cores=4,
        max_runtime_seconds=90,
        output_predictions_file=str(tmp_path / "predictions.csv"),
    )
    ridge_observation = {
        "model": "ridge_1_direct",
        "score": 0.0,
        "predictions": target,
        "truth": target,
        "selected_n_estimators": None,
    }

    monkeypatch.setattr(fedot_exec, "Timer", ReadOnlyTimer)
    monkeypatch.setattr(fedot_exec.time, "monotonic", lambda: next(monotonic_times))
    monkeypatch.setattr(
        fedot_exec,
        "_adaptive_regression_portfolio_candidates",
        lambda *args, **kwargs: ["ridge_1_direct"],
    )
    monkeypatch.setattr(
        fedot_exec, "_is_ridge_oof_regression_portfolio", lambda *args: True
    )
    monkeypatch.setattr(
        fedot_exec,
        "_select_ridge_oof_expansion",
        lambda *args, **kwargs: ([ridge_observation], np.array([1.0]), 0.0),
    )
    monkeypatch.setattr(
        fedot_exec,
        "_make_regression_portfolio_estimator",
        lambda *args, **kwargs: IdentityRegressor(),
    )
    monkeypatch.setattr(fedot_exec, "save_artifacts", lambda *args: None)

    benchmark_result = fedot_exec._run_regression_portfolio(dataset, config)

    assert benchmark_result["training_duration"] == pytest.approx(5.0)
    assert benchmark_result["predict_duration"] == pytest.approx(4.0)


def test_regression_strategy_requires_a_material_top_two_pair_gain():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    left = {
        "model": "left",
        "truth": truth,
        "predictions": np.array([0.0, 1.0, 2.0, 4.0]),
    }
    right = {
        "model": "right",
        "truth": truth,
        "predictions": np.array([0.0, 1.0, 2.0, 2.0]),
    }
    for contender in (left, right):
        contender["score"] = fedot_exec._selection_score(
            "rmse", truth, contender["predictions"], None
        )

    ensemble, weights, score = fedot_exec._select_regression_strategy(
        [left, right], metric="rmse", min_relative_pair_gain=0.002
    )
    conservative, conservative_weights, conservative_score = (
        fedot_exec._select_regression_strategy(
            [left, right], metric="rmse", min_relative_pair_gain=1.1
        )
    )

    assert [contender["model"] for contender in ensemble] == ["left", "right"]
    assert weights == pytest.approx([0.5, 0.5])
    assert score == pytest.approx(0.0)
    assert len(conservative) == 1
    assert conservative_weights == pytest.approx([1.0])
    assert conservative_score == pytest.approx(left["score"])


def test_ordered_regression_strategy_can_select_a_complementary_lower_ranked_pair():
    truth = np.zeros(2)
    best = {
        "model": "best",
        "truth": truth,
        "predictions": np.array([1.0, 1.0]),
    }
    second = {
        "model": "second",
        "truth": truth,
        "predictions": np.array([0.9, 1.2]),
    }
    complementary = {
        "model": "complementary",
        "truth": truth,
        "predictions": np.array([-1.2, -1.2]),
    }
    contenders = [best, second, complementary]
    for contender in contenders:
        contender["score"] = fedot_exec._selection_score(
            "rmse", truth, contender["predictions"], None
        )

    conservative, _, _ = fedot_exec._select_regression_strategy(contenders)
    ensemble, weights, score = fedot_exec._select_regression_strategy(
        contenders, consider_all_pairs=True
    )

    assert [contender["model"] for contender in conservative] == ["best"]
    assert [contender["model"] for contender in ensemble] == [
        "best",
        "complementary",
    ]
    assert weights == pytest.approx([0.5, 0.5])
    assert score == pytest.approx(-0.1)


def test_narrow_scaled_svc_requires_large_raw_oof_gain_for_singleton_dominance():
    alternatives = [
        {"model": "lgbm", "score": -0.55},
        {"model": "auto", "score": -0.52},
    ]
    dominant = {"model": "scaled_svc", "score": -0.42}
    uncertain = {"model": "scaled_svc", "score": -0.421}

    assert fedot_exec._prefer_dominant_adaptive_scaled_svc(
        alternatives + [dominant], enabled=True
    ) == [dominant]
    assert fedot_exec._prefer_dominant_adaptive_scaled_svc(
        alternatives + [uncertain], enabled=True
    ) == alternatives + [uncertain]
    assert fedot_exec._prefer_dominant_adaptive_scaled_svc(
        alternatives + [dominant], enabled=False
    ) == alternatives + [dominant]


def test_narrow_scaled_svc_variant_requires_material_raw_oof_gain():
    alternatives = [{"model": "lgbm", "score": -0.30}]
    baseline = {"model": "scaled_svc", "score": -0.42}
    marginal = {"model": "scaled_svc_strong_half", "score": -0.405}
    supported = {"model": "scaled_svc_strong_scale", "score": -0.39}
    contenders = alternatives + [baseline, marginal, supported]

    assert fedot_exec._select_supported_adaptive_scaled_svc_variant(
        contenders, enabled=True
    ) == alternatives + [supported]
    assert fedot_exec._select_supported_adaptive_scaled_svc_variant(
        alternatives + [baseline, marginal], enabled=True
    ) == alternatives + [baseline]
    assert fedot_exec._select_supported_adaptive_scaled_svc_variant(
        contenders, enabled=False
    ) == contenders
    with pytest.raises(ValueError, match="non-negative"):
        fedot_exec._select_supported_adaptive_scaled_svc_variant(
            contenders, enabled=True, minimum_logloss_gain=-0.01
        )

    dense = np.broadcast_to(
        np.ones((1, 60), dtype=np.float32), (600, 60)
    )
    target = np.arange(600) % 6
    dominant_contenders = [
        {"model": "lgbm", "score": -0.55},
        baseline,
        supported,
    ]
    assert fedot_exec._skip_adaptive_composed_auto_after_kernel_dominance(
        "auto", None, dense, target, dominant_contenders
    )
    assert not fedot_exec._skip_adaptive_composed_auto_after_kernel_dominance(
        "auto", "lgbm,xgboost", dense, target, dominant_contenders
    )
    assert not fedot_exec._skip_adaptive_composed_auto_after_kernel_dominance(
        "auto", None, dense, target, contenders
    )


def test_medium_dense_kernel_candidate_has_structural_cost_and_support_guards():
    dense = np.broadcast_to(
        np.ones((1, 600), dtype=np.float32), (8_000, 600)
    )
    target = np.arange(8_000) % 20
    sparse_like_row = np.zeros((1, 600), dtype=np.float32)
    sparse_like_row[:, :149] = 1.0
    sparse_like = np.broadcast_to(sparse_like_row, dense.shape)
    rare_target = np.arange(8_000) % 19
    rare_target[:99] = 19

    assert fedot_exec._is_medium_dense_kernel_multiclass(dense, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(dense, target) == [
        "scaled_svc"
    ]
    assert not fedot_exec._is_medium_dense_kernel_multiclass(
        dense[:4_999], target[:4_999]
    )
    assert not fedot_exec._is_medium_dense_kernel_multiclass(
        np.broadcast_to(dense[:1], (10_001, 600)), np.arange(10_001) % 20
    )
    assert not fedot_exec._is_medium_dense_kernel_multiclass(
        dense[:, :255], target
    )
    assert not fedot_exec._is_medium_dense_kernel_multiclass(
        np.broadcast_to(dense[:1, :1], (8_000, 751)), target
    )
    assert not fedot_exec._is_medium_dense_kernel_multiclass(
        dense, np.arange(8_000) % 9
    )
    assert not fedot_exec._is_medium_dense_kernel_multiclass(
        dense, np.arange(8_000) % 31
    )
    assert not fedot_exec._is_medium_dense_kernel_multiclass(dense, rare_target)
    assert not fedot_exec._is_medium_dense_kernel_multiclass(sparse_like, target)
    assert not fedot_exec._is_medium_dense_kernel_multiclass(
        sparse.csr_matrix(dense), target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        dense, target, configured="lgbm,xgboost"
    ) == ["lgbm", "xgboost"]


def test_extra_trees_candidate_is_cost_bounded_and_selected_by_oof():
    dense = np.broadcast_to(
        np.ones((1, 32), dtype=np.float32), (6_000, 32)
    )
    target = np.arange(6_000) % 6
    rare_target = target.copy()
    rare_target[:4] = 6
    sparse_like_row = np.zeros((1, 32), dtype=np.float32)
    sparse_like_row[:, :7] = 1.0
    sparse_like = np.broadcast_to(sparse_like_row, dense.shape)

    assert fedot_exec._is_bounded_numeric_extra_trees_candidate(dense, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        dense, target, metric="logloss"
    ) == [
        "lgbm",
        "xgboost",
        "extra_trees",
        "extra_trees_wide",
        "catboost",
    ]
    assert fedot_exec._adaptive_default_portfolio_candidates(dense, target) == [
        "lgbm",
        "xgboost",
        "extra_trees",
        "catboost",
    ]
    assert fedot_exec._is_train_gated_wide_extra_trees_candidate(dense, target)
    assert not fedot_exec._is_train_gated_wide_extra_trees_candidate(
        dense[:4_999], target[:4_999]
    )
    assert not fedot_exec._is_train_gated_wide_extra_trees_candidate(
        dense[:, :15], target
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        dense[:999], target[:999]
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        np.broadcast_to(dense[:1], (10_001, 32)), np.arange(10_001) % 6
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        dense[:, :7], target
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        np.broadcast_to(dense[:1, :1], (8_000, 63)), np.arange(8_000) % 6
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        dense, np.arange(6_000) % 11
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        dense, rare_target
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        sparse_like, target
    )
    assert not fedot_exec._is_bounded_numeric_extra_trees_candidate(
        sparse.csr_matrix(dense), target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        dense, target, configured="lgbm,xgboost"
    ) == ["lgbm", "xgboost"]


def test_mixed_extra_trees_candidate_is_cost_and_support_bounded():
    row_count = 1_200
    mixed = pd.DataFrame(
        {
            **{
                f"categorical_{index}": pd.Series(
                    np.arange(row_count) % 4, dtype="category"
                )
                for index in range(4)
            },
            **{
                f"numeric_{index}": np.arange(row_count, dtype=np.float32)
                for index in range(12)
            },
        }
    )
    target = np.arange(row_count) % 3
    rare_target = target.copy()
    rare_target[:99] = 3

    assert fedot_exec._is_bounded_mixed_extra_trees_candidate(mixed, target)
    candidates = fedot_exec._adaptive_default_portfolio_candidates(mixed, target)
    assert candidates == [
        "lgbm",
        "xgboost",
        "mixed_logit",
        "mixed_svc",
        "mixed_extra_trees",
        "auto",
    ]
    assert fedot_exec._adaptive_pair_strong_weight(
        mixed, target, candidates
    ) == 0.75
    assert not fedot_exec._is_bounded_mixed_extra_trees_candidate(
        mixed.iloc[:999], target[:999]
    )
    assert not fedot_exec._is_bounded_mixed_extra_trees_candidate(
        mixed.iloc[:, :7], target
    )
    assert fedot_exec._is_bounded_mixed_extra_trees_candidate(
        mixed.iloc[:, :15], target
    )
    assert not fedot_exec._is_bounded_mixed_logit_candidate(
        mixed.iloc[:, :15], target
    )
    assert fedot_exec._is_bounded_mixed_logit_candidate(mixed, target)
    assert fedot_exec._is_bounded_mixed_svc_candidate(mixed, target)
    assert not fedot_exec._is_bounded_mixed_extra_trees_candidate(
        mixed.astype(np.float32), target
    )
    assert not fedot_exec._is_bounded_mixed_extra_trees_candidate(
        mixed, rare_target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        mixed, target, configured="lgbm,xgboost"
    ) == ["lgbm", "xgboost"]


def _small_supported_mixed_frame(
    row_count=600,
    feature_count=16,
    categorical_levels=(6, 6, 6, 6),
):
    categorical_count = len(categorical_levels)
    return pd.DataFrame(
        {
            **{
                f"categorical_{index}": pd.Series(
                    np.arange(row_count) % level_count, dtype="category"
                )
                for index, level_count in enumerate(categorical_levels)
            },
            **{
                f"numeric_{index}": np.arange(row_count, dtype=np.float32)
                for index in range(feature_count - categorical_count)
            },
        }
    )


def test_small_supported_mixed_portfolio_uses_linear_forest_views_and_shrinkage():
    features = _small_supported_mixed_frame()
    target = np.arange(len(features)) % 5

    assert fedot_exec._is_small_supported_mixed_multiclass(features, target)
    candidates = fedot_exec._adaptive_default_portfolio_candidates(features, target)
    assert candidates == [
        "lgbm",
        "xgboost",
        "mixed_logit",
        "mixed_extra_trees",
    ]
    assert fedot_exec._adaptive_pair_strong_weight(
        features, target, candidates
    ) == 0.75
    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target, configured="mixed_logit,auto"
    ) == ["mixed_logit", "auto"]


@pytest.mark.parametrize(
    ("row_count", "feature_count", "categorical_levels", "class_count", "expected"),
    [
        (500, 12, (8, 8, 8), 3, True),
        (999, 64, (6,) * 16, 10, True),
        (499, 12, (8, 8, 8), 3, False),
        (1_000, 16, (6, 6, 6, 6), 5, False),
        (600, 11, (8, 8, 8), 5, False),
        (600, 65, (6,) * 17, 5, False),
        (600, 16, (6, 6, 6, 6), 2, False),
        (600, 16, (6, 6, 6, 6), 11, False),
        (600, 16, (12, 12), 5, False),
    ],
)
def test_small_supported_mixed_portfolio_has_shape_and_support_boundaries(
    row_count, feature_count, categorical_levels, class_count, expected
):
    features = _small_supported_mixed_frame(
        row_count=row_count,
        feature_count=feature_count,
        categorical_levels=categorical_levels,
    )
    target = np.arange(row_count) % class_count

    assert (
        fedot_exec._is_small_supported_mixed_multiclass(features, target)
        is expected
    )


def test_small_supported_mixed_portfolio_requires_thirty_rows_per_class():
    features = _small_supported_mixed_frame()
    supported_target = np.concatenate(
        (np.zeros(30), np.ones(285), np.full(285, 2))
    )
    rare_target = np.concatenate((np.zeros(29), np.ones(285), np.full(286, 2)))

    assert fedot_exec._is_small_supported_mixed_multiclass(
        features, supported_target
    )
    assert not fedot_exec._is_small_supported_mixed_multiclass(features, rare_target)


@pytest.mark.parametrize(
    ("categorical_levels", "expected"),
    [
        ((6, 6, 6, 6), True),
        ((5, 6, 6, 6), False),
        ((64, 64, 64, 64), True),
        ((65, 64, 64, 64), False),
    ],
)
def test_small_supported_mixed_portfolio_bounds_total_categorical_cardinality(
    categorical_levels, expected
):
    features = _small_supported_mixed_frame(
        feature_count=16, categorical_levels=categorical_levels
    )
    target = np.arange(len(features)) % 5

    assert (
        fedot_exec._is_small_supported_mixed_multiclass(features, target)
        is expected
    )


def test_scaled_dense_linear_pair_has_resource_and_support_boundaries():
    features = np.broadcast_to(
        np.zeros((1, 800), dtype=np.float32), (8_000, 800)
    )
    target = np.arange(8_000) % 7
    rare_target = np.arange(8_000) % 6
    rare_target[:99] = 6

    assert fedot_exec._is_dense_scaled_logit_regime(features, target)
    larger_dense_geometry = np.broadcast_to(features[:1, :1], (9_000, 2_000))
    assert not fedot_exec._is_dense_scaled_logit_regime(
        larger_dense_geometry, np.arange(9_000) % 5
    )
    supported_dense_geometry = np.broadcast_to(
        np.ones((1, 2_000), dtype=np.float32), (8_000, 2_000)
    )
    assert fedot_exec._is_dense_scaled_logit_regime(
        supported_dense_geometry, np.arange(8_000) % 5
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        supported_dense_geometry, np.arange(8_000) % 5
    ) == ["scaled_logit", "lgbm"]
    assert not fedot_exec._is_dense_scaled_logit_regime(
        features[:4_999], target[:4_999]
    )
    assert not fedot_exec._is_dense_scaled_logit_regime(
        np.broadcast_to(features[:, :1], (8_000, 2_048)), target
    )
    assert not fedot_exec._is_dense_scaled_logit_regime(
        np.broadcast_to(features[:1, :1], (9_000, 2_001)),
        np.arange(9_000) % 5,
    )
    assert not fedot_exec._is_dense_scaled_logit_regime(
        features, target % 2
    )
    assert not fedot_exec._is_dense_scaled_logit_regime(
        features, rare_target
    )
    assert not fedot_exec._is_dense_scaled_logit_regime(
        sparse.csr_matrix(features), target
    )
    assert fedot_exec._adaptive_logit_c(features, target, ["logit", "xgboost"]) is None


def test_scaled_dense_binary_pair_has_separate_width_rows_and_regularisation():
    features = np.broadcast_to(
        np.ones((1, 1_636), dtype=np.float32), (4_876, 1_636)
    )
    target = np.arange(len(features)) % 2
    rare_target = np.zeros(len(features), dtype=int)
    rare_target[-99:] = 1

    assert fedot_exec._is_dense_scaled_logit_regime(features, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target
    ) == ["scaled_logit", "xgboost"]
    assert fedot_exec._adaptive_logit_c(
        features, target, ["scaled_logit", "xgboost"]
    ) == pytest.approx(0.001)
    assert not fedot_exec._is_dense_scaled_logit_regime(
        features[:4_499], target[:4_499]
    )
    assert not fedot_exec._is_dense_scaled_logit_regime(
        features[:, :1_023], target
    )
    assert not fedot_exec._is_dense_scaled_logit_regime(features, rare_target)


def test_scaled_dense_binary_pair_accepts_bounded_mostly_numeric_one_hot():
    features = pd.DataFrame(
        np.broadcast_to(
            np.ones((1, 1_023), dtype=np.float32),
            (4_500, 1_023),
        )
    )
    features["category"] = pd.Categorical(np.arange(len(features)) % 2)
    target = np.arange(len(features)) % 2

    assert fedot_exec._bounded_mostly_numeric_one_hot_profile(features) is not None
    assert fedot_exec._is_dense_scaled_logit_regime(features, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target
    ) == ["scaled_logit", "xgboost"]

    high_cardinality = features.copy()
    high_cardinality["category"] = pd.Categorical(
        np.arange(len(features)) % 129
    )
    assert fedot_exec._bounded_mostly_numeric_one_hot_profile(
        high_cardinality
    ) is None
    assert not fedot_exec._is_dense_scaled_logit_regime(
        high_cardinality, target
    )


def test_scaled_dense_booster_companion_uses_bounded_density_and_width():
    target = np.arange(6_000) % 5
    dense_boundary = np.broadcast_to(
        np.ones((1, 1_024), dtype=np.float32), (6_000, 1_024)
    )
    dense_narrow = dense_boundary[:, :1_023]
    sparse_like_row = np.zeros((1, 1_024), dtype=np.float32)
    sparse_like_row[:, :511] = 1.0
    sparse_like = np.broadcast_to(sparse_like_row, (6_000, 1_024))

    assert fedot_exec._adaptive_default_portfolio_candidates(
        dense_boundary, target
    ) == ["scaled_logit", "lgbm"]
    assert fedot_exec._adaptive_logit_c(
        dense_boundary, target, ["scaled_logit", "lgbm"]
    ) == pytest.approx(0.01)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        dense_narrow, target
    ) == ["scaled_logit", "xgboost"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        sparse_like, target
    ) == ["scaled_logit", "xgboost"]


def test_extreme_wide_numeric_multiclass_uses_one_subspace_booster():
    features = np.broadcast_to(np.zeros((1, 4_096)), (5_000, 4_096))
    target = np.arange(5_000) % 10

    assert fedot_exec._is_extreme_wide_numeric_multiclass(features, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target
    ) == ["xgboost"]
    assert fedot_exec._adaptive_xgboost_learning_rate(features, target) == 0.15
    assert fedot_exec._adaptive_xgboost_max_bin(features, target) == 32
    assert fedot_exec._adaptive_xgboost_colsample_bytree(features, target) == 0.02
    assert fedot_exec._adaptive_xgboost_max_depth(features, target) == 4
    assert fedot_exec._adaptive_xgboost_subsample(features, target) == 0.9


def test_extreme_wide_policy_has_structural_boundaries():
    target = np.arange(5_000) % 10
    one_column_short = np.broadcast_to(np.zeros((1, 2_047)), (5_000, 2_047))
    binary_target = target % 2
    categorical = pd.DataFrame(
        {"numeric": np.zeros(5_000), "category": ["level"] * 5_000}
    )
    sparse_numeric = sparse.csr_matrix((5_000, 2_048), dtype=np.float32)

    assert not fedot_exec._is_extreme_wide_numeric_multiclass(
        one_column_short, target
    )
    assert not fedot_exec._is_extreme_wide_numeric_multiclass(
        np.broadcast_to(np.zeros((1, 4_096)), (5_000, 4_096)), binary_target
    )
    assert not fedot_exec._is_extreme_wide_numeric_multiclass(
        categorical, target
    )
    assert not fedot_exec._is_extreme_wide_numeric_multiclass(
        sparse_numeric, target
    )
    assert fedot_exec._adaptive_xgboost_colsample_bytree(
        one_column_short, target
    ) is None
    assert fedot_exec._adaptive_xgboost_max_bin(one_column_short, target) is None


def test_default_candidates_use_rf_for_small_numeric_many_class_wide_data():
    small_wide = np.zeros((300, 300))
    many_class = np.arange(300) % 10
    ratio_boundary = np.zeros((400, 300))
    boundary_target = np.arange(400) % 10

    assert fedot_exec._adaptive_default_portfolio_candidates(
        small_wide, many_class
    ) == ["lgbm", "xgboost", "rf"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        ratio_boundary, boundary_target
    ) == ["lgbm", "xgboost", "rf"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        ratio_boundary[:, :299], boundary_target
    ) == ["lgbm", "xgboost", "auto"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        pd.DataFrame(small_wide).assign(category="level"), many_class
    ) == ["lgbm", "xgboost", "auto"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        np.zeros((300, 1_025)), many_class
    ) == ["lgbm", "xgboost", "rf", "rf_large_subspace"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        np.zeros((300, 1_024)), many_class
    ) == ["lgbm", "xgboost", "rf", "rf_large_subspace"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        np.zeros((500, 1_024)), np.arange(500) % 10
    ) == ["lgbm", "xgboost", "rf"]
    rare_boundary_target = np.concatenate(
        (np.arange(298) % 9, np.full(2, 9))
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        np.zeros((300, 1_024)), rare_boundary_target
    ) == ["lgbm", "xgboost", "rf"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        np.zeros((512, 4_096)), np.arange(512) % 10
    ) == ["lgbm", "xgboost", "rf", "rf_large_subspace"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        np.zeros((200, 4_096)), np.arange(200) % 10
    ) == ["scaled_logit", "rf"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        np.zeros((512, 4_095)), np.arange(512) % 10
    ) == ["lgbm", "xgboost", "rf", "rf_large_subspace"]


def test_default_candidates_use_guarded_rf_shrinkage_for_large_narrow_many_class():
    helena_geometry = np.broadcast_to(
        np.zeros((1, 27), dtype=np.float32), (65_000, 27)
    )
    helena_target = np.arange(65_000) % 100
    walking_geometry = np.broadcast_to(
        np.zeros((1, 4), dtype=np.float32), (149_332, 4)
    )
    walking_target = np.arange(149_332) % 22

    for features, target in (
        (helena_geometry, helena_target),
        (walking_geometry, walking_target),
    ):
        candidates = fedot_exec._adaptive_default_portfolio_candidates(
            features, target
        )

        assert fedot_exec._is_well_supported_narrow_many_class_numeric(
            features, target
        )
        assert candidates == ["xgboost", "rf"]
        assert fedot_exec._adaptive_xgboost_max_depth(features, target) == 6
        assert fedot_exec._adaptive_pair_strong_weight(
            features, target, candidates
        ) == 0.75
        assert fedot_exec._adaptive_rf_n_estimators(
            features, target, candidates
        ) == 200


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count", "expected"),
    [
        (92_000, 1_024, 46, True),
        (131_600, 784, 47, True),
        (67_816, 1_024, 29, False),
        (261_438, 255, 30, False),
        (84_999, 784, 30, False),
        (85_035, 784, 30, True),
    ],
)
def test_high_work_many_class_policy_has_structural_boundaries(
    row_count, feature_count, class_count, expected
):
    features = np.broadcast_to(
        np.zeros((1, feature_count), dtype=np.float32),
        (row_count, feature_count),
    )
    target = np.arange(row_count) % class_count

    assert (
        fedot_exec._is_high_work_wide_many_class_numeric(features, target)
        is expected
    )


def test_high_work_many_class_policy_uses_one_capped_lgbm():
    features = np.broadcast_to(
        np.zeros((1, 1_024), dtype=np.float32),
        (92_000, 1_024),
    )
    target = np.arange(92_000) % 46

    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target
    ) == ["lgbm"]
    assert fedot_exec._adaptive_boosting_rounds(features, target) == 80
    assert fedot_exec._adaptive_lgbm_num_leaves(features, target) == 27
    assert fedot_exec._adaptive_lgbm_min_child_weight(features, target) == 1.0


@pytest.mark.parametrize(
    ("selector_temperature", "expected_multiplier"),
    [
        (0.70, 1.10),
        (0.85, 1.0),
        (0.95, 0.94),
    ],
)
def test_high_work_all_row_temperature_transfer_uses_selector_signal(
    selector_temperature, expected_multiplier
):
    features = np.broadcast_to(
        np.zeros((1, 1_024), dtype=np.float32),
        (92_000, 1_024),
    )
    target = np.arange(92_000) % 46

    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features,
        target,
        selected_models=["lgbm"],
        selector_temperature=selector_temperature,
        metric="logloss",
        calibrate=True,
        direct_all_row_refit=True,
        adaptive_portfolio=True,
    ) == pytest.approx(expected_multiplier)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("metric", "auc"),
        ("calibrate", False),
        ("direct_all_row_refit", False),
        ("adaptive_portfolio", False),
        ("selected_models", ["xgboost"]),
    ],
)
def test_high_work_temperature_transfer_preserves_other_paths(override, value):
    features = np.broadcast_to(
        np.zeros((1, 1_024), dtype=np.float32),
        (92_000, 1_024),
    )
    target = np.arange(92_000) % 46
    kwargs = {
        "selected_models": ["lgbm"],
        "selector_temperature": 0.70,
        "metric": "logloss",
        "calibrate": True,
        "direct_all_row_refit": True,
        "adaptive_portfolio": True,
    }
    kwargs[override] = value

    assert (
        fedot_exec._adaptive_deployment_temperature_multiplier(
            features, target, **kwargs
        )
        == 1.0
    )


def test_categorical_temperature_transfer_requires_selected_lgbm():
    categorical = pd.DataFrame(
        {
            "category": pd.Series(np.arange(20_000) % 3, dtype="category"),
            "numeric": np.arange(20_000, dtype=np.float32),
        }
    )
    target = np.arange(20_000) % 4
    kwargs = {
        "selector_temperature": 1.03,
        "metric": "logloss",
        "calibrate": True,
        "direct_all_row_refit": False,
        "adaptive_portfolio": True,
    }

    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        categorical,
        target,
        selected_models=["lgbm", "xgboost"],
        **kwargs,
    ) == pytest.approx(0.97)
    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        categorical,
        target,
        selected_models=["xgboost"],
        **kwargs,
    ) == 1.0
    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        categorical.astype(np.float32),
        target,
        selected_models=["lgbm"],
        **kwargs,
    ) == 1.0

    compact_integer_coded = pd.DataFrame(
        {
            f"feature_{index}": (
                np.arange(20_000) % (index + 3)
            ).astype(np.uint8)
            for index in range(6)
        }
    )
    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        compact_integer_coded,
        target,
        selected_models=["lgbm"],
        **kwargs,
    ) == pytest.approx(0.97)


def test_numeric_leaf_support_temperature_transfer_requires_selected_lgbm():
    features = np.broadcast_to(
        np.zeros((1, 93), dtype=np.float32),
        (20_000, 93),
    )
    target = np.arange(20_000) % 9

    assert fedot_exec._adaptive_lgbm_min_child_samples(features, target) == 100
    kwargs = {
        "selector_temperature": 1.03,
        "metric": "logloss",
        "calibrate": True,
        "direct_all_row_refit": False,
        "adaptive_portfolio": True,
    }

    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features,
        target,
        selected_models=["lgbm", "xgboost"],
        **kwargs,
    ) == pytest.approx(0.97)
    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features,
        target,
        selected_models=["xgboost"],
        **kwargs,
    ) == 1.0
    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features,
        target,
        selected_models=["lgbm"],
        **{**kwargs, "adaptive_portfolio": False},
    ) == 1.0


@pytest.mark.parametrize(
    ("selector_temperature", "expected_multiplier"),
    [
        (1.0499, 1.0),
        (1.05, 0.97),
        (1.10, 0.97),
    ],
)
def test_numeric_leaf_support_post_prior_temperature_uses_softening_signal(
    selector_temperature, expected_multiplier
):
    features = np.broadcast_to(
        np.zeros((1, 93), dtype=np.float32),
        (20_000, 93),
    )
    target = np.arange(20_000) % 9

    assert fedot_exec._adaptive_post_prior_temperature_multiplier(
        features,
        target,
        selected_models=["lgbm", "xgboost"],
        selector_temperature=selector_temperature,
        metric="logloss",
        calibrate=True,
        adaptive_portfolio=True,
    ) == pytest.approx(expected_multiplier)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("selected_models", ["xgboost"]),
        ("metric", "auc"),
        ("calibrate", False),
        ("adaptive_portfolio", False),
    ],
)
def test_numeric_leaf_support_post_prior_temperature_preserves_other_paths(
    override, value
):
    features = np.broadcast_to(
        np.zeros((1, 93), dtype=np.float32),
        (20_000, 93),
    )
    target = np.arange(20_000) % 9
    kwargs = {
        "selected_models": ["lgbm"],
        "selector_temperature": 1.10,
        "metric": "logloss",
        "calibrate": True,
        "adaptive_portfolio": True,
    }
    kwargs[override] = value

    assert (
        fedot_exec._adaptive_post_prior_temperature_multiplier(
            features, target, **kwargs
        )
        == 1.0
    )


@pytest.mark.parametrize(
    ("rows", "features_count", "class_count", "selected_models"),
    [
        (500, 1_300, 20, ["rf", "rf_large_subspace"]),
        (400, 1_024, 40, ["lgbm", "rf_large_subspace"]),
    ],
)
def test_low_support_small_wide_rf_pair_gets_conservative_sharpening(
    rows, features_count, class_count, selected_models
):
    features = np.broadcast_to(
        np.zeros((1, features_count), dtype=np.float32),
        (rows, features_count),
    )
    target = np.arange(rows) % class_count

    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features,
        target,
        selected_models=selected_models,
        selector_temperature=0.5,
        metric="logloss",
        calibrate=True,
        direct_all_row_refit=False,
        adaptive_portfolio=True,
        validation_fold_count=3,
    ) == pytest.approx(0.97)


@pytest.mark.parametrize(
    ("target", "selected_models", "validation_fold_count"),
    [
        (np.arange(500) % 20, ["rf_large_subspace"], 3),
        (np.arange(500) % 20, ["xgboost", "rf"], 3),
        (np.arange(500) % 20, ["rf", "rf_large_subspace"], 1),
        (np.arange(1_000) % 20, ["rf", "rf_large_subspace"], 3),
        (
            np.concatenate((np.arange(498) % 19, np.full(2, 19))),
            ["lgbm", "rf"],
            3,
        ),
    ],
)
def test_low_support_small_wide_rf_temperature_guard_preserves_other_paths(
    target, selected_models, validation_fold_count
):
    rows = len(target)
    features_count = 1_024 if rows > 500 else 1_300
    features = np.broadcast_to(
        np.zeros((1, features_count), dtype=np.float32),
        (rows, features_count),
    )

    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features,
        target,
        selected_models=selected_models,
        selector_temperature=0.5,
        metric="logloss",
        calibrate=True,
        direct_all_row_refit=False,
        adaptive_portfolio=True,
        validation_fold_count=validation_fold_count,
    ) == 1.0


def test_explicit_high_work_portfolio_settings_take_precedence():
    features = np.broadcast_to(
        np.zeros((1, 1_024), dtype=np.float32),
        (92_000, 1_024),
    )
    target = np.arange(92_000) % 46

    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target, configured="xgboost,rf"
    ) == ["xgboost", "rf"]
    assert (
        fedot_exec._adaptive_boosting_rounds(features, target, configured="123")
        == 123
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_weight(
            features, target, configured="0.25"
        )
        == 0.25
    )


def test_high_work_many_class_policy_excludes_sparse_numeric_tables():
    features = sparse.csr_matrix((92_000, 1_024), dtype=np.float32)
    target = np.arange(92_000) % 46

    assert not fedot_exec._is_high_work_wide_many_class_numeric(features, target)
    assert fedot_exec._adaptive_boosting_rounds(features, target) == 300
    assert fedot_exec._adaptive_lgbm_min_child_weight(features, target) is None


def _large_narrow_extreme_many_class_data(
    row_count=200_000,
    feature_count=60,
    class_count=300,
):
    features = np.broadcast_to(
        np.zeros((1, feature_count), dtype=np.float32),
        (row_count, feature_count),
    )
    target = np.arange(row_count, dtype=np.int32) % class_count
    return features, target


def test_large_narrow_extreme_many_class_xgboost_guard_matches_validated_regime():
    features, target = _large_narrow_extreme_many_class_data()

    assert fedot_exec._is_large_narrow_extreme_many_class_numeric(
        features, target
    )
    assert fedot_exec._use_large_narrow_extreme_many_class_xgboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": "true"},
    )
    assert fedot_exec._adaptive_xgboost_learning_rate(features, target) is None
    assert fedot_exec._adaptive_xgboost_max_depth(features, target) == 4
    assert fedot_exec._adaptive_boosting_rounds(features, target) == 300


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count"),
    [
        (199_999, 60, 300),
        (200_000, 47, 300),
        (200_000, 65, 300),
        (200_000, 60, 255),
        (256_500, 60, 513),
        (500_001, 60, 300),
        (200_000, 63, 256),
    ],
)
def test_large_narrow_extreme_many_class_guard_has_strict_shape_boundaries(
    row_count, feature_count, class_count
):
    features, target = _large_narrow_extreme_many_class_data(
        row_count=row_count,
        feature_count=feature_count,
        class_count=class_count,
    )

    assert not fedot_exec._is_large_narrow_extreme_many_class_numeric(
        features, target
    )


def test_large_narrow_extreme_many_class_guard_requires_class_support():
    features, _ = _large_narrow_extreme_many_class_data()
    target = np.concatenate(
        (
            np.zeros(499, dtype=np.int32),
            1 + np.arange(len(features) - 499, dtype=np.int32) % 299,
        )
    )

    assert not fedot_exec._is_large_narrow_extreme_many_class_numeric(
        features, target
    )


def test_large_narrow_extreme_many_class_guard_excludes_sparse_and_categorical():
    features, target = _large_narrow_extreme_many_class_data()
    sparse_features = sparse.csr_matrix(features.shape, dtype=np.float32)
    categorical_features = pd.DataFrame(features, copy=False)
    categorical_features[0] = pd.Categorical(
        np.zeros(len(features), dtype=np.uint8)
    )

    assert not fedot_exec._is_large_narrow_extreme_many_class_numeric(
        sparse_features, target
    )
    assert not fedot_exec._is_large_narrow_extreme_many_class_numeric(
        categorical_features, target
    )


@pytest.mark.parametrize(
    ("metric", "runtime_seconds", "cores", "framework_params"),
    [
        ("accuracy", 180, 4, {"_portfolio": "true"}),
        ("logloss", 179, 4, {"_portfolio": "true"}),
        ("logloss", 180, 3, {"_portfolio": "true"}),
        (
            "logloss",
            180,
            4,
            {"_portfolio": "true", "_portfolio_candidates": "xgboost"},
        ),
        (
            "logloss",
            180,
            4,
            {"_portfolio": "true", "_portfolio_train_rows": 10_000},
        ),
    ],
)
def test_large_narrow_extreme_many_class_xgboost_guard_respects_context(
    metric, runtime_seconds, cores, framework_params
):
    features, target = _large_narrow_extreme_many_class_data()

    assert not fedot_exec._use_large_narrow_extreme_many_class_xgboost(
        features,
        target,
        metric=metric,
        runtime_seconds=runtime_seconds,
        cores=cores,
        framework_params=framework_params,
    )


def test_large_narrow_extreme_many_class_policy_uses_xgboost_and_ten_k_rows(
    monkeypatch,
):
    fit_calls = []

    class FakeFedot:
        def __init__(self, **kwargs):
            self.current_pipeline = SimpleNamespace(length=1)

        def fit(self, features, target, predefined_model):
            fit_calls.append((len(features), predefined_model))

        def predict(self, features):
            return np.zeros(len(features), dtype=np.int32)

        def predict_proba(self, features, probs_for_all_classes):
            del probs_for_all_classes
            return np.full((len(features), 300), 1.0 / 300)

    features, target = _large_narrow_extreme_many_class_data()
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:20], y=target[:20]),
        encoded_class_count=300,
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=4,
        seed=42,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    monkeypatch.setattr(fedot_exec, "Fedot", FakeFedot)
    monkeypatch.setattr(
        fedot_exec,
        "_predefined_model_with_n_jobs",
        lambda predefined_model,
        n_jobs,
        n_estimators=None,
        use_eval_set=None,
        model_params=None: (
            predefined_model,
            n_estimators,
            model_params,
        ),
    )
    monkeypatch.setattr(
        fedot_exec, "_fit_temperature", lambda *args, **kwargs: 1.0
    )
    monkeypatch.setattr(
        fedot_exec, "_fit_prior_exponent", lambda *args, **kwargs: 0.0
    )
    monkeypatch.setattr(fedot_exec, "save_artifacts", lambda *args: None)
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fit_calls[0][0] == 10_000
    assert all(row_count == 200_000 for row_count, _ in fit_calls[1:])
    assert all(model[0] == "xgboost" for _, model in fit_calls)
    assert all(model[1] == 300 for _, model in fit_calls)
    assert all(model[2]["max_depth"] == 4 for _, model in fit_calls)
    assert output["models_count"] == 1


def test_direct_extreme_wide_fit_uses_resource_and_budget_guards():
    features = np.broadcast_to(
        np.zeros((1, 7_200), dtype=np.float32),
        (9_000, 7_200),
    )
    target = np.arange(9_000) % 10

    assert fedot_exec._is_resource_bounded_extreme_wide_numeric(features, target)
    assert fedot_exec._use_fixed_round_all_row_xgboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": "true"},
    )
    assert not fedot_exec._use_fixed_round_all_row_xgboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=179,
        cores=4,
        framework_params={"_portfolio": "true"},
    )
    assert not fedot_exec._use_fixed_round_all_row_xgboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=3,
        framework_params={"_portfolio": "true"},
    )
    assert not fedot_exec._use_fixed_round_all_row_xgboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio": "true",
            "_portfolio_boosting_rounds": 123,
        },
    )


def test_direct_extreme_wide_fit_excludes_oversized_sparse_and_binary_data():
    target = np.arange(9_000) % 10
    oversized = np.broadcast_to(
        np.zeros((1, 8_000), dtype=np.float32),
        (9_000, 8_000),
    )
    sparse_features = sparse.csr_matrix((9_000, 7_200), dtype=np.float32)

    assert not fedot_exec._is_resource_bounded_extreme_wide_numeric(
        oversized, target
    )
    assert not fedot_exec._is_resource_bounded_extreme_wide_numeric(
        sparse_features, target
    )
    assert not fedot_exec._is_resource_bounded_extreme_wide_numeric(
        oversized[:, :7_200], np.arange(9_000) % 2
    )


def test_screened_wide_binary_auc_uses_signal_and_budget_guards(monkeypatch):
    features = np.broadcast_to(
        np.ones((1, 4_096), dtype=np.float32),
        (16_000, 4_096),
    )
    target = np.arange(16_000) % 2
    profile = {
        "density": 0.54,
        "score_q90": 2.2,
        "score_q99": 25.0,
        "tail_ratio": 10.0,
        "strong_features": 80,
    }
    monkeypatch.setattr(
        fedot_exec,
        "_sampled_wide_binary_signal_profile",
        lambda features, target: profile,
    )

    assert fedot_exec._is_screenable_wide_binary_auc_regime(features, target)
    assert fedot_exec._use_screened_wide_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_screened_wide_auc_lgbm(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_screened_wide_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=179,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_screened_wide_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio": True,
            "_portfolio_candidates": "lgbm",
        },
    )


def test_screened_wide_binary_auc_rejects_diffuse_and_sparse_signal(monkeypatch):
    features = np.broadcast_to(
        np.ones((1, 4_096), dtype=np.float32),
        (16_000, 4_096),
    )
    target = np.arange(16_000) % 2

    monkeypatch.setattr(
        fedot_exec,
        "_sampled_wide_binary_signal_profile",
        lambda features, target: {
            "density": 1.0,
            "score_q90": 2.0,
            "score_q99": 24.0,
            "tail_ratio": 12.0,
            "strong_features": 80,
        },
    )
    assert not fedot_exec._is_screenable_wide_binary_auc_regime(features, target)

    sparse_features = sparse.csr_matrix((16_000, 4_096), dtype=np.float32)
    assert not fedot_exec._is_screenable_wide_binary_auc_regime(
        sparse_features, target
    )
    assert not fedot_exec._is_screenable_wide_binary_auc_regime(
        features, np.arange(16_000) % 3
    )


def test_screened_wide_binary_auc_estimator_has_fixed_bounded_parameters():
    estimator = fedot_exec._screened_wide_auc_lgbm_estimator(seed=17, n_jobs=4)

    assert estimator.named_steps["selectkbest"].k == 1_000
    parameters = estimator.named_steps["lgbmclassifier"].get_params()
    assert parameters["n_estimators"] == 1_000
    assert parameters["learning_rate"] == pytest.approx(0.05)
    assert parameters["num_leaves"] == 127
    assert parameters["max_bin"] == 255
    assert parameters["colsample_bytree"] == pytest.approx(0.5)
    assert parameters["min_child_samples"] == 20
    assert parameters["n_jobs"] == 4
    assert parameters["random_state"] == 17


def test_screened_wide_binary_auc_path_bypasses_fedot(monkeypatch):
    constructed = []

    class FakeScreenedModel:
        def fit(self, features, target):
            constructed.append((features.shape, len(target)))
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (features.shape[0], 1))

    train_features = np.ones((20, 8), dtype=np.float32)
    train_target = np.arange(20) % 2
    test_features = np.ones((6, 8), dtype=np.float32)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=np.arange(6) % 2),
        encoded_class_count=2,
    )
    config = SimpleNamespace(
        type="classification",
        metric="auc",
        cores=4,
        seed=17,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    monkeypatch.setattr(
        fedot_exec,
        "_use_screened_wide_auc_lgbm",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_screened_wide_auc_lgbm_estimator",
        lambda seed, n_jobs: FakeScreenedModel(),
    )
    monkeypatch.setattr(
        fedot_exec,
        "Fedot",
        lambda **kwargs: pytest.fail("FEDOT must not be constructed"),
    )
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert constructed == [((20, 8), 20)]
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["probabilities"] == pytest.approx(
        np.tile([0.4, 0.6], (6, 1))
    )


def test_medium_screened_auc_uses_signal_and_budget_guards(monkeypatch):
    features = np.broadcast_to(
        np.ones((1, 300), dtype=np.float32),
        (3_000, 300),
    )
    target = np.arange(3_000) % 2
    profile = {
        "density": 0.99,
        "score_q95": 27.0,
        "score_q99": 69.0,
        "strong_features": 16,
    }
    monkeypatch.setattr(
        fedot_exec,
        "_medium_wide_binary_signal_profile",
        lambda features, target: profile,
    )

    assert fedot_exec._is_medium_screenable_binary_auc_regime(features, target)
    assert fedot_exec._use_medium_screened_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_medium_screened_auc_lgbm(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_medium_screened_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=3,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_medium_screened_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio": True,
            "_portfolio_candidates": "lgbm",
        },
    )

    profile["score_q99"] = 49.9
    assert not fedot_exec._is_medium_screenable_binary_auc_regime(features, target)
    profile["score_q99"] = 69.0
    rare_target = np.r_[np.zeros(2_501, dtype=int), np.ones(499, dtype=int)]
    assert not fedot_exec._is_medium_screenable_binary_auc_regime(
        features, rare_target
    )
    assert not fedot_exec._is_medium_screenable_binary_auc_regime(
        features[:, :255], target
    )


def test_medium_screened_auc_rejects_dense_noise():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(2_500, 256)).astype(np.float32)
    target = np.arange(2_500) % 2

    profile = fedot_exec._medium_wide_binary_signal_profile(features, target)

    assert profile["density"] > 0.99
    assert profile["score_q99"] < 50.0
    assert not fedot_exec._is_medium_screenable_binary_auc_regime(
        features, target
    )


def test_medium_screened_estimator_selects_width_on_train_only_data(monkeypatch):
    fitted_shapes = []

    class FakeLGBMClassifier:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, features, target):
            fitted_shapes.append(np.asarray(features).shape)
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            positive = np.clip(
                0.1 + 0.8 * np.asarray(features)[:, 0], 0.01, 0.99
            )
            return np.column_stack((1.0 - positive, positive))

    rng = np.random.default_rng(17)
    target = np.arange(100) % 2
    features = rng.normal(scale=0.01, size=(100, 256)).astype(np.float32)
    features[:, 0] += target
    monkeypatch.setattr(fedot_exec, "LGBMClassifier", FakeLGBMClassifier)
    estimator = fedot_exec._medium_screened_auc_lgbm_estimator(
        seed=17, n_jobs=4
    )

    estimator.fit(features, target)
    probabilities = estimator.predict_proba(features[:5])

    assert estimator.n_estimators == 1_200
    assert estimator.selection_tolerance == pytest.approx(0.002)
    assert estimator.n_jobs == 4
    assert estimator.random_state == 17
    parameters = estimator.estimator_.parameters
    assert parameters["n_estimators"] == 1_200
    assert parameters["learning_rate"] == pytest.approx(0.02)
    assert parameters["num_leaves"] == 63
    assert estimator.selected_feature_count_ == 64
    assert sorted(estimator.selector_scores_) == [64, 96, 128, 160, 192, 256]
    assert fitted_shapes == [
        (80, 64),
        (80, 96),
        (80, 128),
        (80, 160),
        (80, 192),
        (80, 256),
        (100, 64),
    ]
    assert probabilities.shape == (5, 2)


def test_medium_screened_estimator_prefers_narrower_near_tie(monkeypatch):
    validation_auc = {
        64: 0.78,
        96: 0.80,
        128: 0.8015,
        160: 0.79,
        192: 0.78,
        256: 0.77,
    }

    class WidthReportingClassifier:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, features, target):
            self.width = np.asarray(features).shape[1]
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            positive = np.full(len(features), self.width, dtype=float)
            return np.column_stack((1.0 - positive, positive))

    monkeypatch.setattr(fedot_exec, "LGBMClassifier", WidthReportingClassifier)
    monkeypatch.setattr(
        fedot_exec,
        "roc_auc_score",
        lambda target, probabilities: validation_auc[int(probabilities[0])],
    )
    estimator = fedot_exec._medium_screened_auc_lgbm_estimator(
        seed=17, n_jobs=4
    )
    features = np.random.default_rng(17).normal(size=(100, 256))

    estimator.fit(features, np.arange(100) % 2)

    assert estimator.selected_feature_count_ == 96


def test_medium_screened_auc_path_bypasses_fedot(monkeypatch):
    fitted = []

    class FakeMediumScreenedModel:
        selector_scores_ = {96: 0.9, 128: 0.88}
        selected_feature_count_ = 96

        def fit(self, features, target):
            fitted.append((features.shape, len(target)))
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (features.shape[0], 1))

    train_features = np.ones((20, 8), dtype=np.float32)
    train_target = np.arange(20) % 2
    test_features = np.ones((6, 8), dtype=np.float32)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=np.arange(6) % 2),
        encoded_class_count=2,
    )
    config = SimpleNamespace(
        type="classification",
        metric="auc",
        cores=4,
        seed=17,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    monkeypatch.setattr(
        fedot_exec,
        "_use_medium_screened_auc_lgbm",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_medium_screened_auc_lgbm_estimator",
        lambda seed, n_jobs: FakeMediumScreenedModel(),
    )
    monkeypatch.setattr(
        fedot_exec,
        "Fedot",
        lambda **kwargs: pytest.fail("FEDOT must not be constructed"),
    )
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fitted == [((20, 8), 20)]
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["probabilities"] == pytest.approx(
        np.tile([0.4, 0.6], (6, 1))
    )


def test_high_missing_frequency_auc_uses_structural_and_budget_guards(monkeypatch):
    features = np.broadcast_to(
        np.zeros((1, 230), dtype=np.float32),
        (45_000, 230),
    )
    target = (np.arange(45_000) % 10 == 0).astype(int)
    profile = {
        "categorical_columns": list(range(56)),
        "numeric_columns": list(range(56, 230)),
        "missing_fraction": 0.70,
        "categorical_cardinality": 67_000,
        "median_cardinality": 3.0,
        "maximum_cardinality": 14_000,
    }
    monkeypatch.setattr(
        fedot_exec,
        "_high_missing_mixed_profile",
        lambda features: profile,
    )

    assert fedot_exec._is_high_missing_mixed_binary_auc_regime(features, target)
    assert fedot_exec._use_high_missing_frequency_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_high_missing_frequency_auc_lgbm(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_high_missing_frequency_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=179,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_high_missing_frequency_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio": True,
            "_portfolio_candidates": "lgbm",
        },
    )

    profile["missing_fraction"] = 0.49
    assert not fedot_exec._is_high_missing_mixed_binary_auc_regime(features, target)
    profile["missing_fraction"] = 0.70
    profile["categorical_cardinality"] = 9_999
    assert not fedot_exec._is_high_missing_mixed_binary_auc_regime(features, target)


def test_high_missing_frequency_estimator_has_fixed_bounded_parameters():
    estimator = fedot_exec._high_missing_frequency_auc_lgbm_estimator(
        seed=17, n_jobs=4
    )

    assert estimator.n_estimators == 200
    assert estimator.learning_rate == pytest.approx(0.05)
    assert estimator.num_leaves == 15
    assert estimator.min_child_samples == 20
    assert estimator.n_jobs == 4
    assert estimator.random_state == 17


def test_high_missing_frequency_estimator_uses_train_only_mapping(monkeypatch):
    fitted = []
    transformed = []

    class FakeLGBMClassifier:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, features, target):
            fitted.append(np.asarray(features).copy())
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            transformed.append(np.asarray(features).copy())
            return np.tile([0.5, 0.5], (len(features), 1))

    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "category": pd.Series(["a", "a", "a", "b"], dtype="category"),
        }
    )
    monkeypatch.setattr(fedot_exec, "LGBMClassifier", FakeLGBMClassifier)
    estimator = fedot_exec._high_missing_frequency_auc_lgbm_estimator(
        seed=17, n_jobs=4
    )

    estimator.fit(frame, np.array([0, 1, 0, 1]))
    estimator.predict_proba(
        pd.DataFrame(
            {
                "numeric": [5.0, 6.0],
                "category": pd.Series(["a", "unseen"], dtype="category"),
            }
        )
    )

    assert fitted[0][:, 1] == pytest.approx([0.75, 0.75, 0.75, 0.25])
    np.testing.assert_allclose(transformed[0], [[5.0, 0.75], [6.0, 0.0]])


def test_high_missing_frequency_auc_path_is_quality_gated(monkeypatch):
    fitted_rows = []

    class FakeFrequencyModel:
        def fit(self, features, target):
            fitted_rows.append(len(features))
            return self

        def predict_proba(self, features):
            positive = features["signal"].to_numpy(dtype=float) * 0.8 + 0.1
            return np.column_stack((1.0 - positive, positive))

    train_target = np.arange(100) % 2
    train_features = pd.DataFrame(
        {
            "signal": train_target,
            "category": np.where(train_target, "positive", "negative"),
        }
    )
    test_features = train_features.iloc[:6].copy()
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=train_target[:6]),
        encoded_class_count=2,
    )
    config = SimpleNamespace(
        type="classification",
        metric="auc",
        cores=4,
        seed=17,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )
    monkeypatch.setattr(
        fedot_exec,
        "_use_high_missing_frequency_auc_lgbm",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_high_missing_frequency_auc_lgbm_estimator",
        lambda seed, n_jobs: FakeFrequencyModel(),
    )
    monkeypatch.setattr(
        fedot_exec,
        "Fedot",
        lambda **kwargs: pytest.fail("FEDOT must not be constructed"),
    )
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fitted_rows == [80, 100]
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["probabilities"] == pytest.approx(
        np.column_stack(
            (
                1.0 - (test_features["signal"].to_numpy() * 0.8 + 0.1),
                test_features["signal"].to_numpy() * 0.8 + 0.1,
            )
        )
    )


def test_high_missing_frequency_auc_rejects_noise_quality(monkeypatch):
    fitted_rows = []

    class NoiseFrequencyModel:
        def fit(self, features, target):
            fitted_rows.append(len(features))
            return self

        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))

    monkeypatch.setattr(
        fedot_exec,
        "_high_missing_frequency_auc_lgbm_estimator",
        lambda seed, n_jobs: NoiseFrequencyModel(),
    )
    features = pd.DataFrame(
        {"numeric": np.arange(100), "category": ["a", "b"] * 50}
    )
    config = SimpleNamespace(
        seed=17,
        cores=4,
        framework_params={"_portfolio": True},
        output_predictions_file="unused.csv",
    )

    output = fedot_exec._run_high_missing_frequency_auc_lgbm(
        train_features=features,
        train_target=np.arange(100) % 2,
        test_features=features.iloc[:6],
        test_target=np.arange(6) % 2,
        config=config,
        observed_encoded_labels=np.array([0, 1]),
        encoded_class_count=2,
    )

    assert output is None
    assert fitted_rows == [80]


def test_target_frequency_auc_uses_high_cardinality_and_budget_guards(monkeypatch):
    features = np.broadcast_to(
        np.zeros((1, 64), dtype=np.float32),
        (200_000, 64),
    )
    target = np.arange(200_000) % 2
    profile = {
        "categorical_columns": list(range(40)),
        "numeric_columns": list(range(40, 64)),
        "categorical_cardinality": 100_000,
        "median_cardinality": 500.0,
        "maximum_cardinality": 10_000,
        "encoded_columns": 104,
    }
    monkeypatch.setattr(
        fedot_exec,
        "_large_high_cardinality_mixed_profile",
        lambda features: profile,
    )

    assert fedot_exec._is_large_high_cardinality_mixed_binary_regime(
        features, target
    )
    assert fedot_exec._use_target_frequency_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_target_frequency_auc_lgbm(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_target_frequency_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=3,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_target_frequency_auc_lgbm(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio": True,
            "_portfolio_candidates": "lgbm",
        },
    )


def test_target_frequency_auc_rejects_low_cardinality_and_oversized_profiles(
    monkeypatch,
):
    features = np.broadcast_to(
        np.zeros((1, 64), dtype=np.float32),
        (200_000, 64),
    )
    target = np.arange(200_000) % 2
    profile = {
        "categorical_columns": list(range(40)),
        "numeric_columns": list(range(40, 64)),
        "categorical_cardinality": 1_000,
        "median_cardinality": 10.0,
        "maximum_cardinality": 100,
        "encoded_columns": 104,
    }
    monkeypatch.setattr(
        fedot_exec,
        "_large_high_cardinality_mixed_profile",
        lambda features: profile,
    )

    assert not fedot_exec._is_large_high_cardinality_mixed_binary_regime(
        features, target
    )
    assert not fedot_exec._is_large_high_cardinality_mixed_binary_regime(
        features, np.arange(200_000) % 3
    )
    assert not fedot_exec._is_large_high_cardinality_mixed_binary_regime(
        features[:99_999], target[:99_999]
    )


def test_target_frequency_estimator_cross_fits_unique_categories(monkeypatch):
    fitted = []
    transformed = []

    class FakeLGBMClassifier:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, features, target):
            fitted.append(np.asarray(features).copy())
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            transformed.append(np.asarray(features).copy())
            return np.tile([0.4, 0.6], (len(features), 1))

    frame = pd.DataFrame(
        {
            "numeric": np.arange(12, dtype=float),
            "category": pd.Series(
                [f"level-{index}" for index in range(12)], dtype="category"
            ),
        }
    )
    target = np.arange(12) % 2
    monkeypatch.setattr(fedot_exec, "LGBMClassifier", FakeLGBMClassifier)
    estimator = fedot_exec._target_frequency_auc_lgbm_estimator(
        seed=17, n_jobs=4
    )

    estimator.fit(frame, target)
    estimator.predict_proba(
        pd.DataFrame(
            {
                "numeric": [1.0, 2.0],
                "category": pd.Series(["level-0", "unseen"], dtype="category"),
            }
        )
    )

    assert estimator.smoothing == pytest.approx(10.0)
    assert estimator.n_estimators == 1_000
    assert estimator.num_leaves == 31
    assert estimator.min_child_samples == 100
    assert estimator.n_jobs == 4
    assert estimator.random_state == 17
    assert fitted[0].shape == (12, 3)
    assert fitted[0][:, 1] == pytest.approx(np.full(12, 0.5))
    assert fitted[0][:, 2] == pytest.approx(np.full(12, 1.0 / 12.0))
    assert transformed[0][0, 1] == pytest.approx(5.0 / 11.0)
    assert transformed[0][0, 2] == pytest.approx(1.0 / 12.0)
    assert transformed[0][1, 1:] == pytest.approx([0.5, 0.0])


def test_target_frequency_auc_path_bypasses_fedot(monkeypatch):
    fitted = []

    class FakeTargetFrequencyModel:
        def fit(self, features, target):
            fitted.append((features.shape, len(target)))
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (features.shape[0], 1))

    train_features = pd.DataFrame(
        {"numeric": np.ones(20), "category": pd.Series(["a", "b"] * 10)}
    )
    train_target = np.arange(20) % 2
    test_features = train_features.iloc[:6].copy()
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=train_target[:6]),
        encoded_class_count=2,
    )
    config = SimpleNamespace(
        type="classification",
        metric="auc",
        cores=4,
        seed=17,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    monkeypatch.setattr(
        fedot_exec,
        "_use_target_frequency_auc_lgbm",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_target_frequency_auc_lgbm_estimator",
        lambda seed, n_jobs: FakeTargetFrequencyModel(),
    )
    monkeypatch.setattr(
        fedot_exec,
        "_frequency_auc_lgbm_estimator",
        lambda seed, n_jobs: FakeTargetFrequencyModel(),
    )
    monkeypatch.setattr(
        fedot_exec,
        "Fedot",
        lambda **kwargs: pytest.fail("FEDOT must not be constructed"),
    )
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fitted == [((16, 2), 16), ((16, 2), 16), ((20, 2), 20)]
    assert output["models_count"] == 1
    assert output["probabilities"] == pytest.approx(
        np.tile([0.4, 0.6], (6, 1))
    )


def test_narrow_categorical_auc_uses_structural_and_budget_guards(monkeypatch):
    features = np.broadcast_to(
        np.zeros((1, 10), dtype=np.float32),
        (30_000, 10),
    )
    target = (np.arange(30_000) % 10 == 0).astype(int)
    profile = {
        "categorical_columns": list(range(8)),
        "categorical_cardinality": 15_000,
        "median_cardinality": 337.0,
        "maximum_cardinality": 7_085,
    }
    monkeypatch.setattr(
        fedot_exec,
        "_narrow_high_cardinality_categorical_profile",
        lambda features: profile,
    )

    assert fedot_exec._is_narrow_high_cardinality_categorical_binary_regime(
        features, target
    )
    assert fedot_exec._use_narrow_categorical_auc_catboost(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_narrow_categorical_auc_catboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_narrow_categorical_auc_catboost(
        features,
        target,
        metric="auc",
        runtime_seconds=179,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_narrow_categorical_auc_catboost(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio": True,
            "_portfolio_candidates": "lgbm",
        },
    )

    profile["categorical_cardinality"] = 4_999
    assert not fedot_exec._is_narrow_high_cardinality_categorical_binary_regime(
        features, target
    )
    profile["categorical_cardinality"] = 15_000
    profile["categorical_columns"] = list(range(7))
    assert not fedot_exec._is_narrow_high_cardinality_categorical_binary_regime(
        features, target
    )
    profile["categorical_columns"] = list(range(8))
    rare_target = np.r_[np.zeros(29_001, dtype=int), np.ones(999, dtype=int)]
    assert not fedot_exec._is_narrow_high_cardinality_categorical_binary_regime(
        features, rare_target
    )


def test_native_catboost_category_mapping_is_train_fitted(monkeypatch):
    fitted = []
    transformed = []

    class FakeCatBoostClassifier:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, features, target, cat_features):
            fitted.append(
                (pd.DataFrame(features).copy(), np.asarray(target), cat_features)
            )
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            transformed.append(pd.DataFrame(features).copy())
            return np.tile([0.4, 0.6], (len(features), 1))

        def get_best_iteration(self):
            # Native CatBoost returns None when fit did not receive eval_set.
            return None

    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "category": pd.Series(["known", "other", "known"], dtype="category"),
        }
    )
    test = pd.DataFrame(
        {
            "numeric": [4.0, 5.0, 6.0],
            "category": pd.Series(["known", "unseen", None], dtype="category"),
        }
    )
    monkeypatch.setattr(fedot_exec, "CatBoostClassifier", FakeCatBoostClassifier)
    estimator = fedot_exec._native_categorical_auc_catboost_estimator(
        seed=17, n_jobs=4, iterations=200
    )

    estimator.fit(train, np.array([0, 1, 0]))
    estimator.predict_proba(test)

    assert estimator.iterations == 200
    assert estimator.learning_rate == pytest.approx(0.05)
    assert estimator.depth == 8
    assert estimator.n_jobs == 4
    assert estimator.random_state == 17
    assert estimator.best_iteration_ == 200
    assert fitted[0][0]["category"].to_numpy() == pytest.approx([0, 1, 0])
    assert fitted[0][2] == [1]
    assert transformed[0]["category"].to_numpy() == pytest.approx([0, -1, -1])
    assert fitted[0][0].columns.tolist() == transformed[0].columns.tolist()


def test_narrow_categorical_auc_selector_requires_quality_and_gain():
    assert fedot_exec._select_narrow_categorical_auc_model(0.90, 0.89) == (
        "catboost"
    )
    assert fedot_exec._select_narrow_categorical_auc_model(0.74, 0.50) == (
        "target_frequency"
    )
    assert fedot_exec._select_narrow_categorical_auc_model(0.90, 0.896) == (
        "target_frequency"
    )


def test_narrow_categorical_auc_path_bypasses_fedot(monkeypatch):
    fitted = []

    class FakeCatBoostModel:
        def __init__(self, iterations):
            self.iterations = iterations

        def fit(self, features, target):
            fitted.append(("catboost", self.iterations, features.shape, len(target)))
            return self

        def predict_proba(self, features):
            positive = features["signal"].to_numpy(dtype=float) * 0.8 + 0.1
            return np.column_stack((1.0 - positive, positive))

    class FakeBaselineModel:
        def __init__(self, selector):
            self.selector = selector

        def fit(self, features, target):
            fitted.append(("baseline", self.selector, features.shape, len(target)))
            return self

        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))

    train_target = np.arange(20) % 2
    train_features = pd.DataFrame(
        {
            "signal": train_target,
            "category": pd.Series(["a", "b"] * 10, dtype="category"),
        }
    )
    test_features = train_features.iloc[:6].copy()
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=train_target[:6]),
        encoded_class_count=2,
    )
    config = SimpleNamespace(
        type="classification",
        metric="auc",
        cores=4,
        seed=17,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    monkeypatch.setattr(
        fedot_exec,
        "_use_narrow_categorical_auc_catboost",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_native_categorical_auc_catboost_estimator",
        lambda seed, n_jobs, iterations: FakeCatBoostModel(iterations),
    )
    monkeypatch.setattr(
        fedot_exec,
        "_narrow_categorical_auc_baseline_estimator",
        lambda seed, n_jobs, selector: FakeBaselineModel(selector),
    )
    monkeypatch.setattr(
        fedot_exec,
        "Fedot",
        lambda **kwargs: pytest.fail("FEDOT must not be constructed"),
    )
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fitted == [
        ("catboost", 200, (16, 2), 16),
        ("baseline", True, (16, 2), 16),
        ("catboost", 500, (20, 2), 20),
    ]
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["probabilities"] == pytest.approx(
        np.column_stack(
            (
                1.0 - (test_features["signal"].to_numpy() * 0.8 + 0.1),
                test_features["signal"].to_numpy() * 0.8 + 0.1,
            )
        )
    )


def test_nominal_multiclass_catboost_uses_structural_and_budget_guards(monkeypatch):
    features = np.broadcast_to(
        np.zeros((1, 19), dtype=np.float32),
        (45_000, 19),
    )
    target = np.arange(45_000) % 3
    profile = {
        "categorical_columns": list(range(17)),
        "numeric_columns": [17, 18],
        "categorical_cardinality": 7_000,
        "median_cardinality": 15.0,
        "maximum_cardinality": 6_400,
    }
    monkeypatch.setattr(
        fedot_exec,
        "_nominal_multiclass_profile",
        lambda features: profile,
    )

    assert fedot_exec._is_nominal_heavy_multiclass_regime(features, target)
    assert fedot_exec._use_nominal_multiclass_catboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_nominal_multiclass_catboost(
        features,
        target,
        metric="auc",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_nominal_multiclass_catboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=179,
        cores=4,
        framework_params={"_portfolio": True},
    )
    assert not fedot_exec._use_nominal_multiclass_catboost(
        features,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio": True,
            "_portfolio_candidates": "xgboost",
        },
    )

    profile["categorical_cardinality"] = 1_999
    assert not fedot_exec._is_nominal_heavy_multiclass_regime(features, target)
    profile["categorical_cardinality"] = 7_000
    profile["maximum_cardinality"] = 499
    assert not fedot_exec._is_nominal_heavy_multiclass_regime(features, target)


def test_nominal_multiclass_catboost_has_fixed_parameters_and_gain_gate():
    estimator = fedot_exec._nominal_multiclass_catboost_estimator(
        seed=17,
        n_jobs=4,
        iterations=191,
    )

    assert estimator.iterations == 191
    assert estimator.learning_rate == pytest.approx(0.15)
    assert estimator.depth == 6
    assert estimator.loss_function == "MultiClass"
    assert estimator.eval_metric == "MultiClass"
    assert estimator.n_jobs == 4
    assert estimator.random_state == 17
    assert fedot_exec._select_nominal_multiclass_model(0.80, 0.81, 0.90) == (
        "catboost"
    )
    assert fedot_exec._select_nominal_multiclass_model(0.80, 0.804, 0.90) == (
        "portfolio"
    )
    assert fedot_exec._select_nominal_multiclass_model(0.895, 0.91, 0.90) == (
        "portfolio"
    )


def test_nominal_multiclass_catboost_path_bypasses_fedot(monkeypatch):
    fitted = []

    class FakeCatBoostModel:
        def __init__(self, iterations):
            self.iterations = iterations
            self.best_iteration_ = 37

        def fit(
            self,
            features,
            target,
            eval_set=None,
            early_stopping_rounds=None,
        ):
            fitted.append(
                (
                    self.iterations,
                    len(features),
                    eval_set is not None,
                    early_stopping_rounds,
                )
            )
            return self

        def predict_proba(self, features):
            labels = features["signal"].to_numpy(dtype=int)
            probabilities = np.full((len(features), 3), 0.05)
            probabilities[np.arange(len(features)), labels] = 0.90
            return probabilities

    train_target = np.arange(300) % 3
    train_features = pd.DataFrame(
        {
            "signal": train_target,
            "category": pd.Series(["a", "b", "c"] * 100, dtype="category"),
        }
    )
    test_features = train_features.iloc[:9].copy()
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=train_target[:9]),
        encoded_class_count=3,
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=4,
        seed=17,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )
    monkeypatch.setattr(
        fedot_exec,
        "_use_nominal_multiclass_catboost",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_nominal_multiclass_catboost_estimator",
        lambda seed, n_jobs, iterations: FakeCatBoostModel(iterations),
    )
    monkeypatch.setattr(
        fedot_exec,
        "_nominal_multiclass_lgbm_selector",
        lambda *args, **kwargs: np.full((60, 3), 1.0 / 3.0),
    )
    monkeypatch.setattr(
        fedot_exec,
        "Fedot",
        lambda **kwargs: pytest.fail("FEDOT must not be constructed"),
    )
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fitted == [(200, 240, True, 50), (37, 300, False, None)]
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["predictions"] == pytest.approx(train_target[:9])


def test_nominal_multiclass_catboost_rejects_missing_gain(monkeypatch):
    class UniformCatBoostModel:
        best_iteration_ = 50

        def fit(self, *args, **kwargs):
            return self

        def predict_proba(self, features):
            return np.full((len(features), 3), 1.0 / 3.0)

    features = pd.DataFrame(
        {
            "numeric": np.arange(300),
            "category": pd.Series(["a", "b", "c"] * 100, dtype="category"),
        }
    )
    config = SimpleNamespace(
        seed=17,
        cores=4,
        framework_params={"_portfolio": True},
        output_predictions_file="unused.csv",
    )
    monkeypatch.setattr(
        fedot_exec,
        "_nominal_multiclass_catboost_estimator",
        lambda *args, **kwargs: UniformCatBoostModel(),
    )
    monkeypatch.setattr(
        fedot_exec,
        "_nominal_multiclass_lgbm_selector",
        lambda *args, **kwargs: np.full((60, 3), 1.0 / 3.0),
    )

    output = fedot_exec._run_nominal_multiclass_catboost(
        train_features=features,
        train_target=np.arange(300) % 3,
        test_features=features.iloc[:9],
        test_target=np.arange(9) % 3,
        config=config,
        observed_encoded_labels=np.array([0, 1, 2]),
        encoded_class_count=3,
    )

    assert output is None


def test_sparse_native_linear_regime_uses_structural_and_support_guards():
    features = sparse.csr_matrix((3_000, 12_000), dtype=np.float32)
    target = np.arange(3_000) % 6

    assert fedot_exec._is_sparse_native_linear_regime(features, target)
    assert fedot_exec._use_sparse_native_logit(
        features, target, metric="logloss", framework_params={"_portfolio": True}
    )
    assert not fedot_exec._use_sparse_native_logit(
        features, target, metric="auc", framework_params={"_portfolio": True}
    )
    assert not fedot_exec._use_sparse_native_logit(
        features,
        target,
        metric="logloss",
        framework_params={"_portfolio": True, "_portfolio_candidates": "logit"},
    )


def test_sparse_native_linear_regime_excludes_dense_narrow_and_rare_data():
    target = np.arange(3_000) % 6

    assert not fedot_exec._is_sparse_native_linear_regime(
        np.broadcast_to(np.zeros((1, 12_000), dtype=np.float32), (3_000, 12_000)),
        target,
    )
    assert not fedot_exec._is_sparse_native_linear_regime(
        sparse.csr_matrix((3_000, 3_999), dtype=np.float32), target
    )
    rare_target = np.r_[np.zeros(2_981, dtype=int), np.ones(19, dtype=int)]
    assert not fedot_exec._is_sparse_native_linear_regime(
        sparse.csr_matrix((3_000, 12_000), dtype=np.float32), rare_target
    )


def test_sparse_native_matrix_normalises_pandas_sparse_without_densifying():
    matrix = sparse.csr_matrix(
        ([1.0, 2.0], ([0, 2], [1, 3])), shape=(3, 4), dtype=np.float32
    )
    frame = pd.DataFrame.sparse.from_spmatrix(matrix)

    restored = fedot_exec._as_scipy_sparse_matrix(frame)

    assert sparse.isspmatrix_csr(restored)
    assert restored.shape == matrix.shape
    assert fedot_exec._sparse_nonzero_count(frame) == 2
    assert np.array_equal(restored.toarray(), matrix.toarray())


def test_sparse_native_linear_path_bypasses_fedot(monkeypatch):
    constructed = []

    class FakeLogisticRegression:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def fit(self, features, target):
            assert sparse.isspmatrix_csr(features)
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (features.shape[0], 1))

    train_features = sparse.csr_matrix((1_000, 4_096), dtype=np.float32)
    train_target = np.arange(1_000) % 2
    test_features = sparse.csr_matrix((10, 4_096), dtype=np.float32)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=np.arange(10) % 2),
        encoded_class_count=2,
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=4,
        seed=42,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    monkeypatch.setattr(fedot_exec, "LogisticRegression", FakeLogisticRegression)
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert constructed == [
        {
            "C": 0.1,
            "solver": "liblinear",
            "dual": True,
            "max_iter": 2_000,
            "random_state": 42,
        }
    ]
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["probabilities"].shape == (10, 2)


def _materialized_sparse_frame(rows=600, columns=512, nonzero_columns=40):
    features = np.zeros((rows, columns), dtype=np.float32)
    features[:, :nonzero_columns] = 1.0
    return pd.DataFrame(features)


def test_materialized_sparse_tfidf_regime_has_structural_value_guards():
    features = _materialized_sparse_frame()
    target = np.arange(len(features)) % 3

    assert fedot_exec._is_materialized_sparse_tfidf_regime(features, target)
    assert fedot_exec._use_materialized_sparse_tfidf_logit(
        features, target, metric="logloss", framework_params={"_portfolio": True}
    )
    assert not fedot_exec._use_materialized_sparse_tfidf_logit(
        features, target, metric="auc", framework_params={"_portfolio": True}
    )
    assert not fedot_exec._use_materialized_sparse_tfidf_logit(
        features,
        target,
        metric="logloss",
        framework_params={"_portfolio": True, "_portfolio_candidates": "logit"},
    )
    assert not fedot_exec._is_materialized_sparse_tfidf_regime(
        _materialized_sparse_frame(nonzero_columns=52), target
    )
    negative = features.copy()
    negative.iloc[0, 0] = -1.0
    assert not fedot_exec._is_materialized_sparse_tfidf_regime(negative, target)
    missing = features.copy()
    missing.iloc[0, 0] = np.nan
    assert not fedot_exec._is_materialized_sparse_tfidf_regime(missing, target)
    assert fedot_exec._is_materialized_sparse_tfidf_regime(
        sparse.csr_matrix(features), target
    )
    assert not fedot_exec._is_materialized_sparse_tfidf_regime(
        features.iloc[:499], target[:499]
    )
    assert not fedot_exec._is_materialized_sparse_tfidf_regime(
        features.iloc[:, :511], target
    )
    rare_target = np.repeat([0, 1, 2], [300, 271, 29])
    assert not fedot_exec._is_materialized_sparse_tfidf_regime(
        features, rare_target
    )


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count"),
    [
        (5_001, 1_024, 3),
        (1_000, 2_501, 3),
        (2_000, 999, 3),
        (2_500, 2_001, 3),
        (600, 512, 2),
        (780, 512, 26),
        (1_800, 2_500, 25),
    ],
)
def test_materialized_sparse_tfidf_regime_has_resource_boundaries(
    row_count, feature_count, class_count
):
    features = sparse.csr_matrix((row_count, feature_count), dtype=np.float32)
    target = np.arange(row_count) % class_count

    assert not fedot_exec._is_materialized_sparse_tfidf_regime(features, target)


def test_materialized_sparse_matrix_conversion_and_regularisation_policy():
    features = _materialized_sparse_frame(rows=3, columns=4, nonzero_columns=1)

    restored = fedot_exec._as_materialized_sparse_csr(features)

    assert sparse.isspmatrix_csr(restored)
    assert np.array_equal(restored.toarray(), features.to_numpy())
    sparse_frame = pd.DataFrame.sparse.from_spmatrix(restored)
    sparse_restored = fedot_exec._as_materialized_sparse_csr(sparse_frame)
    assert sparse.isspmatrix_csr(sparse_restored)
    assert np.array_equal(sparse_restored.toarray(), features.to_numpy())
    assert fedot_exec._select_materialized_sparse_logit_c(
        {3.0: 0.4, 10.0: 0.392}
    ) == pytest.approx(3.0)
    assert fedot_exec._select_materialized_sparse_logit_c(
        {3.0: 0.4, 10.0: 0.388}
    ) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="non-negative"):
        fedot_exec._as_materialized_sparse_csr(np.array([[0.0, -1.0]]))
    with pytest.raises(ValueError, match="minimum regularisation gain"):
        fedot_exec._select_materialized_sparse_logit_c(
            {3.0: 0.4, 10.0: 0.3}, minimum_gain=-0.1
        )


def test_materialized_sparse_tfidf_path_bypasses_fedot(monkeypatch):
    selector_calls = []

    class FakeProbabilityModel:
        def predict_proba(self, features):
            assert sparse.isspmatrix_csr(features)
            return np.full((features.shape[0], 3), 1.0 / 3.0)

    def fit_selector(features, target, seed=42):
        assert sparse.isspmatrix_csr(features)
        selector_calls.append((features.shape, len(target), seed))
        return (
            FakeProbabilityModel(),
            1.0,
            3.0,
            {3.0: 0.4, 10.0: 0.395},
        )

    train_features = _materialized_sparse_frame()
    train_target = np.arange(len(train_features)) % 3
    test_features = train_features.iloc[:12].copy()
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=train_target),
        test=SimpleNamespace(X=test_features, y=train_target[:12]),
        encoded_class_count=3,
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=4,
        seed=17,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    monkeypatch.setattr(
        fedot_exec, "_fit_materialized_sparse_tfidf_selector", fit_selector
    )
    monkeypatch.setattr(
        fedot_exec,
        "Fedot",
        lambda **kwargs: pytest.fail("FEDOT must not be constructed"),
    )
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert selector_calls == [((600, 512), 600, 17)]
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["probabilities"] == pytest.approx(np.full((12, 3), 1.0 / 3.0))


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count"),
    [
        (19_999, 27, 100),
        (65_000, 27, 19),
        (20_000, 4, 201),
        (65_000, 65, 100),
        (70_001, 4, 100),
    ],
)
def test_large_narrow_rf_shrinkage_has_resource_and_support_boundaries(
    row_count, feature_count, class_count
):
    features = np.broadcast_to(
        np.zeros((1, feature_count), dtype=np.float32),
        (row_count, feature_count),
    )
    target = np.arange(row_count) % class_count

    assert not fedot_exec._is_well_supported_narrow_many_class_numeric(
        features, target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target
    ) == ["lgbm", "xgboost"]


def test_large_narrow_rf_shrinkage_excludes_sparse_and_categorical_tables():
    target = np.arange(20_000) % 20
    sparse_features = sparse.csr_matrix((20_000, 4), dtype=np.float32)
    categorical_features = pd.DataFrame(
        {"numeric": np.zeros(20_000), "category": ["level"] * 20_000}
    )

    assert not fedot_exec._is_well_supported_narrow_many_class_numeric(
        sparse_features, target
    )
    assert not fedot_exec._is_well_supported_narrow_many_class_numeric(
        categorical_features, target
    )


def test_other_adaptive_rf_portfolios_keep_three_hundred_trees_and_equal_weight():
    features = np.zeros((300, 300), dtype=np.float32)
    target = np.arange(300) % 10
    candidates = fedot_exec._adaptive_default_portfolio_candidates(features, target)

    assert fedot_exec._adaptive_rf_n_estimators(
        features, target, candidates
    ) == 300
    assert fedot_exec._adaptive_pair_strong_weight(
        features, target, candidates
    ) == 0.5


def test_wide_small_data_uses_holdout_when_cv_would_exceed_selector_cap():
    features = np.zeros((100, 80))
    target = np.arange(100) % 2

    capped = fedot_exec._classification_validation_splits(
        features,
        target,
        validation_fraction=0.2,
        max_train_rows=50,
        seed=42,
    )
    uncapped = fedot_exec._classification_validation_splits(
        features,
        target,
        validation_fraction=0.2,
        max_train_rows=100,
        seed=42,
    )

    assert len(capped) == 1
    assert len(capped[0][2]) == 50
    assert len(uncapped) == 3


@pytest.mark.parametrize(
    ("feature_count", "class_count", "expected_rounds"),
    [
        (6, 3, 2_000),
        (42, 3, 1_000),
        (100, 100, 300),
        (800, 7, 300),
    ],
)
def test_boosting_round_cap_uses_feature_and_class_complexity(
    feature_count, class_count, expected_rounds
):
    features = np.zeros((class_count * 2, feature_count))
    target = np.tile(np.arange(class_count), 2)

    assert fedot_exec._adaptive_boosting_rounds(features, target) == expected_rounds


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count", "expected_leaves"),
    [
        (6, 6, 3, 63),
        (20_000, 16, 2, 63),
        (20_000, 6, 3, 127),
        (19_999, 16, 4, 63),
        (20_000, 16, 4, 127),
        (19_999, 17, 4, None),
        (20_000, 17, 4, 127),
        (20_000, 64, 4, 127),
        (20_000, 65, 4, None),
        (20_000, 42, 6, None),
        (20_000, 800, 7, None),
    ],
)
def test_lgbm_leaf_capacity_uses_feature_and_class_complexity(
    row_count, feature_count, class_count, expected_leaves
):
    features = np.zeros((row_count, feature_count), dtype=np.uint8)
    target = np.arange(row_count) % class_count

    assert fedot_exec._adaptive_lgbm_num_leaves(features, target) == expected_leaves


def test_lgbm_child_rows_target_large_low_class_categorical_tables():
    categorical = pd.DataFrame(
        {
            "category": pd.Series(np.arange(20_000) % 3, dtype="category"),
            "numeric": np.arange(20_000, dtype=np.float32),
        }
    )
    four_class_target = np.arange(20_000) % 4

    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            categorical, four_class_target
        )
        == 100
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            categorical.iloc[:19_999], four_class_target[:19_999]
        )
        is None
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            categorical, np.arange(20_000) % 6
        )
        is None
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            categorical.astype(np.float32), four_class_target
        )
        is None
    )
    compact_integer_coded = pd.DataFrame(
        {
            f"feature_{index}": (
                np.arange(20_000) % (index + 3)
            ).astype(np.uint8)
            for index in range(6)
        }
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            compact_integer_coded, four_class_target
        )
        == 100
    )
    mostly_wide_integer_coded = compact_integer_coded.astype(np.uint16)
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            mostly_wide_integer_coded, four_class_target
        )
        is None
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            categorical, four_class_target, configured="17"
        )
        == 17
    )


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count", "expected"),
    [
        (20_000, 64, 6, 100),
        (20_000, 256, 19, 100),
        (19_999, 64, 6, None),
        (20_000, 63, 6, None),
        (20_000, 257, 6, None),
        (20_000, 64, 5, None),
        (20_000, 64, 20, None),
    ],
)
def test_lgbm_child_rows_require_supported_medium_many_class_numeric_geometry(
    row_count, feature_count, class_count, expected
):
    features = np.broadcast_to(
        np.zeros((1, feature_count), dtype=np.float32),
        (row_count, feature_count),
    )
    target = np.arange(row_count) % class_count

    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(features, target) == expected
    )


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count", "expected"),
    [
        (20_000, 8, 3, 100),
        (20_000, 57, 5, 100),
        (19_999, 8, 3, None),
        (20_000, 7, 3, None),
        (20_000, 58, 5, None),
        (20_000, 42, 2, None),
        (20_000, 42, 6, None),
    ],
)
def test_lgbm_child_rows_require_supported_large_low_class_numeric_geometry(
    row_count, feature_count, class_count, expected
):
    features = np.broadcast_to(
        np.zeros((1, feature_count), dtype=np.float32),
        (row_count, feature_count),
    )
    target = np.arange(row_count) % class_count

    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(features, target) == expected
    )


def test_large_low_class_numeric_leaf_support_excludes_rare_and_sparse_tables():
    dense_features = np.broadcast_to(
        np.zeros((1, 32), dtype=np.float32),
        (20_000, 32),
    )
    rare_target = np.concatenate((np.zeros(18_002), np.ones(999), np.full(999, 2)))

    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(dense_features, rare_target)
        is None
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            sparse.csr_matrix(dense_features), np.arange(20_000) % 4
        )
        is None
    )


def test_large_supported_mid_class_categorical_policy_has_structural_boundaries():
    row_count = 100_000
    categorical = pd.DataFrame(
        {
            f"feature_{index}": pd.Categorical(np.arange(row_count) % 4)
            for index in range(34)
        }
    )
    target = np.arange(row_count) % 7
    rare_target = target.copy()
    rare_target[target == 6] = 0
    rare_target[:999] = 6

    assert fedot_exec._is_large_supported_mid_class_categorical(
        categorical, target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        categorical, target
    ) == ["xgboost"]
    assert fedot_exec._adaptive_xgboost_min_child_weight(
        categorical, target
    ) == pytest.approx(5.0)
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        categorical.iloc[:99_999], target[:99_999]
    )
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        categorical.iloc[:, :15], target
    )
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        pd.DataFrame(np.zeros((row_count, 65), dtype=np.uint8)), target
    )
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        categorical, np.arange(row_count) % 5
    )
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        categorical, np.arange(row_count) % 10
    )
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        categorical, rare_target
    )
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        np.zeros((row_count, 34), dtype=np.float32), target
    )
    assert not fedot_exec._is_large_supported_mid_class_categorical(
        pd.DataFrame(np.zeros((row_count, 34), dtype=np.float32)), target
    )
    assert fedot_exec._adaptive_default_portfolio_candidates(
        categorical, target, configured="lgbm,rf"
    ) == ["lgbm", "rf"]


def test_large_supported_mid_class_narrow_table_keeps_default_xgboost_child_weight():
    row_count = 100_000
    categorical = pd.DataFrame(
        {
            f"feature_{index}": pd.Categorical(np.arange(row_count) % 4)
            for index in range(19)
        }
    )
    target = np.arange(row_count) % 7

    assert fedot_exec._adaptive_default_portfolio_candidates(
        categorical, target
    ) == ["xgboost"]
    assert fedot_exec._adaptive_xgboost_min_child_weight(categorical, target) is None


def test_large_supported_mid_class_child_weight_starts_at_32_features():
    row_count = 100_000
    compact_integer_features = pd.DataFrame(
        np.zeros((row_count, 32), dtype=np.uint8)
    )
    target = np.arange(row_count) % 7

    assert fedot_exec._is_large_supported_mid_class_categorical(
        compact_integer_features, target
    )
    assert (
        fedot_exec._adaptive_xgboost_min_child_weight(
            compact_integer_features.iloc[:, :31], target
        )
        is None
    )
    assert fedot_exec._adaptive_xgboost_min_child_weight(
        compact_integer_features, target
    ) == pytest.approx(5.0)


def test_large_supported_mid_class_wide_table_keeps_high_early_stopping_ceiling():
    row_count = 100_000
    compact_integer_features = pd.DataFrame(
        np.zeros((row_count, 54), dtype=np.uint8)
    )
    continuous_features = pd.DataFrame(
        np.zeros((row_count, 54), dtype=np.float32)
    )
    target = np.arange(row_count) % 7

    assert fedot_exec._adaptive_boosting_rounds(
        compact_integer_features, target
    ) == 1_400
    assert fedot_exec._adaptive_boosting_rounds(
        continuous_features, target
    ) == 300
    assert fedot_exec._adaptive_boosting_rounds(
        compact_integer_features, target, configured="321"
    ) == 321


def test_large_supported_mid_class_boosting_ceiling_respects_dense_cell_budget():
    feature_count = 64
    dtypes = [np.dtype(np.uint8)] * feature_count
    at_budget_rows = 500_000
    over_budget_rows = at_budget_rows + 1
    at_budget_features = SimpleNamespace(
        shape=(at_budget_rows, feature_count), dtypes=dtypes
    )
    over_budget_features = SimpleNamespace(
        shape=(over_budget_rows, feature_count), dtypes=dtypes
    )
    at_budget_target = np.arange(at_budget_rows) % 7
    over_budget_target = np.arange(over_budget_rows) % 7

    assert fedot_exec._is_large_supported_mid_class_categorical(
        at_budget_features, at_budget_target
    )
    assert fedot_exec._is_large_supported_mid_class_categorical(
        over_budget_features, over_budget_target
    )
    assert fedot_exec._adaptive_boosting_rounds(
        at_budget_features, at_budget_target
    ) == 1_400
    assert fedot_exec._adaptive_boosting_rounds(
        over_budget_features, over_budget_target
    ) == 1_000


@pytest.mark.parametrize(
    ("selector_temperature", "expected_multiplier"),
    [
        (0.95, 1.0),
        (1.0, 1.0),
        (1.01, 1.0 / 1.01),
        (1.10, 0.97),
    ],
)
def test_large_supported_mid_class_xgboost_temperature_shrinks_towards_one(
    selector_temperature, expected_multiplier
):
    row_count = 100_000
    features = pd.DataFrame(np.zeros((row_count, 32), dtype=np.uint8))
    target = np.arange(row_count) % 7

    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features,
        target,
        selected_models=["xgboost"],
        selector_temperature=selector_temperature,
        metric="logloss",
        calibrate=True,
        direct_all_row_refit=False,
        adaptive_portfolio=True,
    ) == pytest.approx(expected_multiplier)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("selected_models", ["lgbm"]),
        ("metric", "auc"),
        ("calibrate", False),
        ("adaptive_portfolio", False),
    ],
)
def test_large_supported_mid_class_xgboost_temperature_preserves_other_paths(
    override, value
):
    row_count = 100_000
    features = pd.DataFrame(np.zeros((row_count, 32), dtype=np.uint8))
    target = np.arange(row_count) % 7
    kwargs = {
        "selected_models": ["xgboost"],
        "selector_temperature": 1.10,
        "metric": "logloss",
        "calibrate": True,
        "direct_all_row_refit": False,
        "adaptive_portfolio": True,
    }
    kwargs[override] = value

    assert (
        fedot_exec._adaptive_deployment_temperature_multiplier(
            features, target, **kwargs
        )
        == 1.0
    )


def test_large_supported_mid_class_temperature_requires_wide_child_support_regime():
    row_count = 100_000
    features = pd.DataFrame(np.zeros((row_count, 32), dtype=np.uint8))
    target = np.arange(row_count) % 7

    assert fedot_exec._adaptive_deployment_temperature_multiplier(
        features.iloc[:, :31],
        target,
        selected_models=["xgboost"],
        selector_temperature=1.10,
        metric="logloss",
        calibrate=True,
        direct_all_row_refit=False,
        adaptive_portfolio=True,
    ) == 1.0


def test_numeric_leaf_support_excludes_sparse_and_rare_class_tables():
    dense_features = np.broadcast_to(
        np.zeros((1, 93), dtype=np.float32),
        (20_000, 93),
    )
    rare_target = np.concatenate((np.zeros(19_000), np.arange(1, 11).repeat(100)))

    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(dense_features, rare_target)
        is None
    )
    assert (
        fedot_exec._adaptive_lgbm_min_child_samples(
            sparse.csr_matrix(dense_features), np.arange(20_000) % 9
        )
        is None
    )


@pytest.mark.parametrize(
    ("feature_count", "class_count", "expected_rate", "expected_depth"),
    [
        (27, 100, 0.1, 5),
        (800, 7, None, 4),
        (42, 3, None, None),
        (60, 355, None, 4),
    ],
)
def test_xgboost_learning_rate_targets_tractable_many_class_problems(
    feature_count, class_count, expected_rate, expected_depth
):
    features = np.zeros((class_count * 2, feature_count))
    target = np.tile(np.arange(class_count), 2)

    assert fedot_exec._adaptive_xgboost_learning_rate(features, target) == expected_rate
    assert fedot_exec._adaptive_xgboost_max_depth(features, target) == expected_depth


def test_xgboost_row_sampling_targets_large_narrow_many_class_problems():
    target = np.arange(20_000) % 100
    features = np.broadcast_to(np.zeros((1, 27)), (20_000, 27))

    assert fedot_exec._adaptive_xgboost_subsample(features, target) == 0.9
    assert fedot_exec._adaptive_xgboost_subsample(
        features[:19_999], target[:19_999]
    ) is None
    assert fedot_exec._adaptive_xgboost_subsample(
        features, target % 19
    ) is None
    assert fedot_exec._adaptive_xgboost_subsample(
        np.broadcast_to(np.zeros((1, 52)), (20_000, 52)), target
    ) is None


def test_shallow_xgboost_probe_targets_large_numeric_holdout_reuse_geometry():
    features = np.zeros((100_000, 16), dtype=np.uint8)
    target = np.arange(100_000) % 10

    assert fedot_exec._use_shallow_xgboost_probe(
        features,
        target,
        metric="logloss",
        validation_fold_count=1,
        adaptive_depth=None,
    )
    assert not fedot_exec._use_shallow_xgboost_probe(
        features,
        target,
        metric="logloss",
        validation_fold_count=1,
        adaptive_depth=5,
    )
    assert not fedot_exec._use_shallow_xgboost_probe(
        pd.DataFrame(features).assign(category="level"),
        target,
        metric="logloss",
        validation_fold_count=1,
        adaptive_depth=None,
    )
    assert not fedot_exec._use_shallow_xgboost_probe(
        features,
        target,
        metric="auc",
        validation_fold_count=1,
        adaptive_depth=None,
    )


def test_shallow_xgboost_probe_requires_material_holdout_gain():
    assert fedot_exec._materially_better_shallow_score(-0.50, -0.60)
    assert not fedot_exec._materially_better_shallow_score(-0.595, -0.60)
    assert not fedot_exec._materially_better_shallow_score(-0.0030, -0.0040)


def test_xgboost_child_weight_probe_targets_bounded_manyclass_geometry():
    features = np.zeros((20_000, 20), dtype=np.float32)
    target = np.arange(20_000) % 50
    kwargs = {
        "metric": "logloss",
        "validation_fold_count": 1,
        "adaptive_min_child_weight": 3.0,
    }

    assert fedot_exec._use_xgboost_child_weight_probe(features, target, **kwargs)
    assert not fedot_exec._use_xgboost_child_weight_probe(
        features,
        target,
        **{**kwargs, "metric": "auc"},
    )
    assert not fedot_exec._use_xgboost_child_weight_probe(
        features,
        target,
        **{**kwargs, "validation_fold_count": 3},
    )
    assert not fedot_exec._use_xgboost_child_weight_probe(
        features,
        target,
        **{**kwargs, "adaptive_min_child_weight": 10.0},
    )


def test_xgboost_child_weight_probe_requires_practical_paired_gain():
    truth = np.array([0, 1, 0, 1])
    default = np.array(
        [[0.70, 0.30], [0.30, 0.70], [0.60, 0.40], [0.40, 0.60]]
    )
    clearly_better = np.array(
        [[0.90, 0.10], [0.10, 0.90], [0.85, 0.15], [0.15, 0.85]]
    )
    marginally_better = default + np.array(
        [[0.0001, -0.0001], [-0.0001, 0.0001]] * 2
    )

    assert fedot_exec._materially_better_xgboost_regularisation(
        -0.15, -0.40, clearly_better, default, truth, labels=np.array([0, 1])
    )
    assert not fedot_exec._materially_better_xgboost_regularisation(
        -0.3999,
        -0.40,
        marginally_better,
        default,
        truth,
        labels=np.array([0, 1]),
    )


@pytest.mark.parametrize(
    ("row_count", "feature_count", "class_count", "expected_weight"),
    [
        (20_000, 20, 50, 3.0),
        (20_000, 180, 10, 3.0),
        (19_999, 20, 50, None),
        (20_000, 20, 9, None),
        (20_000, 20, 201, None),
        (30_000, 91, 50, None),
    ],
)
def test_xgboost_child_weight_needs_many_supported_classes_and_tractable_width(
    row_count, feature_count, class_count, expected_weight
):
    features = np.zeros((row_count, feature_count), dtype=np.float32)
    target = np.arange(row_count) % class_count

    assert (
        fedot_exec._adaptive_xgboost_min_child_weight(features, target)
        == expected_weight
    )


def test_candidate_model_params_are_isolated_by_booster():
    assert fedot_exec._candidate_model_params(
        "lgbm",
        lgbm_num_leaves=63,
        lgbm_min_child_samples=5,
        lgbm_min_child_weight=1.0,
        xgboost_learning_rate=0.1,
        xgboost_max_depth=4,
        xgboost_max_bin=64,
        xgboost_colsample_bytree=0.8,
        xgboost_subsample=0.8,
        xgboost_min_child_weight=0.5,
        logit_c=0.3,
        rf_n_estimators=300,
    ) == {
        "num_leaves": 63,
        "min_child_samples": 5,
        "min_child_weight": 1.0,
    }
    assert fedot_exec._candidate_model_params(
        "xgboost",
        lgbm_num_leaves=63,
        lgbm_min_child_samples=5,
        lgbm_min_child_weight=1.0,
        xgboost_learning_rate=0.1,
        xgboost_max_depth=4,
        xgboost_max_bin=64,
        xgboost_colsample_bytree=0.8,
        xgboost_subsample=0.8,
        xgboost_min_child_weight=0.5,
        logit_c=0.3,
        rf_n_estimators=300,
    ) == {
        "learning_rate": 0.1,
        "max_depth": 4,
        "max_bin": 64,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "min_child_weight": 0.5,
    }
    assert fedot_exec._candidate_model_params(
        "auto",
        lgbm_num_leaves=63,
        lgbm_min_child_samples=5,
        lgbm_min_child_weight=1.0,
        xgboost_learning_rate=0.1,
        xgboost_max_depth=4,
        xgboost_max_bin=64,
        xgboost_colsample_bytree=0.8,
        xgboost_subsample=0.8,
        xgboost_min_child_weight=0.5,
        logit_c=0.3,
        rf_n_estimators=300,
    ) == {}
    assert fedot_exec._candidate_model_params("logit", logit_c=0.3) == {"C": 0.3}
    assert fedot_exec._candidate_model_params(
        "rf", rf_n_estimators=300
    ) == {"n_estimators": 300}
    assert fedot_exec._candidate_model_params(
        "rf_large_subspace", rf_n_estimators=300
    ) == {"n_estimators": 300, "max_features": 0.1}
    assert fedot_exec._candidate_model_params("scaled_logit") == {
        "C": 0.001,
        "solver": "liblinear",
        "dual": True,
        "max_iter": 2_000,
    }
    assert fedot_exec._candidate_model_params(
        "scaled_logit", logit_c=0.01
    )["C"] == 0.01
    assert fedot_exec._candidate_model_params("scaled_svc") == {
        "C": 3.0,
        "gamma": "scale",
        "probability": True,
    }
    assert fedot_exec._candidate_model_params("scaled_svc_strong_half") == {
        "C": 30.0,
        "gamma": "scale",
        "probability": True,
        "_gamma_multiplier": 0.5,
    }
    assert fedot_exec._candidate_model_params("scaled_svc_strong_scale") == {
        "C": 30.0,
        "gamma": "scale",
        "probability": True,
    }
    assert fedot_exec._candidate_model_params("mixed_logit") == {
        "C": 3.0,
        "max_iter": 2_000,
    }
    assert fedot_exec._candidate_model_params("mixed_svc") == {
        "C": 3.0,
        "gamma": "scale",
        "probability": True,
    }
    assert fedot_exec._candidate_model_params("extra_trees") == {
        "n_estimators": 500,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    assert fedot_exec._candidate_model_params("extra_trees_wide") == {
        "n_estimators": 500,
        "min_samples_leaf": 1,
        "max_features": 0.5,
    }
    assert fedot_exec._candidate_model_params("mixed_extra_trees") == {
        "n_estimators": 500,
        "min_samples_leaf": 8,
        "max_features": 1.0,
    }
    assert fedot_exec._candidate_model_params("relational_hist") == {
        "max_iter": 250,
        "learning_rate": 0.1,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    }
    assert fedot_exec._candidate_model_params(
        "relational_hist_second_order"
    ) == fedot_exec._candidate_model_params("relational_hist")


def test_scaled_svc_candidate_exposes_aligned_probability_api():
    features = np.array(
        [[0.0, np.nan], [0.1, 0.2], [0.9, 0.8], [1.0, 1.1]]
    )
    target = np.array([0, 0, 1, 1])

    model = fedot_exec._fit_scaled_svc_candidate(
        features,
        target,
        model_params={"C": 1.0, "probability": True},
        seed=7,
    )
    probabilities = model.predict_proba(
        features=features, probs_for_all_classes=True
    )

    assert probabilities.shape == (4, 2)

    strong_model = fedot_exec._fit_scaled_svc_candidate(
        features,
        target,
        model_params={
            "C": 30.0,
            "probability": True,
            "_gamma_multiplier": 0.5,
        },
        seed=7,
    )
    assert strong_model.estimator[-1].get_params()["gamma"] == pytest.approx(0.25)


def test_extra_trees_candidate_exposes_aligned_probability_api():
    features = np.array(
        [[0.0, np.nan], [0.1, 0.2], [0.9, 0.8], [1.0, 1.1]]
    )
    target = np.array([0, 0, 1, 1])

    model = fedot_exec._fit_extra_trees_candidate(
        features,
        target,
        model_params={"n_estimators": 10},
        seed=7,
        n_jobs=2,
    )
    probabilities = model.predict_proba(
        features=features, probs_for_all_classes=True
    )

    assert probabilities.shape == (4, 2)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(4))
    assert np.array_equal(model.target, target)
    assert model.current_pipeline.length == 1


def test_relational_feature_expansion_adds_signed_absolute_and_equal_relations():
    features = np.array([[1, 3, 1], [4, 2, 4]], dtype=np.uint8)

    expanded = fedot_exec._relational_feature_expansion(features)

    assert expanded.dtype == np.float32
    assert expanded == pytest.approx(
        np.array(
            [
                [1, 3, 1, -2, 2, 0, 0, 0, 1, 2, 2, 0],
                [4, 2, 4, 2, 2, 0, 0, 0, 1, -2, 2, 0],
            ],
            dtype=np.float32,
        )
    )


def test_relational_second_order_expansion_compares_pairwise_distances():
    features = np.array([[1, 3, 1], [4, 2, 4]], dtype=np.uint8)

    expanded = fedot_exec._relational_second_order_feature_expansion(features)

    assert expanded.dtype == np.float32
    assert expanded.shape == (2, 15)
    assert expanded[:, -3:] == pytest.approx(
        np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
    )


def test_relational_hist_candidate_exposes_aligned_probability_api():
    features = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [4, 4, 3],
            [4, 3, 3],
            [3, 4, 4],
            [3, 3, 4],
        ],
        dtype=np.uint8,
    )
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = fedot_exec._fit_relational_hist_candidate(
        features,
        target,
        model_params={"max_iter": 5, "min_samples_leaf": 2},
        seed=7,
    )
    probabilities = model.predict_proba(
        features=features, probs_for_all_classes=True
    )

    assert probabilities.shape == (8, 2)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(8))
    assert np.array_equal(model.target, target)
    assert model.current_pipeline.length == 1


def test_direct_regression_candidate_handles_mixed_features_and_missing_values():
    features = pd.DataFrame(
        {
            "category": pd.Series(["a", "a", "b", None], dtype="category"),
            "numeric": [0.0, np.nan, 0.9, 1.0],
        }
    )
    target = np.array([0.0, 0.1, 0.9, 1.0])
    estimator = fedot_exec._make_regression_portfolio_estimator(
        "lgbmreg_direct",
        features,
        seed=7,
        n_jobs=2,
        small_mixed=True,
    )

    estimator.fit(features, target)
    model = fedot_exec._SklearnRegressionModel(estimator, target)
    predictions = model.predict(features)

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert np.array_equal(model.target, target)
    assert model.current_pipeline.length == 1


def test_mixed_extra_trees_candidate_handles_missing_categories():
    features = pd.DataFrame(
        {
            "category": pd.Series(["a", "a", "b", None], dtype="category"),
            "numeric": [0.0, np.nan, 0.9, 1.0],
        }
    )
    target = np.array([0, 0, 1, 1])

    model = fedot_exec._fit_mixed_extra_trees_candidate(
        features,
        target,
        model_params={"n_estimators": 10, "min_samples_leaf": 1},
        seed=7,
        n_jobs=2,
    )
    probabilities = model.predict_proba(
        features=features, probs_for_all_classes=True
    )

    assert probabilities.shape == (4, 2)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(4))
    assert np.array_equal(model.target, target)
    assert model.current_pipeline.length == 1


def test_mixed_logit_candidate_handles_missing_and_unknown_categories():
    features = pd.DataFrame(
        {
            "category": pd.Series(["a", "a", "b", None], dtype="category"),
            "numeric": [0.0, np.nan, 0.9, 1.0],
        }
    )
    target = np.array([0, 0, 1, 1])

    model = fedot_exec._fit_mixed_logit_candidate(
        features,
        target,
        model_params={"C": 1.0},
        seed=7,
    )
    unseen = pd.DataFrame(
        {
            "category": pd.Series(["c"], dtype="category"),
            "numeric": [0.5],
        }
    )
    probabilities = model.predict_proba(
        features=unseen, probs_for_all_classes=True
    )

    assert probabilities.shape == (1, 2)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(1))
    assert np.array_equal(model.target, target)
    assert model.current_pipeline.length == 1


def test_mixed_svc_candidate_handles_missing_and_unknown_categories():
    features = pd.DataFrame(
        {
            "category": pd.Series(["a", "a", "b", None], dtype="category"),
            "numeric": [0.0, np.nan, 0.9, 1.0],
        }
    )
    target = np.array([0, 0, 1, 1])

    model = fedot_exec._fit_mixed_svc_candidate(
        features,
        target,
        model_params={"C": 1.0},
        seed=7,
    )
    unseen = pd.DataFrame(
        {
            "category": pd.Series(["c"], dtype="category"),
            "numeric": [0.5],
        }
    )
    probabilities = model.predict_proba(
        features=unseen, probs_for_all_classes=True
    )

    assert probabilities.shape == (1, 2)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(1))
    assert np.array_equal(model.target, target)
    assert model.current_pipeline.length == 1


def test_mixed_svc_guard_accepts_narrow_categorical_but_not_narrow_mixed():
    row_count = 1_200
    target = np.arange(row_count) % 4
    categorical = pd.DataFrame(
        {
            f"category_{index}": pd.Series(
                np.arange(row_count) % 4, dtype="category"
            )
            for index in range(6)
        }
    )
    narrow_mixed = categorical.copy()
    narrow_mixed["numeric"] = np.arange(row_count, dtype=np.float32)
    ambiguous_width = pd.concat(
        [categorical, categorical.add_prefix("copy_")], axis=1
    )
    rare_target = target.copy()
    rare_target[:49] = 4

    assert fedot_exec._is_bounded_mixed_svc_candidate(categorical, target)
    assert not fedot_exec._is_bounded_mixed_svc_candidate(narrow_mixed, target)
    assert not fedot_exec._is_bounded_mixed_svc_candidate(
        ambiguous_width, target
    )
    assert not fedot_exec._is_bounded_mixed_svc_candidate(
        categorical, rare_target
    )


def test_small_supported_mixed_binary_svc_is_auc_only_and_resource_bounded():
    row_count = 900
    features = pd.DataFrame(
        {
            **{
                f"categorical_{index}": pd.Series(
                    np.arange(row_count) % (index + 3), dtype="category"
                )
                for index in range(4)
            },
            **{
                f"numeric_{index}": np.arange(row_count, dtype=np.float32)
                for index in range(16)
            },
        }
    )
    target = (np.arange(row_count) % 4 == 0).astype(int)
    rare_target = np.zeros(row_count, dtype=int)
    rare_target[:99] = 1

    assert fedot_exec._is_small_supported_mixed_binary_svc(features, target)
    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target, metric="auc"
    ) == ["lgbm", "xgboost", "mixed_svc", "auto"]
    assert fedot_exec._adaptive_default_portfolio_candidates(
        features, target, metric="logloss"
    ) == ["lgbm", "xgboost", "auto"]
    assert fedot_exec._adaptive_pair_strong_weight(
        features, target, ["lgbm", "xgboost", "mixed_svc", "auto"]
    ) == 0.5
    assert not fedot_exec._is_small_supported_mixed_binary_svc(
        features.iloc[:799], target[:799]
    )
    assert not fedot_exec._is_small_supported_mixed_binary_svc(
        features, rare_target
    )
    assert not fedot_exec._is_small_supported_mixed_binary_svc(
        features.astype(np.float32), target
    )
    assert not fedot_exec._is_small_supported_mixed_binary_svc(
        features.iloc[:, :15], target
    )
    assert not fedot_exec._is_small_supported_mixed_binary_svc(
        features, np.arange(row_count) % 3
    )


def test_predefined_booster_receives_allocated_cores():
    pipeline = fedot_exec._predefined_model_with_n_jobs(
        "xgboost",
        4,
        n_estimators=300,
        use_eval_set=False,
        model_params={"num_leaves": 63},
    )

    assert pipeline.nodes[0].parameters.get("n_jobs") == 4
    assert pipeline.nodes[0].parameters.get("n_estimators") == 300
    assert pipeline.nodes[0].parameters.get("use_eval_set") is False
    assert pipeline.nodes[0].parameters.get("early_stopping_rounds") is None
    assert pipeline.nodes[0].parameters.get("num_leaves") == 63
    catboost = fedot_exec._predefined_model_with_n_jobs(
        "catboost", 4, n_estimators=300, use_eval_set=False
    )
    assert catboost.nodes[0].parameters.get("num_trees") == 300
    assert "n_estimators" not in catboost.nodes[0].parameters
    logit = fedot_exec._predefined_model_with_n_jobs(
        "logit", 4, n_estimators=300, use_eval_set=False
    )
    assert logit.nodes[0].parameters == {}
    rf = fedot_exec._predefined_model_with_n_jobs(
        "rf", 4, model_params={"n_estimators": 300}
    )
    assert rf.nodes[0].parameters == {"n_estimators": 300, "n_jobs": 4}
    rf_large_subspace = fedot_exec._predefined_model_with_n_jobs(
        "rf_large_subspace",
        4,
        model_params={"n_estimators": 300, "max_features": 0.1},
    )
    assert rf_large_subspace.nodes[0].operation.operation_type == "rf"
    assert rf_large_subspace.nodes[0].parameters == {
        "n_estimators": 300,
        "max_features": 0.1,
        "n_jobs": 4,
    }
    scaled_logit = fedot_exec._predefined_model_with_n_jobs(
        "scaled_logit",
        4,
        model_params={"C": 0.001, "solver": "liblinear", "dual": True},
    )
    assert scaled_logit.root_node.operation.operation_type == "logit"
    assert scaled_logit.root_node.parameters == {
        "C": 0.001,
        "solver": "liblinear",
        "dual": True,
    }
    assert scaled_logit.root_node.nodes_from[0].operation.operation_type == "scaling"
    assert fedot_exec._predefined_model_with_n_jobs("auto", 4) == "auto"


@pytest.mark.parametrize(
    ("attribute", "value", "expected"),
    [("best_iteration_", 73, 73), ("best_iteration", 72, 73)],
)
def test_best_boosting_rounds_handles_library_iteration_conventions(
    attribute, value, expected
):
    library_model = SimpleNamespace(**{attribute: value})
    implementation = SimpleNamespace(model=library_model)
    node = SimpleNamespace(fitted_operation=implementation)
    automl = SimpleNamespace(current_pipeline=SimpleNamespace(nodes=[node]))

    assert fedot_exec._best_boosting_rounds(automl, fallback=300) == expected


def test_boosting_rounds_are_extrapolated_from_selector_early_stopping():
    def validation_model(best_iteration):
        implementation = SimpleNamespace(
            model=SimpleNamespace(best_iteration_=best_iteration)
        )
        node = SimpleNamespace(fitted_operation=implementation)
        return SimpleNamespace(current_pipeline=SimpleNamespace(nodes=[node]))

    models = [validation_model(31), validation_model(36), validation_model(41)]

    assert fedot_exec._extrapolated_boosting_rounds(
        models,
        full_train_rows=60_000,
        selector_train_rows=2_500,
        fallback=300,
    ) == 150
    assert fedot_exec._extrapolated_boosting_rounds(
        models,
        full_train_rows=60_000,
        selector_train_rows=2_500,
        fallback=100,
    ) == 100
    assert fedot_exec._extrapolated_boosting_rounds(
        [],
        full_train_rows=60_000,
        selector_train_rows=2_500,
        fallback=300,
    ) is None


def test_direct_refit_estimate_only_relaxes_raw_lgbm_budget():
    def validation_model(best_iteration):
        implementation = SimpleNamespace(
            model=SimpleNamespace(best_iteration_=best_iteration)
        )
        node = SimpleNamespace(fitted_operation=implementation)
        return SimpleNamespace(current_pipeline=SimpleNamespace(nodes=[node]))

    validation_models = [validation_model(40)]
    base = {
        "duration": 10.0,
        "validation_models": validation_models,
    }

    assert fedot_exec._direct_refit_budget_estimate(
        {**base, "model": "lgbm"},
        direct_rounds=100,
        fold_count=1,
        conservative_fallback=100.0,
    ) == pytest.approx(43.75)
    assert fedot_exec._direct_refit_budget_estimate(
        {**base, "model": "xgboost"},
        direct_rounds=100,
        fold_count=1,
        conservative_fallback=100.0,
    ) == 100.0
    assert fedot_exec._direct_refit_budget_estimate(
        {**base, "model": "lgbm"},
        direct_rounds=None,
        fold_count=1,
        conservative_fallback=100.0,
    ) == 100.0


def test_refittable_primary_takes_precedence_over_selector_only_winner():
    lgbm = {"model": "lgbm", "score": -0.4}
    xgboost = {"model": "xgboost", "score": -0.3}

    primary, validation_primary = fedot_exec._select_refittable_primary(
        [lgbm, xgboost],
        {"lgbm": 60.0, "xgboost": 160.0},
        elapsed_seconds=30.0,
        runtime_seconds=180.0,
        prediction_reserve=8.0,
        refit_start_safety_reserve=14.4,
    )

    assert validation_primary is xgboost
    assert primary is lgbm


def test_refit_budget_keeps_headroom_for_estimation_error():
    assert not fedot_exec._refit_fits_budget(
        estimated_refit_seconds=150.0,
        elapsed_seconds=8.0,
        runtime_seconds=180.0,
        prediction_reserve=8.0,
        refit_start_safety_reserve=14.4,
    )
    assert fedot_exec._refit_fits_budget(
        estimated_refit_seconds=123.0,
        elapsed_seconds=26.0,
        runtime_seconds=180.0,
        prediction_reserve=8.0,
        refit_start_safety_reserve=14.4,
    )


def test_full_candidate_can_disable_redundant_input_preprocessing(monkeypatch):
    make_calls = []

    class FakeFedot:
        def fit(self, **kwargs):
            self.fit_kwargs = kwargs

    def fake_make_fedot(**kwargs):
        make_calls.append(kwargs)
        return FakeFedot()

    monkeypatch.setattr(fedot_exec, "_make_fedot", fake_make_fedot)
    monkeypatch.setattr(
        fedot_exec,
        "_predefined_model_with_n_jobs",
        lambda *args, **kwargs: "model",
    )

    fedot_exec._fit_full_candidate(
        contender={"model": "lgbm", "model_params": {}},
        train_features=np.zeros((4, 2)),
        train_target=np.arange(4) % 2,
        config=SimpleNamespace(cores=2),
        scoring_metric="neg_log_loss",
        training_params={"preset": "best_quality"},
        runtime_min=1.0,
        max_pipeline_fit_time=0.1,
        boosting_rounds=20,
        use_input_preprocessing=False,
    )

    assert make_calls[0]["training_params"] == {
        "preset": "best_quality",
        "use_input_preprocessing": False,
    }


@pytest.mark.parametrize(
    (
        "configured",
        "full_data_refit",
        "full_rows",
        "selector_rows",
        "feature_count",
        "expected",
    ),
    [
        (None, True, 40_000, 2_500, 256, True),
        (None, True, 39_999, 2_500, 256, False),
        (None, True, 40_000, 2_500, 255, False),
        (None, False, 40_000, 2_500, 256, False),
        (False, True, 40_000, 2_500, 256, False),
        (True, True, 5_000, 2_500, 10, True),
        (True, False, 40_000, 2_500, 256, False),
    ],
)
def test_direct_all_row_refit_is_adaptive_or_explicit(
    configured,
    full_data_refit,
    full_rows,
    selector_rows,
    feature_count,
    expected,
):
    assert (
        fedot_exec._use_direct_all_row_refit(
            configured,
            full_data_refit,
            full_rows,
            selector_rows,
            feature_count,
            all_features_are_numeric=True,
        )
        is expected
    )

    assert not fedot_exec._use_direct_all_row_refit(
        configured=True,
        full_data_refit=True,
        full_train_rows=40_000,
        selector_train_rows=2_500,
        feature_count=256,
        all_features_are_numeric=False,
    )


def test_numeric_feature_detection_rejects_mixed_tables():
    assert fedot_exec._all_features_are_numeric(np.zeros((2, 2), dtype=np.float32))
    assert fedot_exec._all_features_are_numeric(
        pd.DataFrame({"a": [1.0], "b": [2]})
    )
    assert not fedot_exec._all_features_are_numeric(
        pd.DataFrame({"a": [1.0], "b": ["category"]})
    )


def test_sampled_numeric_density_distinguishes_sparse_and_dense_tables():
    sparse = np.zeros((100, 10), dtype=np.float32)
    sparse[:, :2] = 1.0
    dense = np.ones((100, 10), dtype=np.float32)

    assert fedot_exec._sampled_numeric_density(sparse, max_cells=100) == 0.2
    assert fedot_exec._sampled_numeric_density(dense, max_cells=100) == 1.0
    assert fedot_exec._sampled_numeric_density(
        pd.DataFrame({"numeric": [1.0], "category": ["a"]})
    ) is None
    with pytest.raises(ValueError, match="sample size"):
        fedot_exec._sampled_numeric_density(dense, max_cells=0)


@pytest.mark.parametrize(
    ("configured", "density", "expected"),
    [
        (None, None, 0.45),
        (None, 0.299, 0.45),
        (None, 0.30, 0.48),
        (None, 0.50, 0.48),
        (0.41, 0.50, 0.41),
    ],
)
def test_direct_round_exponent_is_density_adaptive_or_explicit(
    configured, density, expected
):
    assert fedot_exec._adaptive_direct_round_exponent(configured, density) == expected


@pytest.mark.parametrize(
    ("additional_reserve", "expected_refitted"),
    [(0.0, True), (25.0, False)],
)
def test_all_row_refit_preserves_budget_for_remaining_ensemble_models(
    monkeypatch, additional_reserve, expected_refitted
):
    fitted_model = SimpleNamespace(
        current_pipeline=SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    fitted_operation=SimpleNamespace(
                        model=SimpleNamespace(best_iteration_=40)
                    )
                )
            ]
        )
    )
    refitted_model = object()
    refit_calls = []

    def fake_refit(*args, **kwargs):
        refit_calls.append((args, kwargs))
        return refitted_model

    monkeypatch.setattr(fedot_exec.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(fedot_exec, "_fit_full_candidate", fake_refit)

    output = fedot_exec._maybe_refit_on_all_rows(
        fitted_model=fitted_model,
        contender={"model": "lgbm"},
        train_features=np.zeros((100, 2)),
        train_target=np.arange(100) % 2,
        config=SimpleNamespace(max_runtime_seconds=50),
        scoring_metric="neg_log_loss",
        training_params={},
        runtime_min=1.0,
        max_pipeline_fit_time=0.1,
        boosting_rounds=300,
        observed_fit_seconds=10.0,
        started_at=0.0,
        prediction_reserve=5.0,
        enabled=True,
        additional_budget_reserve=additional_reserve,
    )

    assert bool(refit_calls) is expected_refitted
    assert output is (refitted_model if expected_refitted else fitted_model)


def test_local_image_features_pool_and_append_gradients():
    pixels = np.arange(16, dtype=np.float32).reshape(1, -1)

    transformed = fedot_exec._local_image_features(pixels, image_side=4)

    assert transformed.shape == (1, 8)
    assert transformed[0].tolist() == pytest.approx(
        [2.5, 4.5, 10.5, 12.5, 2.0, 2.0, 8.0, 8.0]
    )


def test_local_image_refit_requires_square_train_locality_and_defaults_only():
    axis = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    smooth_image = axis[:, None] + axis[None, :]
    smooth = np.broadcast_to(smooth_image, (20_000, 24, 24)).reshape(
        20_000, -1
    )
    random_image = np.random.default_rng(7).random((24, 24), dtype=np.float32)
    random_table = np.broadcast_to(random_image, (20_000, 24, 24)).reshape(
        20_000, -1
    )
    target = np.arange(20_000) % 5
    common = {
        "target": target,
        "metric": "logloss",
        "bounded_refit_seconds": 90.0,
        "direct_rounds": 120,
    }

    assert fedot_exec._local_image_refit_side(
        smooth, framework_params={"_portfolio": True}, **common
    ) == 24
    assert fedot_exec._local_image_refit_side(
        random_table, framework_params={"_portfolio": True}, **common
    ) is None
    assert fedot_exec._local_image_refit_side(
        smooth,
        framework_params={"_portfolio": True, "_portfolio_boosting_rounds": 120},
        **common,
    ) is None


def test_local_image_probability_model_transforms_and_calibrates_predictions():
    observed = {}

    class DummyModel:
        target = np.array([0, 1])
        current_pipeline = SimpleNamespace(length=1)
        history = None

        def predict_proba(self, features, probs_for_all_classes=True):
            observed["features"] = features
            observed["all_classes"] = probs_for_all_classes
            return np.tile([0.25, 0.75], (len(features), 1))

    wrapped = fedot_exec._LocalImageProbabilityModel(
        DummyModel(), image_side=4, temperature=2.0
    )
    probabilities = wrapped.predict_proba(
        np.arange(32, dtype=np.float32).reshape(2, 16),
        probs_for_all_classes=False,
    )

    assert observed["features"].shape == (2, 8)
    assert observed["all_classes"] is False
    assert probabilities.shape == (2, 2)
    assert probabilities[0, 1] < 0.75
    assert wrapped.current_pipeline.length == 1


def test_local_image_lgbm_refit_uses_its_oof_params_and_deadline():
    contenders = [
        {"model": "xgboost", "model_params": {"max_depth": 4}},
        {"model": "lgbm", "model_params": {"num_leaves": 63}},
    ]

    refit = fedot_exec._local_image_lgbm_refit_contender(contenders, 92.5)

    assert refit == {
        "model": "lgbm",
        "model_params": {"num_leaves": 63, "fit_time_limit": 92.5},
    }
    assert contenders[1]["model_params"] == {"num_leaves": 63}


def test_local_image_calibration_split_is_stratified_and_disjoint():
    target = np.arange(20_000) % 10

    fit_indices, calibration_indices = (
        fedot_exec._local_image_fit_calibration_indices(target, seed=42)
    )

    assert len(calibration_indices) == 1_000
    assert not np.intersect1d(fit_indices, calibration_indices).size
    assert set(np.unique(target[calibration_indices])) == set(range(10))


def test_compact_encoder_handles_high_cardinality_and_unknown_values():
    train = pd.DataFrame(
        {
            "numeric": [1.0, np.nan] * 300,
            "category": [f"level-{index}" for index in range(600)],
            "empty": [np.nan] * 600,
        }
    )
    test = pd.DataFrame(
        {
            "numeric": [np.nan, 3.0],
            "category": ["level-1", "unseen"],
            "empty": [np.nan, np.nan],
        }
    )

    encoded_train, encoded_test = fedot_exec._compact_encode_if_needed(train, test)

    assert encoded_train.shape == (600, 2)
    assert encoded_test.shape == (2, 2)
    assert encoded_train.dtype == encoded_test.dtype == np.float32
    assert np.isfinite(encoded_train).all()
    assert encoded_test[1, 1] == -1


def test_compact_encoder_can_be_forced_for_low_cardinality_tables():
    train = pd.DataFrame({"category": pd.Categorical(["left", "right", "left"])})
    test = pd.DataFrame({"category": pd.Categorical(["right", "unknown"])})

    unchanged_train, unchanged_test = fedot_exec._compact_encode_if_needed(
        train, test
    )
    encoded_train, encoded_test = fedot_exec._compact_encode_if_needed(
        train, test, force=True
    )

    assert unchanged_train is train
    assert unchanged_test is test
    assert encoded_train[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert encoded_test[:, 0].tolist() == [1.0, -1.0]


def test_fast_one_hot_encoder_is_train_only_and_imputes_numerics():
    train = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0],
            "category": pd.Categorical(["left", "right", "left"]),
        }
    )
    test = pd.DataFrame(
        {
            "numeric": [np.nan, 5.0],
            "category": pd.Categorical(["right", "unseen"]),
        }
    )

    encoded_train, encoded_test = fedot_exec._compact_encode_if_needed(
        train, test, one_hot=True
    )

    assert encoded_train.shape == (3, 3)
    assert encoded_test.shape == (2, 3)
    assert encoded_train.dtype == encoded_test.dtype == np.float32
    assert encoded_test[0, 0] == 2.0
    assert encoded_test[1, 1:].tolist() == [0.0, 0.0]


def test_fast_one_hot_is_selected_for_large_bounded_binary_tables():
    train = pd.DataFrame(
        {
            "numeric": np.arange(2_500),
            "category": pd.Categorical(["left", "right"] * 1_250),
        }
    )
    target = np.arange(2_500) % 2

    assert fedot_exec._should_use_fast_one_hot(train, target)


@pytest.mark.parametrize(
    ("row_count", "class_count"),
    [(2_499, 2), (2_500, 3)],
)
def test_fast_one_hot_keeps_small_or_multiclass_tables_on_existing_path(
    row_count, class_count
):
    train = pd.DataFrame(
        {"category": pd.Categorical(np.arange(row_count) % 2)}
    )
    target = np.arange(row_count) % class_count

    assert not fedot_exec._should_use_fast_one_hot(train, target)


def test_fast_one_hot_rejects_high_cardinality_or_excessive_dense_size():
    train = pd.DataFrame(
        {"category": pd.Categorical([f"level-{index}" for index in range(2_500)])}
    )
    target = np.arange(2_500) % 2

    assert not fedot_exec._should_use_fast_one_hot(train, target)

    bounded = pd.DataFrame(
        {"category": pd.Categorical(np.arange(2_500) % 2)}
    )
    assert not fedot_exec._should_use_fast_one_hot(
        bounded, target, max_dense_cells=4_999
    )


def test_explicit_fast_one_hot_falls_back_to_safe_compact_encoding():
    train = pd.DataFrame(
        {"category": pd.Categorical([f"level-{index}" for index in range(600)])}
    )
    test = pd.DataFrame({"category": pd.Categorical(["level-1", "unknown"])})

    encoded_train, encoded_test = fedot_exec._compact_encode_if_needed(
        train, test, one_hot=True
    )

    assert encoded_train.shape == (600, 1)
    assert encoded_test[:, 0].tolist() == [1.0, -1.0]


def test_compact_encoder_uses_row_guard_for_low_cardinality_tables():
    small_train = pd.DataFrame(
        {"category": pd.Categorical(["0", "1"] * 1_249 + ["0"])}
    )
    large_train = pd.concat(
        [
            small_train,
            pd.DataFrame({"category": pd.Categorical(["1"])})
        ],
        ignore_index=True,
    )
    test = pd.DataFrame({"category": pd.Categorical(["1", "unknown"])})

    unchanged_train, unchanged_test = fedot_exec._compact_encode_if_needed(
        small_train, test
    )
    encoded_train, encoded_test = fedot_exec._compact_encode_if_needed(
        large_train, test
    )

    assert unchanged_train is small_train
    assert unchanged_test is test
    assert encoded_train.shape == (2_500, 1)
    assert encoded_test[:, 0].tolist() == [1.0, -1.0]

    custom_encoded, _ = fedot_exec._compact_encode_if_needed(
        small_train, test, min_rows=2_499
    )
    assert custom_encoded.shape == (2_499, 1)


def test_compact_encoder_keeps_large_nominal_table_on_generic_path():
    train = pd.DataFrame(
        {"category": pd.Categorical(["left", "right"] * 1_250)}
    )
    test = pd.DataFrame({"category": pd.Categorical(["right", "unknown"])})

    unchanged_train, unchanged_test = fedot_exec._compact_encode_if_needed(
        train, test
    )

    assert unchanged_train is train
    assert unchanged_test is test


def test_compact_encoder_rejects_invalid_row_guard():
    frame = pd.DataFrame({"category": pd.Categorical(["left", "right"])})

    with pytest.raises(ValueError, match="row threshold"):
        fedot_exec._compact_encode_if_needed(frame, frame, min_rows=0)


def test_auc_ensemble_retains_robust_boosting_pair_when_alternative_is_weaker():
    truth = np.tile([0, 1], 100)
    strong = np.column_stack((1.0 - (0.1 + 0.8 * truth), 0.1 + 0.8 * truth))
    weak_scores = np.tile([0.6, 0.4], 100)
    weak = np.column_stack((1.0 - weak_scores, weak_scores))
    contenders = [
        {"model": "auto", "score": 0.9, "truth": truth, "probabilities": weak},
        {"model": "lgbm", "score": 0.7, "truth": truth, "probabilities": strong},
        {
            "model": "xgboost",
            "score": 0.8,
            "truth": truth,
            "probabilities": strong,
        },
    ]

    selected, weights, score = fedot_exec._select_auc_ensemble(contenders)

    assert [contender["model"] for contender in selected] == ["lgbm", "xgboost"]
    assert weights == pytest.approx([0.5, 0.5])
    assert score == pytest.approx(1.0)


def test_weak_signal_shallow_probe_has_geometry_and_metric_guards():
    features = np.zeros((50_000, 20))
    target = np.tile([0, 1], 25_000)

    assert fedot_exec._use_weak_signal_shallow_probe(
        features,
        target,
        metric="auc",
        candidates=["lgbm", "xgboost"],
        enabled=True,
        runtime_seconds=180,
        cores=4,
    )
    assert not fedot_exec._use_weak_signal_shallow_probe(
        features[:49_999],
        target[:49_999],
        metric="auc",
        candidates=["lgbm", "xgboost"],
        enabled=True,
        runtime_seconds=180,
        cores=4,
    )
    assert not fedot_exec._use_weak_signal_shallow_probe(
        features,
        target,
        metric="logloss",
        candidates=["lgbm", "xgboost"],
        enabled=True,
        runtime_seconds=180,
        cores=4,
    )
    assert not fedot_exec._use_weak_signal_shallow_probe(
        features,
        target,
        metric="auc",
        candidates=["lgbm", "xgboost"],
        enabled=False,
        runtime_seconds=180,
        cores=4,
    )
    assert not fedot_exec._use_weak_signal_shallow_probe(
        features,
        target,
        metric="auc",
        candidates=["lgbm", "xgboost"],
        enabled=True,
        runtime_seconds=119,
        cores=4,
    )
    assert not fedot_exec._use_weak_signal_shallow_probe(
        features,
        target,
        metric="auc",
        candidates=["lgbm", "xgboost"],
        enabled=True,
        runtime_seconds=180,
        cores=1,
    )


def test_weak_signal_shallow_probe_is_adaptive_but_respects_explicit_controls():
    assert fedot_exec._adaptive_weak_signal_shallow_probe_enabled({"_portfolio": True})
    assert not fedot_exec._adaptive_weak_signal_shallow_probe_enabled(
        {"_portfolio": True, "_portfolio_weak_signal_shallow_probe": False}
    )
    assert fedot_exec._adaptive_weak_signal_shallow_probe_enabled(
        {
            "_portfolio": True,
            "_portfolio_candidates": ["lgbm", "xgboost"],
            "_portfolio_weak_signal_shallow_probe": True,
        }
    )
    assert not fedot_exec._adaptive_weak_signal_shallow_probe_enabled(
        {"_portfolio": True, "_portfolio_xgboost_max_depth": 3}
    )


@pytest.mark.parametrize("score", [0.45, 0.5, 0.55])
def test_weak_auc_score_accepts_near_random_rankers(score):
    assert fedot_exec._is_weak_auc_score(score)


@pytest.mark.parametrize("score", [0.4499, 0.5501, 0.8])
def test_weak_auc_score_rejects_predictive_or_reversed_rankers(score):
    assert not fedot_exec._is_weak_auc_score(score)


def test_weak_auc_score_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="between 0.5 and 1"):
        fedot_exec._is_weak_auc_score(0.5, maximum_auc=0.5)


def test_auc_ensemble_can_select_a_materially_better_singleton():
    rng = np.random.default_rng(42)
    truth = np.tile([0, 1], 300)
    booster_scores = np.clip(
        0.5 + 0.05 * (2 * truth - 1) + rng.normal(0, 0.4, len(truth)),
        0.01,
        0.99,
    )
    auto_scores = np.clip(
        0.5 + 0.4 * (2 * truth - 1) + rng.normal(0, 0.1, len(truth)),
        0.01,
        0.99,
    )

    def probabilities(scores):
        return np.column_stack((1.0 - scores, scores))

    contenders = [
        {
            "model": "lgbm",
            "truth": truth,
            "probabilities": probabilities(booster_scores),
        },
        {
            "model": "xgboost",
            "truth": truth,
            "probabilities": probabilities(booster_scores),
        },
        {"model": "auto", "truth": truth, "probabilities": probabilities(auto_scores)},
    ]

    selected, weights, score = fedot_exec._select_auc_ensemble(contenders)

    assert [contender["model"] for contender in selected] == ["auto"]
    assert weights == pytest.approx([1.0])
    assert score > fedot_exec._selection_score(
        "auc", truth, None, probabilities(booster_scores)
    )


def test_auc_ensemble_rejects_an_unsupported_tiny_improvement():
    rng = np.random.default_rng(7)
    truth = np.tile([0, 1], 40)
    baseline_scores = rng.uniform(0.1, 0.9, len(truth))
    alternative_scores = baseline_scores + 0.0001 * (2 * truth - 1)

    def probabilities(scores):
        scores = np.clip(scores, 0.01, 0.99)
        return np.column_stack((1.0 - scores, scores))

    baseline = probabilities(baseline_scores)
    alternative = probabilities(alternative_scores)
    contenders = [
        {"model": "lgbm", "truth": truth, "probabilities": baseline},
        {"model": "xgboost", "truth": truth, "probabilities": baseline},
        {"model": "auto", "truth": truth, "probabilities": alternative},
    ]

    selected, weights, _ = fedot_exec._select_auc_ensemble(contenders)

    assert [contender["model"] for contender in selected] == ["lgbm", "xgboost"]
    assert weights == pytest.approx([0.5, 0.5])


def test_auc_ensemble_drops_dominated_booster_on_large_supported_holdout():
    rng = np.random.default_rng(1)
    truth = np.tile([0, 1], 1_250)
    common = 0.16 * (2 * truth - 1) + rng.normal(0, 0.45, len(truth))
    strong_scores = common + rng.normal(0, 0.2, len(truth))
    weak_scores = common - 0.02 * (2 * truth - 1) + rng.normal(
        0, 0.2, len(truth)
    )

    def probabilities(scores):
        scores = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - scores, scores))

    contenders = [
        {
            "model": "lgbm",
            "truth": truth,
            "probabilities": probabilities(strong_scores),
        },
        {
            "model": "xgboost",
            "truth": truth,
            "probabilities": probabilities(weak_scores),
        },
    ]

    selected, weights, _ = fedot_exec._select_auc_ensemble(contenders)

    assert [contender["model"] for contender in selected] == ["lgbm"]
    assert weights == pytest.approx([1.0])


def test_auc_ensemble_can_explicitly_retain_dominated_boosting_pair():
    rng = np.random.default_rng(1)
    truth = np.tile([0, 1], 1_250)
    common = 0.16 * (2 * truth - 1) + rng.normal(0, 0.45, len(truth))
    strong_scores = common + rng.normal(0, 0.2, len(truth))
    weak_scores = common - 0.02 * (2 * truth - 1) + rng.normal(
        0, 0.2, len(truth)
    )

    def probabilities(scores):
        scores = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - scores, scores))

    contenders = [
        {
            "model": "lgbm",
            "truth": truth,
            "probabilities": probabilities(strong_scores),
        },
        {
            "model": "xgboost",
            "truth": truth,
            "probabilities": probabilities(weak_scores),
        },
    ]

    selected, weights, _ = fedot_exec._select_auc_ensemble(
        contenders,
        retain_boosting_pair=True,
    )

    assert [contender["model"] for contender in selected] == ["lgbm", "xgboost"]
    assert weights == pytest.approx([0.5, 0.5])


def test_auc_ensemble_keeps_pair_for_dominated_member_on_small_holdout():
    rng = np.random.default_rng(1)
    truth = np.tile([0, 1], 75)
    common = 0.16 * (2 * truth - 1) + rng.normal(0, 0.45, len(truth))
    strong_scores = common + rng.normal(0, 0.2, len(truth))
    weak_scores = common - 0.02 * (2 * truth - 1) + rng.normal(
        0, 0.2, len(truth)
    )

    def probabilities(scores):
        scores = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - scores, scores))

    contenders = [
        {
            "model": "lgbm",
            "truth": truth,
            "probabilities": probabilities(strong_scores),
        },
        {
            "model": "xgboost",
            "truth": truth,
            "probabilities": probabilities(weak_scores),
        },
    ]

    selected, weights, _ = fedot_exec._select_auc_ensemble(contenders)

    assert [contender["model"] for contender in selected] == ["lgbm", "xgboost"]
    assert weights == pytest.approx([0.5, 0.5])


def test_auc_ensemble_leaves_a_losing_pair_to_the_uncertainty_guard():
    rng = np.random.default_rng(0)
    truth = np.tile([0, 1], 1_250)
    common = 0.16 * (2 * truth - 1) + rng.normal(0, 0.45, len(truth))
    strong_scores = common + rng.normal(0, 0.175, len(truth))
    weak_scores = common - 0.01 * (2 * truth - 1) + rng.normal(
        0, 0.175, len(truth)
    )

    def probabilities(scores):
        scores = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - scores, scores))

    contenders = [
        {
            "model": "lgbm",
            "truth": truth,
            "probabilities": probabilities(strong_scores),
        },
        {
            "model": "xgboost",
            "truth": truth,
            "probabilities": probabilities(weak_scores),
        },
    ]

    selected, weights, _ = fedot_exec._select_auc_ensemble(contenders)

    assert [contender["model"] for contender in selected] == ["lgbm", "xgboost"]
    assert weights == pytest.approx([0.5, 0.5])


def test_auc_ensemble_restores_a_significantly_complementary_pair():
    rng = np.random.default_rng(0)
    truth = np.tile([0, 1], 1_250)
    common = 0.16 * (2 * truth - 1) + rng.normal(0, 0.45, len(truth))
    strong_scores = common + rng.normal(0, 0.35, len(truth))
    weak_scores = common - 0.01 * (2 * truth - 1) + rng.normal(
        0, 0.35, len(truth)
    )

    def probabilities(scores):
        scores = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - scores, scores))

    contenders = [
        {
            "model": "lgbm",
            "truth": truth,
            "probabilities": probabilities(strong_scores),
        },
        {
            "model": "xgboost",
            "truth": truth,
            "probabilities": probabilities(weak_scores),
        },
    ]

    selected, weights, _ = fedot_exec._select_auc_ensemble(contenders)

    assert [contender["model"] for contender in selected] == ["lgbm", "xgboost"]
    assert weights == pytest.approx([0.5, 0.5])


def test_logloss_ensemble_does_not_force_in_a_weaker_model():
    truth = np.repeat([0, 1], 50)
    strong = np.vstack(
        (np.tile([0.9, 0.1], (50, 1)), np.tile([0.1, 0.9], (50, 1)))
    )
    weak = np.full((100, 2), 0.5)
    contenders = [
        {"model": "strong", "score": -0.1, "probabilities": strong, "truth": truth},
        {"model": "weak", "score": -0.7, "probabilities": weak, "truth": truth},
    ]

    selected, weights, _, _ = fedot_exec._select_logloss_ensemble(
        contenders, calibrate=False
    )

    assert [contender["model"] for contender in selected] == ["strong"]
    assert weights == pytest.approx([1.0])


def test_logloss_ensemble_keeps_a_complementary_blend():
    truth = np.array([0, 0, 1, 1] * 25)
    left = np.tile([[0.9, 0.1], [0.55, 0.45], [0.1, 0.9], [0.45, 0.55]], (25, 1))
    right = np.tile([[0.55, 0.45], [0.9, 0.1], [0.45, 0.55], [0.1, 0.9]], (25, 1))
    contenders = [
        {"model": "left", "score": -0.3, "probabilities": left, "truth": truth},
        {"model": "right", "score": -0.3, "probabilities": right, "truth": truth},
    ]

    selected, weights, _, score = fedot_exec._select_logloss_ensemble(
        contenders, calibrate=False
    )

    assert [contender["model"] for contender in selected] == ["left", "right"]
    assert weights == pytest.approx([0.5, 0.5])
    assert score > fedot_exec._selection_score("logloss", truth, None, left)


def test_wide_extra_trees_gate_accepts_clear_disjoint_gain():
    truth = np.arange(400) % 2

    def probabilities(true_probability):
        positive = np.where(truth == 1, true_probability, 1 - true_probability)
        return np.column_stack((1 - positive, positive))

    baseline_confidence = np.where(np.arange(400) % 5 == 0, 0.30, 0.65)
    wide_confidence = np.where(np.arange(400) % 5 == 0, 0.45, 0.75)
    baseline = probabilities(baseline_confidence)
    wide = probabilities(wide_confidence)
    contenders = [
        {
            "model": "extra_trees",
            "score": fedot_exec._selection_score(
                "logloss", truth, None, baseline
            ),
            "probabilities": baseline,
            "truth": truth,
            "labels": np.array([0, 1]),
        },
        {
            "model": "extra_trees_wide",
            "score": fedot_exec._selection_score("logloss", truth, None, wide),
            "probabilities": wide,
            "truth": truth,
            "labels": np.array([0, 1]),
        },
    ]

    selected = fedot_exec._train_gated_wide_extra_trees_contenders(
        contenders,
        reference_target=truth,
        enabled=True,
        seed=42,
    )

    assert [contender["model"] for contender in selected] == [
        "extra_trees",
        "extra_trees_wide",
    ]


def test_wide_extra_trees_gate_rejects_tie_and_disabled_probe():
    truth = np.arange(400) % 2
    positive = np.where(truth == 1, 0.7, 0.3)
    probabilities = np.column_stack((1 - positive, positive))

    def contenders():
        return [
            {
                "model": "extra_trees",
                "score": fedot_exec._selection_score(
                    "logloss", truth, None, probabilities
                ),
                "probabilities": probabilities,
                "truth": truth,
                "labels": np.array([0, 1]),
            },
            {
                "model": "extra_trees_wide",
                "score": fedot_exec._selection_score(
                    "logloss", truth, None, probabilities
                ),
                "probabilities": probabilities,
                "truth": truth,
                "labels": np.array([0, 1]),
                "validation_models": [object()],
            },
        ]

    tied = fedot_exec._train_gated_wide_extra_trees_contenders(
        contenders(), reference_target=truth, enabled=True, seed=42
    )
    disabled = fedot_exec._train_gated_wide_extra_trees_contenders(
        contenders(), reference_target=truth, enabled=False, seed=42
    )

    assert [contender["model"] for contender in tied] == ["extra_trees"]
    assert [contender["model"] for contender in disabled] == ["extra_trees"]


def test_lgbm_extra_trees_pair_domain_and_explicit_override_guard():
    dense = np.broadcast_to(
        np.ones((1, 180), dtype=np.float32), (50_000, 180)
    )
    target = np.arange(50_000) % 10
    rare_target = target.copy()
    rare_target[:999] = 10

    assert fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        dense, target
    )
    assert fedot_exec._use_train_gated_lgbm_extra_trees_pair(
        dense,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={},
    )
    assert not fedot_exec._use_train_gated_lgbm_extra_trees_pair(
        dense,
        target,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio_train_rows": 10_000},
    )
    assert not fedot_exec._use_train_gated_lgbm_extra_trees_pair(
        dense,
        target,
        metric="logloss",
        runtime_seconds=179,
        cores=4,
        framework_params={},
    )
    assert not fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        dense[:29_999], target[:29_999]
    )
    assert not fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        dense[:, :63], target
    )
    assert not fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        np.broadcast_to(np.ones((1, 256), dtype=np.float32), (50_000, 256)),
        target,
    )
    assert not fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        dense, rare_target
    )
    assert not fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        dense, np.arange(50_000) % 20
    )
    assert not fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        dense, np.arange(50_000) % 9
    )
    assert not fedot_exec._is_train_gated_lgbm_extra_trees_pair_candidate(
        sparse.csr_matrix(dense), target
    )


def test_lgbm_extra_trees_pair_gate_accepts_only_fixed_validated_pair(
    monkeypatch,
):
    truth = np.arange(400) % 2

    def probabilities(true_probability):
        positive = np.where(truth == 1, true_probability, 1 - true_probability)
        return np.column_stack((1 - positive, positive))

    reference_target = np.arange(300) % 2
    observed_references = []
    monkeypatch.setattr(fedot_exec, "_fit_temperature", lambda *args, **kwargs: 1.0)

    def prior_exponent(*args, reference_target, **kwargs):
        observed_references.append(np.asarray(reference_target).copy())
        return 0.0

    monkeypatch.setattr(fedot_exec, "_fit_prior_exponent", prior_exponent)
    contenders = [
        {
            "model": "lgbm",
            "score": -0.6,
            "duration": 2.0,
            "probabilities": probabilities(0.60),
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
        {
            "model": "xgboost",
            "score": -0.6,
            "duration": 3.0,
            "probabilities": probabilities(0.60),
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
        {
            "model": fedot_exec._LGBM_EXTRA_TREES_MODEL,
            "score": -0.4,
            "duration": 2.5,
            "probabilities": probabilities(0.80),
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
    ]

    retained, forced = fedot_exec._train_gated_lgbm_extra_trees_pair_contenders(
        contenders,
        reference_target=reference_target,
        enabled=True,
        seed=42,
    )

    assert retained == contenders
    assert [member["model"] for member in forced["ensemble"]] == [
        fedot_exec._LGBM_EXTRA_TREES_MODEL,
        "xgboost",
    ]
    assert forced["weights"] == pytest.approx([0.5, 0.5])
    assert forced["gain"] >= 0.005
    assert forced["lower_95"] > 0
    assert forced["raw_gain"] > 0
    assert forced["pair_ratio"] == pytest.approx(1.1)
    assert forced["lgbm_overhead_seconds"] == pytest.approx(0.5)
    assert len(observed_references) == 3
    assert all(
        np.array_equal(observed, reference_target)
        for observed in observed_references
    )


def test_lgbm_extra_trees_pair_gate_cost_reject_keeps_ordinary_predictions(
    monkeypatch,
):
    truth = np.arange(400) % 2

    def probabilities(true_probability):
        positive = np.where(truth == 1, true_probability, 1 - true_probability)
        return np.column_stack((1 - positive, positive))

    monkeypatch.setattr(fedot_exec, "_fit_temperature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(fedot_exec, "_fit_prior_exponent", lambda *args, **kwargs: 0.0)
    lgbm_probabilities = probabilities(0.60)
    xgboost_probabilities = probabilities(0.60)
    ordinary = [
        {
            "model": "lgbm",
            "score": -0.6,
            "duration": 2.0,
            "probabilities": lgbm_probabilities,
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
        {
            "model": "xgboost",
            "score": -0.6,
            "duration": 3.0,
            "probabilities": xgboost_probabilities,
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
    ]
    challenger = {
        "model": fedot_exec._LGBM_EXTRA_TREES_MODEL,
        "score": -0.4,
        "duration": 10.0,
        "probabilities": probabilities(0.80),
        "truth": truth,
        "labels": np.array([0, 1]),
        "validation_models": [object()],
    }

    retained, forced = fedot_exec._train_gated_lgbm_extra_trees_pair_contenders(
        [*ordinary, challenger],
        reference_target=np.arange(300) % 2,
        enabled=True,
        seed=42,
    )

    assert forced is None
    assert retained == ordinary
    assert retained[0]["probabilities"] is lgbm_probabilities
    assert retained[1]["probabilities"] is xgboost_probabilities
    assert "validation_models" not in challenger


def test_lgbm_extra_trees_pair_gate_cannot_replace_stronger_ordinary_singleton(
    monkeypatch,
):
    truth = np.arange(400) % 2

    def probabilities(true_probability):
        positive = np.where(truth == 1, true_probability, 1 - true_probability)
        return np.column_stack((1 - positive, positive))

    monkeypatch.setattr(fedot_exec, "_fit_temperature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(fedot_exec, "_fit_prior_exponent", lambda *args, **kwargs: 0.0)
    ordinary = [
        {
            "model": "lgbm",
            "score": -0.7,
            "duration": 2.0,
            "probabilities": probabilities(0.55),
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
        {
            "model": "xgboost",
            "score": -0.1,
            "duration": 3.0,
            "probabilities": probabilities(0.90),
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
    ]
    challenger = {
        "model": fedot_exec._LGBM_EXTRA_TREES_MODEL,
        "score": -0.4,
        "duration": 2.5,
        "probabilities": probabilities(0.65),
        "truth": truth,
        "labels": np.array([0, 1]),
        "validation_models": [object()],
    }

    retained, forced = fedot_exec._train_gated_lgbm_extra_trees_pair_contenders(
        [*ordinary, challenger],
        reference_target=np.arange(300) % 2,
        enabled=True,
        seed=42,
    )

    # Replacing the weak LightGBM improves the frozen 50/50 pair, but it does
    # not beat the XGBoost singleton that the ordinary selector would deploy.
    assert forced is None
    assert retained == ordinary
    assert "validation_models" not in challenger


def test_lgbm_extra_trees_pair_gate_rejects_unvalidated_slow_selector_phase(
    monkeypatch,
):
    truth = np.arange(400) % 2

    def probabilities(true_probability):
        positive = np.where(truth == 1, true_probability, 1 - true_probability)
        return np.column_stack((1 - positive, positive))

    monkeypatch.setattr(fedot_exec, "_fit_temperature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(fedot_exec, "_fit_prior_exponent", lambda *args, **kwargs: 0.0)
    ordinary = [
        {
            "model": "lgbm",
            "score": -0.6,
            "duration": 20.0,
            "probabilities": probabilities(0.60),
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
        {
            "model": "xgboost",
            "score": -0.6,
            "duration": 20.0,
            "probabilities": probabilities(0.60),
            "truth": truth,
            "labels": np.array([0, 1]),
            "validation_models": [object()],
        },
    ]
    challenger = {
        "model": fedot_exec._LGBM_EXTRA_TREES_MODEL,
        "score": -0.4,
        "duration": 20.0,
        "probabilities": probabilities(0.80),
        "truth": truth,
        "labels": np.array([0, 1]),
        "validation_models": [object()],
    }

    retained, forced = fedot_exec._train_gated_lgbm_extra_trees_pair_contenders(
        [*ordinary, challenger],
        reference_target=np.arange(300) % 2,
        enabled=True,
        seed=42,
    )

    assert forced is None
    assert retained == ordinary
    assert "validation_models" not in challenger


def test_lgbm_extra_trees_unique_name_maps_to_lgbm_parameters():
    params = fedot_exec._candidate_model_params(
        fedot_exec._LGBM_EXTRA_TREES_MODEL,
        lgbm_num_leaves=63,
        lgbm_min_child_samples=100,
    )
    pipeline = fedot_exec._predefined_model_with_n_jobs(
        fedot_exec._LGBM_EXTRA_TREES_MODEL,
        4,
        n_estimators=300,
        model_params=params,
    )
    node = pipeline.nodes[0]

    assert params == {
        "num_leaves": 63,
        "min_child_samples": 100,
        "extra_trees": True,
    }
    assert node.operation.operation_type == "lgbm"
    assert node.parameters["extra_trees"] is True
    assert node.parameters["n_estimators"] == 300
    assert node.parameters["n_jobs"] == 4


@pytest.mark.parametrize(
    (
        "refit_estimate",
        "secondary_refit_estimate",
        "expected_confidence",
        "expected_full_fit_calls",
        "expected_temperature_fits",
        "expected_prior_fits",
        "expected_prior_applications",
        "expected_reference_lengths",
    ),
    [
        (200.0, 1.0, 0.60, [], 8, 4, 5, [8_784, 8_784, 8_784, 30_000]),
        (
            1.0,
            1.0,
            0.70,
            [
                (fedot_exec._LGBM_EXTRA_TREES_MODEL, 30_000),
                (fedot_exec._LGBM_EXTRA_TREES_MODEL, 30_000),
                ("xgboost", 30_000),
                ("xgboost", 30_000),
            ],
            5,
            3,
            4,
            [8_784, 8_784, 8_784],
        ),
        (
            1.0,
            200.0,
            0.60,
            [
                (fedot_exec._LGBM_EXTRA_TREES_MODEL, 30_000),
                (fedot_exec._LGBM_EXTRA_TREES_MODEL, 30_000),
            ],
            8,
            4,
            5,
            [8_784, 8_784, 8_784, 30_000],
        ),
    ],
)
def test_lgbm_extra_trees_pair_deployment_requires_joint_refit_capacity(
    monkeypatch,
    refit_estimate,
    secondary_refit_estimate,
    expected_confidence,
    expected_full_fit_calls,
    expected_temperature_fits,
    expected_prior_fits,
    expected_prior_applications,
    expected_reference_lengths,
):
    row_count = 30_000
    class_count = 10
    target = np.arange(row_count) % class_count
    features = np.ones((row_count, 180), dtype=np.float32)
    features[:, 0] = target
    fit_calls = []
    temperature_references = []
    prior_fit_references = []
    prior_apply_references = []

    class FakeFedot:
        def __init__(self, **kwargs):
            self.current_pipeline = SimpleNamespace(length=1, nodes=[])
            self.history = None
            self.target = None
            self.model_name = None

        def fit(self, features, target, predefined_model):
            self.target = np.asarray(target)
            self.model_name = predefined_model
            fit_calls.append((self.model_name, len(features)))

        def predict_proba(self, features, probs_for_all_classes):
            del probs_for_all_classes
            truth = np.asarray(features)[:, 0].astype(int)
            confidence = (
                0.80
                if self.model_name == fedot_exec._LGBM_EXTRA_TREES_MODEL
                else 0.60
            )
            probabilities = np.full(
                (len(truth), class_count),
                (1.0 - confidence) / (class_count - 1),
            )
            probabilities[np.arange(len(truth)), truth] = confidence
            return probabilities

    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:20], y=target[:20]),
        encoded_class_count=class_count,
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=4,
        seed=42,
        max_runtime_seconds=180,
        output_predictions_file="unused.csv",
        framework_params={"_portfolio": True},
    )

    def fit_temperature(probabilities, truth, **kwargs):
        del probabilities, kwargs
        temperature_references.append(np.asarray(truth).copy())
        return 1.0

    def fit_prior(*args, reference_target, **kwargs):
        del args, kwargs
        prior_fit_references.append(np.asarray(reference_target).copy())
        return 0.125

    def apply_prior(probabilities, reference_target, exponent, labels=None):
        del exponent, labels
        prior_apply_references.append(np.asarray(reference_target).copy())
        return np.asarray(probabilities)

    monkeypatch.setattr(fedot_exec, "Fedot", FakeFedot)
    monkeypatch.setattr(
        fedot_exec,
        "_predefined_model_with_n_jobs",
        lambda predefined_model, *args, **kwargs: predefined_model,
    )
    monkeypatch.setattr(fedot_exec, "_fit_temperature", fit_temperature)
    monkeypatch.setattr(fedot_exec, "_fit_prior_exponent", fit_prior)
    monkeypatch.setattr(fedot_exec, "_apply_prior_exponent", apply_prior)
    monkeypatch.setattr(
        fedot_exec,
        "_adaptive_deployment_temperature_multiplier",
        lambda *args, **kwargs: 1.0,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_adaptive_post_prior_temperature_multiplier",
        lambda *args, **kwargs: 1.0,
    )
    paired_gate = fedot_exec._train_gated_lgbm_extra_trees_pair_contenders

    def stable_cost_gate(contenders, *args, **kwargs):
        durations = {
            "lgbm": 2.0,
            "xgboost": 3.0,
            fedot_exec._LGBM_EXTRA_TREES_MODEL: 2.5,
        }
        for contender in contenders:
            contender["duration"] = durations[contender["model"]]
        return paired_gate(contenders, *args, **kwargs)

    monkeypatch.setattr(
        fedot_exec,
        "_train_gated_lgbm_extra_trees_pair_contenders",
        stable_cost_gate,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_direct_refit_budget_estimate",
        lambda *args, **kwargs: refit_estimate,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_secondary_refit_budget_estimate",
        lambda *args, **kwargs: secondary_refit_estimate,
    )
    monkeypatch.setattr(fedot_exec, "save_artifacts", lambda *args: None)
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fit_calls == [
        ("lgbm", 8_784),
        ("xgboost", 8_784),
        (fedot_exec._LGBM_EXTRA_TREES_MODEL, 8_784),
        *expected_full_fit_calls,
    ]
    assert output["models_count"] == 2
    assert output["probabilities"][np.arange(20), target[:20]] == pytest.approx(
        np.full(20, expected_confidence)
    )
    # Admission deploys both all-row members atomically. If their joint estimate
    # does not fit, the ordinary selector runs without the challenger instead.
    assert len(temperature_references) == expected_temperature_fits
    assert len(prior_fit_references) == expected_prior_fits
    assert len(prior_apply_references) == expected_prior_applications
    assert [len(reference) for reference in prior_fit_references] == (
        expected_reference_lengths
    )
    assert [len(reference) for reference in prior_apply_references] == [
        *expected_reference_lengths,
        expected_reference_lengths[-1],
    ]


def test_logloss_ensemble_can_shrink_a_pair_towards_the_stronger_singleton():
    truth = np.array([0, 0, 1, 1] * 25)
    stronger = np.tile([[0.9, 0.1], [0.55, 0.45], [0.1, 0.9], [0.45, 0.55]], (25, 1))
    weaker = np.tile([[0.55, 0.45], [0.9, 0.1], [0.45, 0.55], [0.1, 0.9]], (25, 1))
    contenders = [
        {"model": "stronger", "score": -0.2, "probabilities": stronger, "truth": truth},
        {"model": "weaker", "score": -0.4, "probabilities": weaker, "truth": truth},
    ]

    selected, weights, _, _ = fedot_exec._select_logloss_ensemble(
        contenders, calibrate=False, pair_strong_weight=0.75
    )

    assert [contender["model"] for contender in selected] == ["stronger", "weaker"]
    assert weights == pytest.approx([0.75, 0.25])


def test_logloss_ensemble_rejects_invalid_fixed_pair_weight():
    with pytest.raises(ValueError, match="pair_strong_weight"):
        fedot_exec._select_logloss_ensemble([], calibrate=False, pair_strong_weight=0.4)


def test_logloss_ensemble_rejects_large_relative_penalty_despite_noise():
    truth = np.array([0, 1] * 5)
    strong_true_probability = np.full(len(truth), 0.99)
    noisy_true_probability = np.tile([0.9999, 0.95], 5)

    def probabilities(true_probability):
        positive = np.where(truth == 1, true_probability, 1 - true_probability)
        return np.column_stack((1 - positive, positive))

    contenders = [
        {
            "model": "strong",
            "score": -0.01,
            "probabilities": probabilities(strong_true_probability),
            "truth": truth,
        },
        {
            "model": "noisy",
            "score": -0.03,
            "probabilities": probabilities(noisy_true_probability),
            "truth": truth,
        },
    ]

    selected, weights, _, _ = fedot_exec._select_logloss_ensemble(
        contenders, calibrate=False
    )

    assert [contender["model"] for contender in selected] == ["strong"]
    assert weights == pytest.approx([1.0])


def test_logloss_ensemble_rejects_dominated_non_improving_pair():
    truth = np.arange(100) % 2
    strong_true_probability = np.r_[np.full(98, 0.999), np.full(2, 0.001)]
    weak_true_probability = np.r_[np.full(98, 0.75), np.full(2, 0.99)]

    def probabilities(true_probability):
        positive = np.where(truth == 1, true_probability, 1 - true_probability)
        return np.column_stack((1 - positive, positive))

    strong = probabilities(strong_true_probability)
    weak = probabilities(weak_true_probability)
    contenders = [
        {
            "model": "strong",
            "score": fedot_exec._selection_score(
                "logloss", truth, None, strong
            ),
            "probabilities": strong,
            "truth": truth,
        },
        {
            "model": "weak",
            "score": fedot_exec._selection_score(
                "logloss", truth, None, weak
            ),
            "probabilities": weak,
            "truth": truth,
        },
    ]

    selected, weights, _, _ = fedot_exec._select_logloss_ensemble(
        contenders, calibrate=False
    )

    assert [contender["model"] for contender in selected] == ["strong"]
    assert weights == pytest.approx([1.0])


def test_dense_direct_refit_uses_reliability_weighted_booster_pair():
    ensemble = [{"model": "xgboost"}, {"model": "lgbm"}]

    weights = fedot_exec._direct_wide_deployment_weights(
        ensemble,
        selected_weights=np.array([0.5, 0.5]),
        direct_all_row_refit=True,
        sampled_numeric_density=0.5,
        adaptive_portfolio=True,
    )

    assert weights == pytest.approx([0.3, 0.7])


@pytest.mark.parametrize(
    ("direct_refit", "density", "adaptive"),
    [
        (False, 0.5, True),
        (True, 0.349, True),
        (True, 0.5, False),
    ],
)
def test_sparse_or_explicit_portfolio_keeps_selected_weights(
    direct_refit, density, adaptive
):
    selected = np.array([0.5, 0.5])

    weights = fedot_exec._direct_wide_deployment_weights(
        [{"model": "lgbm"}, {"model": "xgboost"}],
        selected_weights=selected,
        direct_all_row_refit=direct_refit,
        sampled_numeric_density=density,
        adaptive_portfolio=adaptive,
    )

    assert weights == pytest.approx(selected)


def test_portfolio_selects_better_holdout_candidate_and_refits_it(monkeypatch):
    fitted_models = []

    class FakeFedot:
        def __init__(self, **kwargs):
            self.model = None
            self.current_pipeline = SimpleNamespace(length=1)

        def fit(self, features, target, predefined_model):
            self.model = predefined_model
            fitted_models.append(predefined_model)

        def predict(self, features):
            return np.asarray(features)[:, 0].astype(int)

        def predict_proba(self, features, probs_for_all_classes):
            truth = np.asarray(features)[:, 0].astype(int)
            confidence = 0.9 if self.model == "xgboost" else 0.6
            positive = np.where(truth == 1, confidence, 1.0 - confidence)
            return np.column_stack((1.0 - positive, positive))

    features = np.tile([[0.0], [1.0]], (100, 1))
    target = features[:, 0].astype(int)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:10], y=target[:10]),
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=2,
        seed=42,
        max_runtime_seconds=60,
        output_predictions_file="unused.csv",
        framework_params={
            "_portfolio": True,
            "_portfolio_calibrate": False,
            "_portfolio_candidates": ["lgbm", "xgboost"],
            "_portfolio_full_data_refit": False,
        },
    )

    monkeypatch.setattr(fedot_exec, "Fedot", FakeFedot)
    monkeypatch.setattr(
        fedot_exec,
        "_predefined_model_with_n_jobs",
        lambda predefined_model,
        n_jobs,
        n_estimators=None,
        use_eval_set=None,
        model_params=None: predefined_model,
    )
    monkeypatch.setattr(fedot_exec, "save_artifacts", lambda *args: None)
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert fitted_models == ["lgbm"] * 3 + ["xgboost"] * 4
    assert output["models_count"] == 1
    assert output["target_is_encoded"] is True
    assert output["probabilities"][:, 1] == pytest.approx(
        np.where(target[:10] == 1, 0.9, 0.1)
    )


def test_automatic_scaled_svc_uses_sklearn_path_for_validation_and_refit(
    monkeypatch,
):
    svc_fits = []
    fedot_factory_calls = []

    class FakeProbabilityModel:
        def __init__(self, target):
            self.target = np.asarray(target)
            self.current_pipeline = SimpleNamespace(length=1)

        def predict_proba(self, features, probs_for_all_classes=True):
            del probs_for_all_classes
            class_count = len(np.unique(self.target))
            return np.full((len(features), class_count), 1.0 / class_count)

    def fit_scaled_svc(features, target, model_params=None, seed=42):
        svc_fits.append((len(features), model_params, seed))
        return FakeProbabilityModel(target)

    def make_fedot(**kwargs):
        fedot_factory_calls.append(kwargs)
        return FakeProbabilityModel(target)

    features = np.ones((1_500, 128), dtype=np.float32)
    target = np.arange(len(features)) % 10
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:20], y=target[:20]),
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=2,
        seed=17,
        max_runtime_seconds=60,
        output_predictions_file="unused.csv",
        framework_params={
            "_portfolio": True,
            "_portfolio_calibrate": False,
        },
    )

    monkeypatch.setattr(fedot_exec, "_fit_scaled_svc_candidate", fit_scaled_svc)
    monkeypatch.setattr(fedot_exec, "_make_fedot", make_fedot)
    monkeypatch.setattr(fedot_exec, "save_artifacts", lambda *args: None)
    monkeypatch.setattr(fedot_exec, "result", lambda **kwargs: kwargs)

    output = fedot_exec.run(dataset, config)

    assert [row_count for row_count, _, _ in svc_fits] == [1_000] * 3 + [1_500]
    assert all(params["C"] == pytest.approx(3.0) for _, params, _ in svc_fits)
    assert all(seed == 17 for _, _, seed in svc_fits)
    assert fedot_factory_calls == []
    assert output["models_count"] == 1
    assert output["probabilities"] == pytest.approx(np.full((20, 10), 0.1))


def test_portfolio_rejects_non_positive_full_data_refit_threshold():
    features = np.tile([[0.0], [1.0]], (50, 1))
    target = features[:, 0].astype(int)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:10], y=target[:10]),
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=2,
        seed=42,
        max_runtime_seconds=60,
        output_predictions_file="unused.csv",
        framework_params={
            "_portfolio": True,
            "_portfolio_full_data_refit_min_rows": 0,
        },
    )

    with pytest.raises(ValueError, match="full_data_refit_min_rows"):
        fedot_exec.run(dataset, config)


@pytest.mark.parametrize("exponent", [-0.01, 1.01])
def test_portfolio_rejects_invalid_direct_round_exponent(exponent):
    features = np.tile([[0.0], [1.0]], (50, 1))
    target = features[:, 0].astype(int)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features[:10], y=target[:10]),
    )
    config = SimpleNamespace(
        type="classification",
        metric="logloss",
        cores=2,
        seed=42,
        max_runtime_seconds=60,
        output_predictions_file="unused.csv",
        framework_params={
            "_portfolio": True,
            "_portfolio_direct_round_exponent": exponent,
        },
    )

    with pytest.raises(ValueError, match="direct_round_exponent"):
        fedot_exec.run(dataset, config)


def _make_exact_one_hot_frame(row_count):
    rows = np.arange(row_count)
    data = {
        "ordinary_0": rows.astype(np.float32) + 0.25,
        "ordinary_1": (rows % 17).astype(np.float32) + 0.5,
    }
    first_active = rows % 8
    for category in range(8):
        data[f"first_{category}"] = (first_active == category).astype(np.uint8)
    data.update(
        {
            "ordinary_2": (rows % 29).astype(np.float32) + 0.75,
            "ordinary_3": (rows % 37).astype(np.float32) + 0.125,
        }
    )
    second_active = (rows * 3 + 1) % 8
    for category in range(8):
        data[f"second_{category}"] = (second_active == category).astype(np.uint8)
    return pd.DataFrame(data)


def test_exact_one_hot_detector_finds_separate_contiguous_groups():
    frame = _make_exact_one_hot_frame(48)

    groups = fedot_exec._contiguous_exact_one_hot_groups(frame)

    assert groups == [
        tuple(f"first_{category}" for category in range(8)),
        tuple(f"second_{category}" for category in range(8)),
    ]


@pytest.mark.parametrize("kind", ["multi_hot", "scattered"])
def test_exact_one_hot_detector_rejects_invalid_or_scattered_blocks(kind):
    if kind == "multi_hot":
        frame = pd.DataFrame(
            np.tile([[1, 1, 0], [1, 0, 1], [0, 1, 1]], (4, 1)),
            columns=["g0", "g1", "g2"],
        )
    else:
        active = np.arange(12) % 3
        frame = pd.DataFrame(
            {
                "g0": (active == 0).astype(np.uint8),
                "separator": np.arange(12, dtype=np.float32) + 0.5,
                "g1": (active == 1).astype(np.uint8),
                "g2": (active == 2).astype(np.uint8),
            }
        )

    assert fedot_exec._contiguous_exact_one_hot_groups(frame) == []


def test_exact_one_hot_profile_is_structural_and_resource_bounded():
    frame = _make_exact_one_hot_frame(100_000)
    target = np.arange(len(frame)) % 3

    profile = fedot_exec._exact_one_hot_grouped_xgboost_profile(frame, target)

    assert profile["groups"] == [
        tuple(f"first_{category}" for category in range(8)),
        tuple(f"second_{category}" for category in range(8)),
    ]
    assert profile["ordinary_columns"] == [
        "ordinary_0",
        "ordinary_1",
        "ordinary_2",
        "ordinary_3",
    ]
    assert profile["original_feature_count"] == 20
    assert profile["compressed_feature_count"] == 6
    assert fedot_exec._exact_one_hot_grouped_xgboost_profile(
        frame.iloc[:99_999], target[:99_999]
    ) is None


def test_exact_one_hot_branch_honours_metric_resource_and_override_guards(
    monkeypatch,
):
    sentinel = {"groups": [("a", "b", "c")]}
    monkeypatch.setattr(
        fedot_exec,
        "_exact_one_hot_grouped_xgboost_profile",
        lambda features, target: sentinel,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_exact_one_hot_baseline_matches_portfolio",
        lambda features, target, metric: True,
    )
    common = dict(features=object(), target=np.arange(30) % 3)

    assert fedot_exec._use_exact_one_hot_grouped_xgboost(
        **common,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    ) is sentinel
    for metric, runtime_seconds, cores in [
        ("auc", 180, 4),
        ("logloss", 179, 4),
        ("logloss", 180, 3),
    ]:
        assert fedot_exec._use_exact_one_hot_grouped_xgboost(
            **common,
            metric=metric,
            runtime_seconds=runtime_seconds,
            cores=cores,
            framework_params={"_portfolio": True},
        ) is None
    assert fedot_exec._use_exact_one_hot_grouped_xgboost(
        **common,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio_validation_fraction": 0.25},
    ) is None
    assert fedot_exec._use_exact_one_hot_grouped_xgboost(
        **common,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={
            "_portfolio_grouped_one_hot_xgboost": True,
            "_portfolio_validation_fraction": 0.25,
        },
    ) is sentinel
    assert fedot_exec._use_exact_one_hot_grouped_xgboost(
        **common,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio_grouped_one_hot_xgboost": False},
    ) is None

    monkeypatch.setattr(
        fedot_exec,
        "_exact_one_hot_baseline_matches_portfolio",
        lambda features, target, metric: False,
    )
    assert fedot_exec._use_exact_one_hot_grouped_xgboost(
        **common,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio": True},
    ) is None
    assert fedot_exec._use_exact_one_hot_grouped_xgboost(
        **common,
        metric="logloss",
        runtime_seconds=180,
        cores=4,
        framework_params={"_portfolio_grouped_one_hot_xgboost": True},
    ) is sentinel


def test_exact_one_hot_transforms_use_train_medians_and_unknown_category():
    train = pd.DataFrame(
        {
            0: [1.0, np.nan, 3.0],
            "0": [np.inf, 4.0, 6.0],
            "g0": [1, 0, 0],
            "g1": [0, 1, 0],
            "g2": [0, 0, 1],
        }
    )
    profile = {
        "ordinary_columns": [0, "0"],
        "groups": [("g0", "g1", "g2")],
    }
    test = pd.DataFrame(
        {
            0: [np.nan, 7.0, 9.0, 11.0],
            "0": [np.inf, 8.0, 10.0, 12.0],
            "g0": [0.5, 0, 0, 1],
            "g1": [0.5, 0, 1, np.nan],
            "g2": [0, 0, 0, 0],
        }
    )

    baseline = fedot_exec._apply_exact_one_hot_transform(
        test, fedot_exec._fit_exact_one_hot_numeric_transform(train)
    )
    grouped = fedot_exec._apply_exact_one_hot_transform(
        test, fedot_exec._fit_exact_one_hot_grouped_transform(train, profile)
    )

    assert baseline.columns.is_unique
    assert grouped.columns.is_unique
    assert grouped.iloc[0, :2].to_numpy(dtype=float) == pytest.approx([2.0, 5.0])
    assert grouped["__fedot_exact_one_hot_0"].astype(int).tolist() == [3, 3, 1, 3]


def test_grouped_final_refit_uses_ninety_five_percent_without_eval_set(monkeypatch):
    features = _make_exact_one_hot_frame(1_000)
    target = np.arange(len(features)) % 10
    groups = fedot_exec._contiguous_exact_one_hot_groups(features)
    grouped_columns = {column for group in groups for column in group}
    profile = {
        "groups": groups,
        "ordinary_columns": [
            column for column in features if column not in grouped_columns
        ],
    }
    fit_indices, calibration_indices = (
        fedot_exec._exact_one_hot_final_fit_calibration_indices(target, seed=42)
    )
    transformer = fedot_exec._fit_exact_one_hot_grouped_transform(
        features.iloc[fit_indices], profile
    )
    observed = {}

    class FakeModel:
        def fit(self, fit_features, fit_target, **kwargs):
            observed["fit_shape"] = fit_features.shape
            observed["fit_target"] = np.asarray(fit_target)
            observed["fit_kwargs"] = kwargs

    def fake_estimator(
        seed,
        n_jobs,
        maximum_rounds,
        max_depth,
        native_categorical,
        fit_time_limit=None,
        use_eval_set=True,
    ):
        observed["estimator"] = {
            "seed": seed,
            "n_jobs": n_jobs,
            "maximum_rounds": maximum_rounds,
            "max_depth": max_depth,
            "native_categorical": native_categorical,
            "fit_time_limit": fit_time_limit,
            "use_eval_set": use_eval_set,
        }
        return FakeModel()

    monkeypatch.setattr(
        fedot_exec, "_exact_one_hot_xgboost_estimator", fake_estimator
    )
    model, eval_view, eval_target, fit_target = (
        fedot_exec._fit_exact_one_hot_xgboost_with_eval(
            features,
            target,
            transformer,
            seed=42,
            n_jobs=4,
            maximum_rounds=fedot_exec._GROUPED_ONE_HOT_XGBOOST_FINAL_ROUNDS,
            max_depth=7,
            native_categorical=True,
            fit_indices=fit_indices,
            use_eval_set=False,
        )
    )

    assert isinstance(model, FakeModel)
    assert len(fit_indices) == 950
    assert len(calibration_indices) == 50
    assert not np.intersect1d(fit_indices, calibration_indices).size
    assert np.bincount(target[fit_indices]).tolist() == [95] * 10
    assert np.bincount(target[calibration_indices]).tolist() == [5] * 10
    assert observed["fit_shape"] == (950, 6)
    assert np.array_equal(observed["fit_target"], target[fit_indices])
    assert observed["fit_kwargs"] == {}
    assert observed["estimator"]["maximum_rounds"] == 700
    assert observed["estimator"]["use_eval_set"] is False
    assert eval_view is None
    assert eval_target is None
    assert np.array_equal(fit_target, target[fit_indices])

    with pytest.raises(ValueError, match="eval indices require"):
        fedot_exec._fit_exact_one_hot_xgboost_with_eval(
            features,
            target,
            transformer,
            seed=42,
            n_jobs=4,
            maximum_rounds=700,
            max_depth=7,
            native_categorical=True,
            fit_indices=fit_indices,
            eval_indices=calibration_indices,
            use_eval_set=False,
        )


def _binary_probabilities(truth, confidence):
    confidence = np.broadcast_to(np.asarray(confidence, dtype=float), len(truth))
    positive = np.where(np.asarray(truth) == 1, confidence, 1.0 - confidence)
    return np.column_stack((1.0 - positive, positive))


def test_exact_one_hot_selector_accepts_clear_paired_gain():
    truth = np.arange(100) % 2
    baseline = _binary_probabilities(truth, 0.7)
    grouped = _binary_probabilities(truth, 0.9)

    selected, baseline_loss, grouped_loss, lower_95, required = (
        fedot_exec._select_exact_one_hot_grouped_representation(
            baseline, grouped, truth, labels=np.array([0, 1])
        )
    )

    assert selected is True
    assert baseline_loss - grouped_loss > required
    assert lower_95 > 0


def test_exact_one_hot_selector_rejects_uncertain_mean_gain():
    truth = np.arange(100) % 2
    baseline = _binary_probabilities(truth, 0.7)
    grouped = _binary_probabilities(
        truth, np.where(np.arange(100) < 50, 0.9, 0.55)
    )

    selected, baseline_loss, grouped_loss, lower_95, required = (
        fedot_exec._select_exact_one_hot_grouped_representation(
            baseline, grouped, truth, labels=np.array([0, 1])
        )
    )

    assert baseline_loss - grouped_loss > required
    assert lower_95 < 0
    assert selected is False


def test_exact_one_hot_selector_tie_keeps_baseline():
    truth = np.arange(100) % 2
    probabilities = _binary_probabilities(truth, 0.7)

    selected, baseline_loss, grouped_loss, lower_95, _ = (
        fedot_exec._select_exact_one_hot_grouped_representation(
            probabilities, probabilities, truth, labels=np.array([0, 1])
        )
    )

    assert baseline_loss == pytest.approx(grouped_loss)
    assert lower_95 == pytest.approx(0.0)
    assert selected is False


def _minimal_grouped_branch_inputs():
    features = pd.DataFrame({"x": np.arange(30), "y": np.arange(30) % 3})
    target = np.arange(30) % 3
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=features, y=target),
        test=SimpleNamespace(X=features.iloc[:6], y=target[:6]),
        encoded_class_count=3,
    )
    config = SimpleNamespace(
        metric="logloss",
        cores=4,
        seed=42,
        max_runtime_seconds=180,
        framework_params={"_portfolio": True},
    )
    return dataset, config


def test_classification_portfolio_returns_grouped_one_hot_branch(monkeypatch):
    dataset, config = _minimal_grouped_branch_inputs()
    profile = {"groups": [("g0", "g1", "g2")]}
    expected = {"branch": "grouped"}
    monkeypatch.setattr(
        fedot_exec,
        "_use_exact_one_hot_grouped_xgboost",
        lambda *args, **kwargs: profile,
    )
    monkeypatch.setattr(
        fedot_exec,
        "_run_exact_one_hot_grouped_xgboost",
        lambda **kwargs: expected,
    )

    output = fedot_exec._run_classification_portfolio(
        dataset, config, None, {}, 3.0, 3.0
    )

    assert output is expected


def test_classification_portfolio_falls_back_when_grouped_branch_fails(
    monkeypatch,
):
    dataset, config = _minimal_grouped_branch_inputs()

    class FallbackReached(Exception):
        pass

    monkeypatch.setattr(
        fedot_exec,
        "_use_exact_one_hot_grouped_xgboost",
        lambda *args, **kwargs: {"groups": [("g0", "g1", "g2")]},
    )

    def fail_grouped(**kwargs):
        raise RuntimeError("synthetic grouped failure")

    def reach_fallback(*args, **kwargs):
        raise FallbackReached

    monkeypatch.setattr(
        fedot_exec, "_run_exact_one_hot_grouped_xgboost", fail_grouped
    )
    monkeypatch.setattr(fedot_exec, "_compact_encode_if_needed", reach_fallback)

    with pytest.raises(FallbackReached):
        fedot_exec._run_classification_portfolio(
            dataset, config, None, {}, 3.0, 3.0
        )
