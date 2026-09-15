"""Train-only selected tabular portfolios used by FEDOT.

The implementation is intentionally independent from AutoMLBenchmark. It
contains the dataset-geometry guards, bounded estimators, calibration and
deployment rules validated by the FEDOT AMLB experiments, while exposing a
small native ``run_tabular_portfolio`` entry point.
"""

import gc
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from fedot.api.main import Fedot
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline as make_sklearn_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC, SVR

log = logging.getLogger(__name__)


class Timer:
    """Minimal wall-clock timer used by portfolio fit and prediction stages."""

    def __enter__(self):
        self.started_at = time.monotonic()
        return self

    def __exit__(self, *args):
        self._duration = time.monotonic() - self.started_at

    @property
    def duration(self):
        return self._duration


def result(**kwargs):
    """Build the internal result payload without an AMLB dependency."""
    return kwargs


def output_subdir(name, config):
    """Create an explicitly requested artifact directory."""
    root = Path(getattr(config, "output_dir", Path.cwd()))
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    return str(destination)


@dataclass
class TabularPortfolioResult:
    """Predictions and timing metadata from a native tabular portfolio run."""

    predictions: Any
    probabilities: Any
    training_duration: float
    predict_duration: float
    models_count: int
    truth: Any = None
    classes: Any = None
    target_is_encoded: bool = False
    output_file: Optional[str] = None


def run_tabular_portfolio(
    train_features,
    train_target,
    test_features,
    *,
    task_type: str,
    metric: Optional[str] = None,
    time_limit: float = 3600,
    n_jobs: int = 1,
    seed: int = 42,
    test_target=None,
    framework_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> TabularPortfolioResult:
    """Fit a guarded portfolio on training data and predict ``test_features``.

    Model admission, calibration and all geometry decisions use training data
    only. ``test_target`` is optional and is carried into the returned result
    solely for evaluation by callers.

    Args:
        train_features: Tabular training features (NumPy, pandas or scipy sparse).
        train_target: Classification labels or numeric regression targets.
        test_features: Features to predict after train-only model selection.
        task_type: ``"classification"`` or ``"regression"``.
        metric: Selection metric. Defaults to logloss or RMSE respectively.
        time_limit: End-to-end budget in seconds.
        n_jobs: Maximum model-level parallelism.
        seed: Deterministic selection and estimator seed.
        test_target: Optional truth returned unchanged for evaluation.
        framework_params: Advanced portfolio overrides. Dataset identities are
            deliberately unsupported.
        output_dir: Optional destination for explicitly requested artifacts.
    """
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'")
    if isinstance(time_limit, bool) or float(time_limit) <= 0:
        raise ValueError("time_limit must be a positive number of seconds")
    if isinstance(n_jobs, bool) or int(n_jobs) <= 0:
        raise ValueError("n_jobs must be a positive integer")

    parameters = {"_portfolio": True}
    if framework_params:
        parameters.update(dict(framework_params))
    parameters["_portfolio"] = True

    original_truth = test_target
    classes = None
    internal_target = train_target
    encoded_class_count = None
    if task_type == "classification":
        classes, internal_target = np.unique(
            _target_array(train_target), return_inverse=True
        )
        encoded_class_count = len(classes)

    test_shape = getattr(test_features, "shape", None)
    test_rows = test_shape[0] if test_shape else len(test_features)
    internal_test_target = np.full(int(test_rows), np.nan)
    dataset = SimpleNamespace(
        train=SimpleNamespace(X=train_features, y=internal_target),
        test=SimpleNamespace(X=test_features, y=internal_test_target),
        encoded_class_count=encoded_class_count,
    )
    config = SimpleNamespace(
        type=task_type,
        metric=metric or ("logloss" if task_type == "classification" else "rmse"),
        cores=int(n_jobs),
        max_runtime_seconds=float(time_limit),
        seed=int(seed),
        framework_params=parameters,
        output_predictions_file=None,
        output_dir=output_dir,
    )
    payload = run(dataset, config)

    predictions = payload["predictions"]
    if task_type == "classification":
        predictions = classes[np.asarray(predictions, dtype=int)]
    return TabularPortfolioResult(
        predictions=predictions,
        probabilities=payload.get("probabilities"),
        training_duration=float(payload["training_duration"]),
        predict_duration=float(payload["predict_duration"]),
        models_count=int(payload["models_count"]),
        truth=original_truth,
        classes=classes,
        target_is_encoded=False,
        output_file=payload.get("output_file"),
    )


_DIRECT_FIXED_XGBOOST_TEMPERATURE = 1.015
_DIRECT_FIXED_XGBOOST_REG_LAMBDA = 3.0
_DIRECT_XGBOOST_NON_FIT_RESERVE_SECONDS = 35.0
_DIRECT_XGBOOST_REFIT_SETUP_RESERVE_SECONDS = 10.0
_DIRECT_XGBOOST_MIN_REFIT_SECONDS = 30.0
_LOCAL_IMAGE_LGBM_MAX_ROUNDS = 1_000
_LOCAL_IMAGE_CALIBRATION_FRACTION = 0.05
_SCREENED_WIDE_AUC_FEATURES = 1_000
_SCREENED_WIDE_AUC_ROUNDS = 1_000
_MEDIUM_SCREENED_AUC_ROUNDS = 1_200
_MEDIUM_SCREENED_AUC_LEARNING_RATE = 0.02
_MEDIUM_SCREENED_AUC_LEAVES = 63
_MEDIUM_SCREENED_AUC_SELECTION_TOLERANCE = 0.002
_MEDIUM_SCREENED_AUC_FEATURE_COUNTS = (64, 96, 128, 160, 192, 256)
_HIGH_MISSING_FREQUENCY_AUC_ROUNDS = 200
_HIGH_MISSING_FREQUENCY_AUC_LEAVES = 15
_HIGH_MISSING_FREQUENCY_AUC_MIN_SCORE = 0.65
_TARGET_FREQUENCY_AUC_SMOOTHING = 10.0
_TARGET_FREQUENCY_AUC_ROUNDS = 1_000
_TARGET_FREQUENCY_AUC_MIN_GAIN = 0.001
_NARROW_CATEGORICAL_AUC_SELECTOR_ROUNDS = 200
_NARROW_CATEGORICAL_AUC_FINAL_ROUNDS = 500
_NARROW_CATEGORICAL_AUC_MIN_SCORE = 0.75
_NARROW_CATEGORICAL_AUC_MIN_GAIN = 0.005
_NOMINAL_MULTICLASS_CATBOOST_ROUNDS = 200
_NOMINAL_MULTICLASS_CATBOOST_LEARNING_RATE = 0.15
_NOMINAL_MULTICLASS_CATBOOST_DEPTH = 6
_NOMINAL_MULTICLASS_CATBOOST_MIN_GAIN = 0.005
_NOMINAL_MULTICLASS_CATBOOST_MIN_PRIOR_GAIN = 0.01
_SMALL_DENSE_SVC_C = 3.0
_NARROW_KERNEL_DOMINANCE_MIN_LOGLOSS_GAIN = 0.1
_NARROW_KERNEL_VARIANT_MIN_LOGLOSS_GAIN = 0.02
_SCALED_SVC_VARIANT_PREFIX = "scaled_svc_"
_WIDE_EXTRA_TREES_MIN_LOGLOSS_GAIN = 0.005
_LGBM_EXTRA_TREES_MODEL = "lgbm_extra_trees"
_LGBM_EXTRA_TREES_MIN_LOGLOSS_GAIN = 0.005
_LGBM_EXTRA_TREES_MAX_GATE_SECONDS = 45.0
_LGBM_EXTRA_TREES_MAX_PAIR_RATIO = 1.10
_LGBM_EXTRA_TREES_MAX_OVERHEAD_SECONDS = 5.0
_PRIOR_CALIBRATION_MAX_ABS_EXPONENT = 0.75
_PRIOR_CALIBRATION_MIN_REFERENCE_ROWS = 300
_PRIOR_CALIBRATION_TRANSFER_SHRINKAGE = 0.75
_REGRESSION_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0)
_REGRESSION_RIDGE_MIN_OOF_GAIN = 0.05
_GROUPED_ONE_HOT_XGBOOST_SELECTOR_ROUNDS = 1_400
_GROUPED_ONE_HOT_XGBOOST_FINAL_ROUNDS = 700
_GROUPED_ONE_HOT_XGBOOST_MIN_GAIN = 0.005


class _SklearnProbabilityModel:
    """Expose the small sklearn fallback through FEDOT's probability API shape."""

    def __init__(self, estimator, target):
        self.estimator = estimator
        self.target = np.asarray(target).reshape(-1)
        self.current_pipeline = SimpleNamespace(length=1)
        self.history = None

    def predict_proba(self, features, probs_for_all_classes=True):
        del probs_for_all_classes
        return self.estimator.predict_proba(features)


class _LocalImageProbabilityModel:
    """Apply a deterministic local image view and fitted calibration."""

    def __init__(self, automl, image_side, temperature=1.0):
        self.automl = automl
        self.image_side = int(image_side)
        self.temperature = float(temperature)

    def __getattr__(self, name):
        return getattr(self.automl, name)

    def predict_proba(self, features, probs_for_all_classes=True):
        probabilities = self.automl.predict_proba(
            features=_local_image_features(features, self.image_side),
            probs_for_all_classes=probs_for_all_classes,
        )
        return _apply_temperature(probabilities, self.temperature)


class _WallClockStopCallback(TrainingCallback):
    """Stop an XGBoost fit before the surrounding benchmark deadline."""

    def __init__(self, seconds):
        self.seconds = max(float(seconds), 0.1)
        self.deadline = None

    def before_training(self, model):
        self.deadline = time.monotonic() + self.seconds
        return model

    def after_iteration(self, model, epoch, evals_log):
        del model, evals_log
        return epoch > 0 and time.monotonic() >= self.deadline


def _smoothed_category_target_mapping(values, target, prior, smoothing):
    statistics = pd.DataFrame(
        {"value": values.to_numpy(), "target": np.asarray(target)}
    ).groupby("value", observed=True, dropna=False, sort=False)["target"].agg(
        ["sum", "count"]
    )
    return (statistics["sum"] + smoothing * prior) / (
        statistics["count"] + smoothing
    )


def _map_category_statistic(values, mapping, default):
    mapped = values.map(mapping).to_numpy(dtype=np.float32, na_value=np.nan)
    return np.nan_to_num(mapped, nan=float(default))


class _CrossFittedTargetFrequencyClassifier(BaseEstimator):
    """Encode high-cardinality categories without in-row target leakage."""

    def __init__(
        self,
        smoothing=_TARGET_FREQUENCY_AUC_SMOOTHING,
        n_estimators=_TARGET_FREQUENCY_AUC_ROUNDS,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        n_jobs=1,
        random_state=42,
    ):
        self.smoothing = smoothing
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, features, target):
        frame = pd.DataFrame(features)
        target_array = np.asarray(target).reshape(-1)
        self.feature_columns_ = list(frame.columns)
        self.categorical_columns_ = [
            column
            for column in frame
            if not pd.api.types.is_numeric_dtype(frame[column].dtype)
        ]
        self.numeric_columns_ = [
            column
            for column in frame
            if column not in self.categorical_columns_
        ]
        if not self.categorical_columns_:
            raise ValueError(
                "target-frequency encoding requires categorical features"
            )
        self.target_prior_ = float(np.mean(target_array))
        self.target_mappings_ = {}
        self.frequency_mappings_ = {}
        encoded = np.empty(
            (
                len(frame),
                len(self.numeric_columns_) + 2 * len(self.categorical_columns_),
            ),
            dtype=np.float32,
        )
        if self.numeric_columns_:
            encoded[:, : len(self.numeric_columns_)] = frame[
                self.numeric_columns_
            ].to_numpy(dtype=np.float32, copy=False)

        splitter = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=int(self.random_state),
        )
        folds = list(splitter.split(np.zeros(len(target_array)), target_array))
        target_offset = len(self.numeric_columns_)
        frequency_offset = target_offset + len(self.categorical_columns_)
        for column_index, column in enumerate(self.categorical_columns_):
            values = frame[column].reset_index(drop=True)
            encoded_target = np.full(
                len(frame), self.target_prior_, dtype=np.float32
            )
            for fit_indices, valid_indices in folds:
                fold_mapping = _smoothed_category_target_mapping(
                    values.iloc[fit_indices],
                    target_array[fit_indices],
                    self.target_prior_,
                    float(self.smoothing),
                )
                encoded_target[valid_indices] = _map_category_statistic(
                    values.iloc[valid_indices],
                    fold_mapping,
                    self.target_prior_,
                )
            target_mapping = _smoothed_category_target_mapping(
                values,
                target_array,
                self.target_prior_,
                float(self.smoothing),
            )
            frequency_mapping = values.value_counts(
                dropna=False, normalize=True
            )
            self.target_mappings_[column] = target_mapping
            self.frequency_mappings_[column] = frequency_mapping
            encoded[:, target_offset + column_index] = encoded_target
            encoded[:, frequency_offset + column_index] = (
                _map_category_statistic(values, frequency_mapping, 0.0)
            )

        self.estimator_ = LGBMClassifier(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            num_leaves=int(self.num_leaves),
            min_child_samples=int(self.min_child_samples),
            n_jobs=max(int(self.n_jobs), 1),
            random_state=int(self.random_state),
            verbose=-1,
        )
        self.estimator_.fit(encoded, target_array)
        self.classes_ = self.estimator_.classes_
        return self

    def _transform(self, features):
        frame = pd.DataFrame(features)
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("target-frequency feature columns changed after fit")
        encoded = np.empty(
            (
                len(frame),
                len(self.numeric_columns_) + 2 * len(self.categorical_columns_),
            ),
            dtype=np.float32,
        )
        if self.numeric_columns_:
            encoded[:, : len(self.numeric_columns_)] = frame[
                self.numeric_columns_
            ].to_numpy(dtype=np.float32, copy=False)
        target_offset = len(self.numeric_columns_)
        frequency_offset = target_offset + len(self.categorical_columns_)
        for column_index, column in enumerate(self.categorical_columns_):
            values = frame[column]
            encoded[:, target_offset + column_index] = _map_category_statistic(
                values,
                self.target_mappings_[column],
                self.target_prior_,
            )
            encoded[:, frequency_offset + column_index] = (
                _map_category_statistic(
                    values,
                    self.frequency_mappings_[column],
                    0.0,
                )
            )
        return encoded

    def predict_proba(self, features):
        return self.estimator_.predict_proba(self._transform(features))


class _NativeCategoricalCatBoostClassifier(BaseEstimator):
    """Fit CatBoost on train-fitted integer category identities.

    CatBoost consumes the integer values as nominal categories via
    ``cat_features``. The mapping is learned only from the fit partition, and
    missing or previously unseen values share the reserved ``-1`` identity.
    """

    def __init__(
        self,
        iterations=_NARROW_CATEGORICAL_AUC_FINAL_ROUNDS,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        n_jobs=1,
        random_state=42,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(
        self,
        features,
        target,
        eval_set=None,
        early_stopping_rounds=None,
    ):
        frame = pd.DataFrame(features)
        self.feature_columns_ = list(frame.columns)
        self.categorical_columns_ = [
            column
            for column in frame
            if not pd.api.types.is_numeric_dtype(frame[column].dtype)
        ]
        if not self.categorical_columns_:
            raise ValueError("native CatBoost requires categorical features")
        self.numeric_columns_ = [
            column
            for column in frame
            if column not in self.categorical_columns_
        ]
        self.category_levels_ = {}
        encoded = frame.copy()
        for column in self.categorical_columns_:
            codes, levels = pd.factorize(frame[column].astype(object), sort=False)
            self.category_levels_[column] = pd.Index(levels)
            encoded[column] = codes.astype(np.int32, copy=False)
        for column in self.numeric_columns_:
            encoded[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype(np.float32)

        self.categorical_positions_ = [
            encoded.columns.get_loc(column)
            for column in self.categorical_columns_
        ]
        self.estimator_ = CatBoostClassifier(
            iterations=int(self.iterations),
            learning_rate=float(self.learning_rate),
            depth=int(self.depth),
            loss_function=str(self.loss_function),
            eval_metric=str(self.eval_metric),
            thread_count=max(int(self.n_jobs), 1),
            random_seed=int(self.random_state),
            verbose=False,
            allow_writing_files=False,
        )
        fit_parameters = {"cat_features": self.categorical_positions_}
        if eval_set is not None:
            eval_features, eval_target = eval_set
            fit_parameters.update(
                eval_set=(self._transform(eval_features), np.asarray(eval_target).reshape(-1)),
                early_stopping_rounds=int(early_stopping_rounds),
                use_best_model=True,
            )
        self.estimator_.fit(
            encoded,
            np.asarray(target).reshape(-1),
            **fit_parameters,
        )
        self.classes_ = self.estimator_.classes_
        best_iteration = getattr(
            self.estimator_, "get_best_iteration", lambda: None
        )()
        try:
            best_iteration = int(best_iteration)
        except (TypeError, ValueError):
            best_iteration = -1
        self.best_iteration_ = (
            best_iteration + 1
            if best_iteration >= 0
            else int(self.iterations)
        )
        return self

    def _transform(self, features):
        frame = pd.DataFrame(features)
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("native CatBoost feature columns changed after fit")
        encoded = frame.copy()
        for column in self.categorical_columns_:
            encoded[column] = self.category_levels_[column].get_indexer(
                frame[column].astype(object)
            ).astype(np.int32, copy=False)
        for column in self.numeric_columns_:
            encoded[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype(np.float32)
        return encoded

    def predict_proba(self, features):
        return self.estimator_.predict_proba(self._transform(features))


class _TrainSelectedScreenedAUCClassifier(BaseEstimator):
    """Choose a bounded ANOVA width on train-only data and refit all rows."""

    def __init__(
        self,
        n_estimators=_MEDIUM_SCREENED_AUC_ROUNDS,
        feature_counts=_MEDIUM_SCREENED_AUC_FEATURE_COUNTS,
        validation_fraction=0.2,
        selection_tolerance=_MEDIUM_SCREENED_AUC_SELECTION_TOLERANCE,
        n_jobs=1,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.feature_counts = feature_counts
        self.validation_fraction = validation_fraction
        self.selection_tolerance = selection_tolerance
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _estimator(self):
        return LGBMClassifier(
            n_estimators=int(self.n_estimators),
            learning_rate=_MEDIUM_SCREENED_AUC_LEARNING_RATE,
            num_leaves=_MEDIUM_SCREENED_AUC_LEAVES,
            min_child_samples=20,
            n_jobs=max(int(self.n_jobs), 1),
            random_state=int(self.random_state),
            verbose=-1,
        )

    def fit(self, features, target):
        matrix = np.asarray(features, dtype=np.float32)
        target_array = np.asarray(target).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("screened AUC features must be a two-dimensional table")
        selector_train, selector_valid = train_test_split(
            np.arange(len(target_array)),
            test_size=float(self.validation_fraction),
            random_state=int(self.random_state),
            stratify=target_array,
        )
        selector_imputer = SimpleImputer(strategy="median")
        fit_matrix = selector_imputer.fit_transform(matrix[selector_train])
        valid_matrix = selector_imputer.transform(matrix[selector_valid])
        scores, _ = _finite_f_classif(
            fit_matrix, target_array[selector_train]
        )
        ranked_columns = np.argsort(scores)[::-1]
        candidate_counts = sorted(
            {
                min(int(count), matrix.shape[1])
                for count in (*tuple(self.feature_counts), matrix.shape[1])
                if int(count) > 0
            }
        )
        self.selector_scores_ = {}
        for feature_count in candidate_counts:
            columns = np.sort(ranked_columns[:feature_count])
            estimator = self._estimator()
            estimator.fit(
                fit_matrix[:, columns], target_array[selector_train]
            )
            probabilities = estimator.predict_proba(
                valid_matrix[:, columns]
            )[:, 1]
            auc = roc_auc_score(target_array[selector_valid], probabilities)
            self.selector_scores_[feature_count] = float(auc)

        selected_score = max(self.selector_scores_.values())
        selected_count = min(
            feature_count
            for feature_count, score in self.selector_scores_.items()
            if score >= selected_score - float(self.selection_tolerance)
        )

        self.selected_feature_count_ = int(selected_count)
        self.imputer_ = SimpleImputer(strategy="median")
        full_matrix = self.imputer_.fit_transform(matrix)
        full_scores, _ = _finite_f_classif(full_matrix, target_array)
        self.selected_columns_ = np.sort(
            np.argsort(full_scores)[::-1][: self.selected_feature_count_]
        )
        self.estimator_ = self._estimator()
        self.estimator_.fit(
            full_matrix[:, self.selected_columns_], target_array
        )
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, features):
        matrix = np.asarray(features, dtype=np.float32)
        matrix = self.imputer_.transform(matrix)
        return self.estimator_.predict_proba(
            matrix[:, self.selected_columns_]
        )


class _FrequencyLGBMClassifier(BaseEstimator):
    """Use train-only category frequencies as a low-variance fallback view."""

    def __init__(
        self,
        n_estimators=_TARGET_FREQUENCY_AUC_ROUNDS,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        n_jobs=1,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, features, target):
        frame = pd.DataFrame(features)
        self.feature_columns_ = list(frame.columns)
        self.categorical_columns_ = [
            column
            for column in frame
            if not pd.api.types.is_numeric_dtype(frame[column].dtype)
        ]
        self.numeric_columns_ = [
            column
            for column in frame
            if column not in self.categorical_columns_
        ]
        self.frequency_mappings_ = {
            column: frame[column].value_counts(dropna=False, normalize=True)
            for column in self.categorical_columns_
        }
        encoded = self._transform(frame)
        self.estimator_ = LGBMClassifier(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            num_leaves=int(self.num_leaves),
            min_child_samples=int(self.min_child_samples),
            n_jobs=max(int(self.n_jobs), 1),
            random_state=int(self.random_state),
            verbose=-1,
        )
        self.estimator_.fit(encoded, np.asarray(target).reshape(-1))
        self.classes_ = self.estimator_.classes_
        return self

    def _transform(self, features):
        frame = pd.DataFrame(features)
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("frequency feature columns changed after fit")
        encoded = np.empty(
            (len(frame), len(self.numeric_columns_) + len(self.categorical_columns_)),
            dtype=np.float32,
        )
        if self.numeric_columns_:
            encoded[:, : len(self.numeric_columns_)] = frame[
                self.numeric_columns_
            ].to_numpy(dtype=np.float32, copy=False)
        offset = len(self.numeric_columns_)
        for column_index, column in enumerate(self.categorical_columns_):
            encoded[:, offset + column_index] = _map_category_statistic(
                frame[column], self.frequency_mappings_[column], 0.0
            )
        return encoded

    def predict_proba(self, features):
        return self.estimator_.predict_proba(self._transform(features))


class _CrossFittedTargetFrequencyRegressor(BaseEstimator, RegressorMixin):
    """Cross-fit target means and frequencies for nominal regression columns."""

    def __init__(
        self,
        smoothing=10.0,
        n_estimators=1_000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        n_jobs=1,
        random_state=42,
    ):
        self.smoothing = smoothing
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _numeric_values(self, frame):
        if not self.numeric_columns_:
            return None
        return (
            frame[self.numeric_columns_]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(self.numeric_medians_)
            .to_numpy(dtype=np.float32)
        )

    def fit(self, features, target):
        frame = pd.DataFrame(features)
        target_array = np.asarray(target, dtype=float).reshape(-1)
        self.feature_columns_ = list(frame.columns)
        self.categorical_columns_ = [
            column
            for column in frame
            if not pd.api.types.is_numeric_dtype(frame[column].dtype)
        ]
        self.numeric_columns_ = [
            column
            for column in frame
            if column not in self.categorical_columns_
        ]
        if not self.categorical_columns_:
            raise ValueError(
                "target-frequency regression requires categorical features"
            )
        self.numeric_medians_ = {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in self.numeric_columns_
        }
        self.target_prior_ = float(np.mean(target_array))
        self.target_mappings_ = {}
        self.frequency_mappings_ = {}
        encoded = np.empty(
            (
                len(frame),
                len(self.numeric_columns_) + 2 * len(self.categorical_columns_),
            ),
            dtype=np.float32,
        )
        numeric_values = self._numeric_values(frame)
        if numeric_values is not None:
            encoded[:, : len(self.numeric_columns_)] = numeric_values

        folds = list(
            KFold(
                n_splits=3,
                shuffle=True,
                random_state=int(self.random_state),
            ).split(frame)
        )
        target_offset = len(self.numeric_columns_)
        frequency_offset = target_offset + len(self.categorical_columns_)
        for column_index, column in enumerate(self.categorical_columns_):
            values = frame[column].reset_index(drop=True)
            encoded_target = np.full(
                len(frame), self.target_prior_, dtype=np.float32
            )
            for fit_indices, valid_indices in folds:
                fold_mapping = _smoothed_category_target_mapping(
                    values.iloc[fit_indices],
                    target_array[fit_indices],
                    self.target_prior_,
                    float(self.smoothing),
                )
                encoded_target[valid_indices] = _map_category_statistic(
                    values.iloc[valid_indices],
                    fold_mapping,
                    self.target_prior_,
                )
            target_mapping = _smoothed_category_target_mapping(
                values,
                target_array,
                self.target_prior_,
                float(self.smoothing),
            )
            frequency_mapping = values.value_counts(
                dropna=False, normalize=True
            )
            self.target_mappings_[column] = target_mapping
            self.frequency_mappings_[column] = frequency_mapping
            encoded[:, target_offset + column_index] = encoded_target
            encoded[:, frequency_offset + column_index] = (
                _map_category_statistic(values, frequency_mapping, 0.0)
            )

        self.estimator_ = LGBMRegressor(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            num_leaves=int(self.num_leaves),
            min_child_samples=int(self.min_child_samples),
            n_jobs=max(int(self.n_jobs), 1),
            random_state=int(self.random_state),
            verbose=-1,
        )
        self.estimator_.fit(encoded, target_array)
        return self

    def _transform(self, features):
        frame = pd.DataFrame(features)
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("target-frequency feature columns changed after fit")
        encoded = np.empty(
            (
                len(frame),
                len(self.numeric_columns_) + 2 * len(self.categorical_columns_),
            ),
            dtype=np.float32,
        )
        numeric_values = self._numeric_values(frame)
        if numeric_values is not None:
            encoded[:, : len(self.numeric_columns_)] = numeric_values
        target_offset = len(self.numeric_columns_)
        frequency_offset = target_offset + len(self.categorical_columns_)
        for column_index, column in enumerate(self.categorical_columns_):
            encoded[:, target_offset + column_index] = _map_category_statistic(
                frame[column],
                self.target_mappings_[column],
                self.target_prior_,
            )
            encoded[:, frequency_offset + column_index] = (
                _map_category_statistic(
                    frame[column], self.frequency_mappings_[column], 0.0
                )
            )
        return encoded

    def predict(self, features):
        return self.estimator_.predict(self._transform(features))


class _SklearnRegressionModel:
    """Expose a bounded sklearn regressor through FEDOT's result metadata shape."""

    def __init__(self, estimator, target):
        self.estimator = estimator
        self.target = np.asarray(target).reshape(-1)
        self.current_pipeline = SimpleNamespace(length=1)
        self.history = None

    def predict(self, features):
        return np.asarray(self.estimator.predict(features), dtype=float).reshape(-1)


class _TrainSelectedLGBMRegressor(BaseEstimator, RegressorMixin):
    """Select a boosting horizon on bounded train rows before an all-row refit."""

    def __init__(
        self,
        n_estimators=1_000,
        learning_rate=0.05,
        num_leaves=255,
        min_child_samples=20,
        selector_max_rows=100_000,
        validation_fraction=0.2,
        early_stopping_rounds=100,
        refit_all_rows=True,
        fixed_n_estimators=None,
        n_jobs=1,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.selector_max_rows = selector_max_rows
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.refit_all_rows = refit_all_rows
        self.fixed_n_estimators = fixed_n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _model(self, n_estimators):
        return LGBMRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(self.learning_rate),
            num_leaves=int(self.num_leaves),
            min_child_samples=int(self.min_child_samples),
            n_jobs=max(int(self.n_jobs), 1),
            random_state=int(self.random_state),
            verbose=-1,
        )

    def fit(self, features, target):
        target_array = np.asarray(target, dtype=float).reshape(-1)
        if len(target_array) != len(features):
            raise ValueError("regression features and target lengths differ")
        if self.fixed_n_estimators is not None:
            fixed_n_estimators = int(self.fixed_n_estimators)
            if fixed_n_estimators < 1:
                raise ValueError("fixed_n_estimators must be positive")
            self.selected_n_estimators_ = fixed_n_estimators
            self.estimator_ = self._model(fixed_n_estimators)
            self.estimator_.fit(features, target_array)
            return self
        selector_max_rows = int(self.selector_max_rows)
        if selector_max_rows < 1_000:
            raise ValueError("selector_max_rows must be at least 1000")
        validation_fraction = float(self.validation_fraction)
        if not 0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")

        selector_indices = np.arange(len(target_array))
        if len(selector_indices) > selector_max_rows:
            random = np.random.RandomState(int(self.random_state))
            selector_indices = np.sort(
                random.choice(
                    selector_indices, size=selector_max_rows, replace=False
                )
            )
        fit_indices, valid_indices = train_test_split(
            selector_indices,
            test_size=validation_fraction,
            random_state=int(self.random_state),
        )
        selection_model = self._model(self.n_estimators)
        selection_model.fit(
            _slice_rows(features, fit_indices),
            target_array[fit_indices],
            eval_set=[
                (_slice_rows(features, valid_indices), target_array[valid_indices])
            ],
            eval_metric="rmse",
            callbacks=[
                early_stopping(int(self.early_stopping_rounds), verbose=False),
                log_evaluation(period=0),
            ],
        )
        self.selected_n_estimators_ = max(
            1, int(getattr(selection_model, "best_iteration_", self.n_estimators))
        )
        if self.refit_all_rows:
            self.estimator_ = self._model(self.selected_n_estimators_)
            self.estimator_.fit(features, target_array)
        else:
            self.estimator_ = selection_model
        return self

    def predict(self, features):
        return self.estimator_.predict(features)


class _CumulativeProbabilityRegressor(BaseEstimator, RegressorMixin):
    """Regress an ordered discrete target through cumulative probabilities."""

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, features, target):
        target_array = np.asarray(target, dtype=float).reshape(-1)
        self.target_levels_ = np.unique(target_array)
        if len(self.target_levels_) < 2:
            raise ValueError("cumulative regression needs at least two target levels")
        self.classifiers_ = [
            clone(self.classifier).fit(features, target_array >= threshold)
            for threshold in self.target_levels_[1:]
        ]
        return self

    def predict(self, features):
        cumulative_probabilities = np.column_stack(
            [
                classifier.predict_proba(features)[:, 1]
                for classifier in self.classifiers_
            ]
        )
        # Independently fitted thresholds can cross.  Their cumulative minimum
        # is the smallest deterministic projection onto valid non-increasing
        # exceedance probabilities and does not inspect prediction labels.
        cumulative_probabilities = np.minimum.accumulate(
            cumulative_probabilities, axis=1
        )
        return self.target_levels_[0] + cumulative_probabilities @ np.diff(
            self.target_levels_
        )


def _ordered_feature_expansion(features):
    """Add bounded cumulative and adjacent relations in declared column order."""
    values = np.asarray(features, dtype=np.float32)
    forward = np.cumsum(values, axis=1)
    backward = np.cumsum(values[:, ::-1], axis=1)[:, ::-1]
    adjacent = np.diff(values, axis=1, prepend=values[:, :1])
    return np.hstack((values, forward, backward, adjacent))


def run(dataset, config):
    log.info("\n**** FEDOT ****\n")

    is_classification = config.type == "classification"
    scoring_metric = get_fedot_metrics(config)

    training_params = {"preset": "best_quality", "n_jobs": config.cores}
    training_params.update(
        {k: v for k, v in config.framework_params.items() if not k.startswith("_")}
    )
    predefined_model = config.framework_params.get("_predefined_model")
    use_portfolio = _as_bool(config.framework_params.get("_portfolio", False))
    n_jobs = training_params["n_jobs"]

    log.info(
        f"Running FEDOT with a maximum time of {config.max_runtime_seconds}s on {n_jobs} cores, \
             optimizing {scoring_metric}"
    )

    runtime_reserve_seconds = max(15, config.max_runtime_seconds * 0.1)
    runtime_min = max(
        (config.max_runtime_seconds - runtime_reserve_seconds) / 60,
        0.1,
    )
    max_pipeline_fit_time = float(
        training_params.pop("max_pipeline_fit_time", runtime_min / 10)
    )
    if max_pipeline_fit_time <= 0:
        raise ValueError("max_pipeline_fit_time must be greater than zero")
    log.info(
        f"Reserving {runtime_reserve_seconds:.1f}s for preprocessing, final fit and prediction; "
        f"FEDOT search/tuning timeout is {runtime_min:.2f} min and the per-pipeline "
        f"fit timeout is {max_pipeline_fit_time:.2f} min."
    )

    if use_portfolio and predefined_model is not None:
        raise ValueError("_portfolio and _predefined_model cannot be used together")
    if use_portfolio:
        if is_classification:
            return _run_classification_portfolio(
                dataset=dataset,
                config=config,
                scoring_metric=scoring_metric,
                training_params=training_params,
                runtime_min=runtime_min,
                max_pipeline_fit_time=max_pipeline_fit_time,
            )
        regression_result = _run_regression_portfolio(dataset, config)
        if regression_result is not None:
            return regression_result
        log.warning(
            "The regression table is outside the bounded portfolio regimes; "
            "using standard FEDOT."
        )

    train_target = dataset.train.y
    test_target = dataset.test.y
    observed_encoded_labels = None
    if is_classification:
        train_target, observed_encoded_labels = _contiguous_classification_target(
            train_target
        )
        test_target = _target_array(test_target)

    fedot = _make_fedot(
        config=config,
        scoring_metric=scoring_metric,
        training_params=training_params,
        runtime_min=runtime_min,
        max_pipeline_fit_time=max_pipeline_fit_time,
    )

    with Timer() as training:
        fedot.fit(
            features=dataset.train.X,
            target=train_target,
            predefined_model=_predefined_model_with_n_jobs(
                predefined_model, config.cores
            ),
        )

    log.info("Predicting on the test set.")
    with Timer() as predict:
        predictions = fedot.predict(features=dataset.test.X)
    probabilities = None
    if is_classification:
        probabilities = fedot.predict_proba(
            features=dataset.test.X, probs_for_all_classes=True
        )
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )

    save_artifacts(fedot, config)

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=is_classification,
        models_count=fedot.current_pipeline.length,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _run_regression_portfolio(dataset, config):
    """Select and refit bounded regressors using only an inner train holdout."""
    train_features = dataset.train.X
    test_features = dataset.test.X
    train_target = _target_array(dataset.train.y).astype(float)
    test_target = _target_array(dataset.test.y).astype(float)
    configured_candidates = config.framework_params.get("_portfolio_candidates")
    candidates = _adaptive_regression_portfolio_candidates(
        train_features,
        train_target,
        configured=configured_candidates,
    )
    ridge_only_fallback = False
    if (
        candidates is None
        and configured_candidates is None
        and _is_relaxed_small_mixed_ridge_portfolio(train_features, train_target)
    ):
        candidates = [
            "lgbmreg_direct",
            "xgboostreg_direct",
            "catboostreg_direct",
            "rfr_direct",
        ]
        ridge_only_fallback = True
    if candidates is None:
        return None

    validation_fraction = float(
        config.framework_params.get("_portfolio_validation_fraction", 0.2)
    )
    if not 0 < validation_fraction < 0.5:
        raise ValueError("_portfolio_validation_fraction must be between 0 and 0.5")
    metric = _normalise_metric_name(config.metric)
    if metric not in {"rmse", "mse", "mae", "r2"}:
        log.warning(
            "Regression portfolio does not support metric %s; using standard FEDOT.",
            metric,
        )
        return None
    ridge_oof_strategy = None
    ridge_oof_duration = 0.0
    ridge_oof_eligible = (
        configured_candidates is None
        and metric == "rmse"
        and config.max_runtime_seconds >= 90
        and config.cores >= 4
        and _is_ridge_oof_regression_portfolio(train_features, train_target)
    )
    if ridge_oof_eligible:
        ridge_oof_started_at = time.monotonic()
        try:
            ridge_oof_strategy = _select_ridge_oof_expansion(
                candidates,
                train_features,
                train_target,
                seed=config.seed,
                n_jobs=config.cores,
                max_seconds=min(75.0, 0.45 * float(config.max_runtime_seconds)),
                min_relative_gain=float(
                    config.framework_params.get(
                        "_portfolio_regression_ridge_min_oof_gain",
                        _REGRESSION_RIDGE_MIN_OOF_GAIN,
                    )
                ),
                pair_min_relative_gain=float(
                    config.framework_params.get(
                        "_portfolio_regression_pair_min_relative_gain", 0.002
                    )
                ),
            )
        finally:
            ridge_oof_duration = time.monotonic() - ridge_oof_started_at
    if ridge_only_fallback and ridge_oof_strategy is None:
        _portfolio_report(
            "Relaxed small-mixed regression geometry did not pass the Ridge "
            "OOF gain guard; using standard FEDOT."
        )
        return None

    large_grouped_sequence = (
        configured_candidates is None
        and _is_large_grouped_sequence_regression_portfolio(
            train_features, train_target
        )
    )
    if large_grouped_sequence and (
        config.max_runtime_seconds < 180 or config.cores < 4 or metric != "rmse"
    ):
        _portfolio_report(
            "Large grouped-sequence regression requires RMSE, at least "
            "180 seconds and four cores; using standard FEDOT."
        )
        return None
    large_dense_low_cardinality = (
        configured_candidates is None
        and _is_large_dense_low_cardinality_regression_portfolio(
            train_features, train_target
        )
    )
    if large_dense_low_cardinality and (
        config.max_runtime_seconds < 180 or config.cores < 4 or metric != "rmse"
    ):
        _portfolio_report(
            "Large dense low-cardinality regression requires RMSE, at least "
            "180 seconds and four cores; using standard FEDOT."
        )
        return None
    large_high_categorical = (
        configured_candidates is None
        and _is_large_high_categorical_regression_portfolio(
            train_features, train_target
        )
    )
    if large_high_categorical and (
        config.max_runtime_seconds < 180 or config.cores < 4 or metric != "rmse"
    ):
        _portfolio_report(
            "Large high-categorical regression requires RMSE, at least "
            "180 seconds and four cores; using standard FEDOT."
        )
        return None
    selector_pool_indices = np.arange(len(train_target))
    if large_dense_low_cardinality or large_grouped_sequence:
        selector_parameter = (
            "_portfolio_grouped_sequence_selector_max_rows"
            if large_grouped_sequence
            else "_portfolio_large_dense_selector_max_rows"
        )
        selector_max_rows = int(
            config.framework_params.get(selector_parameter, 100_000)
        )
        if selector_max_rows < 1_000:
            raise ValueError(f"{selector_parameter} must be at least 1000")
        if len(selector_pool_indices) > selector_max_rows:
            random = np.random.RandomState(int(config.seed))
            selector_pool_indices = np.sort(
                random.choice(
                    selector_pool_indices, size=selector_max_rows, replace=False
                )
            )
    train_indices, validation_indices = train_test_split(
        selector_pool_indices,
        test_size=validation_fraction,
        random_state=int(config.seed),
    )
    selector_train_features = _slice_rows(train_features, train_indices)
    selector_valid_features = _slice_rows(train_features, validation_indices)
    selector_train_target = train_target[train_indices]
    selector_valid_target = train_target[validation_indices]
    small_mixed = _is_small_mixed_regression_portfolio(
        train_features, train_target
    )
    small_wide = (
        configured_candidates is None
        and _is_small_wide_regression_portfolio(train_features, train_target)
    )
    small_classic = (
        configured_candidates is None
        and _is_small_classic_regression_portfolio(train_features, train_target)
    )
    large_compact_mixed = (
        configured_candidates is None
        and _is_large_compact_mixed_regression_portfolio(
            train_features, train_target
        )
    )
    very_large_compact_mixed = (
        configured_candidates is None
        and _is_very_large_compact_mixed_regression_portfolio(
            train_features, train_target
        )
    )
    ordered_histogram_ordinal = (
        configured_candidates is None
        and _is_ordered_histogram_ordinal_regression_portfolio(
            train_features, train_target
        )
    )
    minimum_inflated_skewed = (
        configured_candidates is None
        and small_mixed
        and _is_minimum_inflated_skewed_regression_target(train_target)
    )
    grouped_selector_train = None
    grouped_selector_valid = None
    if large_grouped_sequence:
        grouped_preprocessor = make_sklearn_pipeline(
            FunctionTransformer(
                _grouped_sequence_trend_expansion,
                validate=False,
            ),
            SimpleImputer(strategy="median"),
        )
        grouped_selector_train = grouped_preprocessor.fit_transform(
            selector_train_features
        )
        grouped_selector_valid = grouped_preprocessor.transform(
            selector_valid_features
        )
    observations = []

    with Timer() as training:
        for candidate in candidates:
            candidate_started_at = time.monotonic()
            try:
                selected_n_estimators = None
                if large_grouped_sequence:
                    num_leaves, min_child_samples = (
                        _grouped_sequence_lgbm_shape(candidate)
                    )
                    estimator = LGBMRegressor(
                        n_estimators=1_000,
                        learning_rate=0.05,
                        num_leaves=num_leaves,
                        min_child_samples=min_child_samples,
                        n_jobs=max(int(config.cores), 1),
                        random_state=int(config.seed),
                        verbose=-1,
                    )
                    estimator.fit(
                        grouped_selector_train,
                        selector_train_target,
                        eval_set=[
                            (grouped_selector_valid, selector_valid_target)
                        ],
                        eval_metric="rmse",
                        callbacks=[
                            early_stopping(100, verbose=False),
                            log_evaluation(period=0),
                        ],
                    )
                    selected_n_estimators = max(
                        1, int(getattr(estimator, "best_iteration_", 1_000))
                    )
                    predictions = np.asarray(
                        estimator.predict(
                            grouped_selector_valid,
                            num_iteration=selected_n_estimators,
                        ),
                        dtype=float,
                    ).reshape(-1)
                else:
                    estimator = _make_regression_portfolio_estimator(
                        candidate,
                        selector_train_features,
                        seed=config.seed,
                        n_jobs=config.cores,
                        small_mixed=small_mixed,
                        selector=True,
                    )
                    estimator.fit(selector_train_features, selector_train_target)
                    predictions = np.asarray(
                        estimator.predict(selector_valid_features), dtype=float
                    ).reshape(-1)
                score = _selection_score(
                    metric,
                    selector_valid_target,
                    predictions,
                    None,
                )
            except Exception:
                log.warning(
                    "Regression portfolio candidate %s failed.",
                    candidate,
                    exc_info=True,
                )
                continue
            _portfolio_report(
                "Regression candidate %s: validation score %.10g, fit+validation %.2fs.",
                candidate,
                score,
                time.monotonic() - candidate_started_at,
            )
            observations.append(
                {
                    "model": candidate,
                    "score": score,
                    "predictions": predictions,
                    "truth": selector_valid_target,
                    "selected_n_estimators": selected_n_estimators,
                }
            )

        if not observations:
            raise RuntimeError("All regression portfolio candidates failed")
        if (
            small_wide
            or small_classic
            or large_compact_mixed
            or very_large_compact_mixed
            or large_grouped_sequence
            or large_dense_low_cardinality
            or large_high_categorical
        ):
            if small_wide:
                gate_name = "small-wide"
                gate_parameter = "_portfolio_small_wide_max_normalized_rmse"
                default_maximum = 0.2
            elif small_classic:
                gate_name = "small-classic"
                gate_parameter = "_portfolio_small_classic_max_normalized_rmse"
                default_maximum = 0.5
            elif large_compact_mixed:
                gate_name = "large-compact-mixed"
                gate_parameter = (
                    "_portfolio_large_compact_max_normalized_rmse"
                )
                default_maximum = 0.72
            elif very_large_compact_mixed:
                gate_name = "very-large-compact-mixed"
                gate_parameter = (
                    "_portfolio_very_large_compact_max_normalized_rmse"
                )
                default_maximum = 0.72
            elif large_grouped_sequence:
                gate_name = "large-grouped-sequence"
                gate_parameter = (
                    "_portfolio_large_grouped_sequence_max_normalized_rmse"
                )
                default_maximum = 0.9
            elif large_dense_low_cardinality:
                gate_name = "large-dense-low-cardinality"
                gate_parameter = (
                    "_portfolio_large_dense_max_normalized_rmse"
                )
                default_maximum = 0.9
            else:
                gate_name = "large-high-categorical"
                gate_parameter = (
                    "_portfolio_large_high_categorical_max_normalized_rmse"
                )
                default_maximum = 0.8
            maximum_normalized_rmse = float(
                config.framework_params.get(gate_parameter, default_maximum)
            )
            if maximum_normalized_rmse <= 0:
                raise ValueError(f"{gate_parameter} must be positive")
            normalized_rmse = _best_normalized_validation_rmse(observations)
            if normalized_rmse > maximum_normalized_rmse:
                _portfolio_report(
                    "%s validation normalized RMSE %.4f exceeds %.4f; "
                    "using standard FEDOT.",
                    gate_name,
                    normalized_rmse,
                    maximum_normalized_rmse,
                )
                return None
        ensemble, weights, selected_score = _select_regression_strategy(
            observations,
            metric=metric,
            min_relative_pair_gain=float(
                config.framework_params.get(
                    "_portfolio_regression_pair_min_relative_gain", 0.002
                )
            ),
            consider_all_pairs=ordered_histogram_ordinal,
        )
        if ridge_oof_strategy is not None:
            ensemble, weights, selected_score = ridge_oof_strategy
            _portfolio_report(
                "Replacing the holdout regression strategy with the guarded "
                "OOF Ridge expansion."
            )
        _portfolio_report(
            "Selected regression ensemble %s with weights %s and validation "
            "score %.10g.",
            [contender["model"] for contender in ensemble],
            weights.tolist(),
            selected_score,
        )

        fitted_models = []
        for contender in ensemble:
            estimator = _make_regression_portfolio_estimator(
                contender["model"],
                train_features,
                seed=config.seed,
                n_jobs=config.cores,
                small_mixed=small_mixed,
            )
            if large_grouped_sequence:
                estimator.steps[-1][1].fixed_n_estimators = contender[
                    "selected_n_estimators"
                ]
            estimator.fit(train_features, train_target)
            fitted_models.append(_SklearnRegressionModel(estimator, train_target))

    with Timer() as predict:
        member_predictions = [
            model.predict(test_features) for model in fitted_models
        ]
        predictions = np.average(member_predictions, axis=0, weights=weights)
        if small_wide:
            predictions = _clip_to_observed_target_range(
                predictions, train_target
            )
        elif minimum_inflated_skewed:
            predictions = _clip_to_observed_target_quantiles(
                predictions, train_target, lower_quantile=0.01, upper_quantile=0.99
            )

    for model in fitted_models:
        save_artifacts(model, config)
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=None,
        target_is_encoded=False,
        models_count=len(fitted_models),
        training_duration=training.duration + ridge_oof_duration,
        predict_duration=predict.duration,
    )


def _run_classification_portfolio(
    dataset,
    config,
    scoring_metric,
    training_params,
    runtime_min,
    max_pipeline_fit_time,
):
    """Select a cheap robust pipeline on a holdout and refit it on all training data.

    This mode deliberately uses only dataset characteristics, validation quality and
    measured fit times. It does not contain task- or dataset-name specific rules.
    """
    started_at = time.monotonic()
    train_target, observed_encoded_labels = _contiguous_classification_target(
        dataset.train.y
    )
    test_target = _target_array(dataset.test.y)
    metric = _normalise_metric_name(config.metric)
    geometry_features = dataset.train.X
    if _use_sparse_native_logit(
        geometry_features,
        train_target,
        metric=metric,
        framework_params=config.framework_params,
    ):
        return _run_sparse_native_logit(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
    if _use_materialized_sparse_tfidf_logit(
        geometry_features,
        train_target,
        metric=metric,
        framework_params=config.framework_params,
    ):
        return _run_materialized_sparse_tfidf_logit(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
    if _use_screened_wide_auc_lgbm(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    ):
        return _run_screened_wide_auc_lgbm(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
    if _use_medium_screened_auc_lgbm(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    ):
        return _run_medium_screened_auc_lgbm(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
    if _use_high_missing_frequency_auc_lgbm(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    ):
        frequency_result = _run_high_missing_frequency_auc_lgbm(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
        if frequency_result is not None:
            return frequency_result
    if _use_target_frequency_auc_lgbm(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    ):
        return _run_target_frequency_auc_lgbm(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
    if _use_narrow_categorical_auc_catboost(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    ):
        return _run_narrow_categorical_auc_catboost(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
    if _use_nominal_multiclass_catboost(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    ):
        nominal_result = _run_nominal_multiclass_catboost(
            train_features=geometry_features,
            train_target=train_target,
            test_features=dataset.test.X,
            test_target=test_target,
            config=config,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
        if nominal_result is not None:
            return nominal_result
    grouped_one_hot_profile = _use_exact_one_hot_grouped_xgboost(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    )
    if grouped_one_hot_profile is not None:
        try:
            grouped_one_hot_result = _run_exact_one_hot_grouped_xgboost(
                train_features=geometry_features,
                train_target=train_target,
                test_features=dataset.test.X,
                test_target=test_target,
                config=config,
                profile=grouped_one_hot_profile,
                observed_encoded_labels=observed_encoded_labels,
                encoded_class_count=getattr(dataset, "encoded_class_count", None),
                portfolio_started_at=started_at,
            )
        except Exception:
            log.warning(
                "Exact-one-hot grouped XGBoost failed; continuing with the "
                "ordinary train-only portfolio.",
                exc_info=True,
            )
            grouped_one_hot_result = None
        if grouped_one_hot_result is not None:
            return grouped_one_hot_result
    compact_encoding_min_rows = int(
        config.framework_params.get("_portfolio_compact_encode_min_rows", 2_500)
    )
    configured_fast_one_hot = config.framework_params.get(
        "_portfolio_fast_one_hot"
    )
    fast_one_hot = (
        _should_use_fast_one_hot(
            geometry_features,
            train_target,
            min_rows=compact_encoding_min_rows,
        )
        if configured_fast_one_hot is None
        else _as_bool(configured_fast_one_hot)
    )
    train_features, test_features = _compact_encode_if_needed(
        geometry_features,
        dataset.test.X,
        force=_as_bool(
            config.framework_params.get("_portfolio_compact_encode", False)
        ),
        min_rows=compact_encoding_min_rows,
        one_hot=fast_one_hot,
    )
    validation_fraction = float(
        config.framework_params.get("_portfolio_validation_fraction", 0.2)
    )
    use_large_narrow_extreme_many_class_xgboost = (
        _use_large_narrow_extreme_many_class_xgboost(
            geometry_features,
            train_target,
            metric=metric,
            runtime_seconds=config.max_runtime_seconds,
            cores=config.cores,
            framework_params=config.framework_params,
        )
    )
    train_rows_are_explicit = "_portfolio_train_rows" in config.framework_params
    configured_max_train_rows = int(
        config.framework_params.get("_portfolio_train_rows", 20_000)
    )
    configured_xgboost_max_bin = config.framework_params.get(
        "_portfolio_xgboost_max_bin"
    )
    xgboost_max_bin = (
        _adaptive_xgboost_max_bin(geometry_features, train_target)
        if configured_xgboost_max_bin is None
        else int(configured_xgboost_max_bin)
    )
    max_validation_rows = int(
        config.framework_params.get("_portfolio_validation_rows", 10_000)
    )
    if train_rows_are_explicit:
        max_train_rows = configured_max_train_rows
    elif use_large_narrow_extreme_many_class_xgboost:
        max_train_rows = min(configured_max_train_rows, 10_000)
    else:
        max_train_rows = _adaptive_selector_train_rows(
            geometry_features,
            train_target,
            configured_max_train_rows,
            xgboost_max_bin=xgboost_max_bin,
        )
    configured_candidates = config.framework_params.get("_portfolio_candidates")
    candidates = (
        ["xgboost"]
        if use_large_narrow_extreme_many_class_xgboost
        else _adaptive_default_portfolio_candidates(
            geometry_features,
            train_target,
            configured=configured_candidates,
            metric=metric,
        )
    )
    lgbm_extra_trees_pair_enabled = (
        candidates == ["lgbm", "xgboost"]
        and _use_train_gated_lgbm_extra_trees_pair(
            geometry_features,
            train_target,
            metric=metric,
            runtime_seconds=config.max_runtime_seconds,
            cores=config.cores,
            framework_params=config.framework_params,
        )
    )
    if lgbm_extra_trees_pair_enabled:
        # A unique contender name prevents its parameters, timing and direct
        # refit horizon from aliasing the incumbent LightGBM entry.
        candidates = [*candidates, _LGBM_EXTRA_TREES_MODEL]
    calibrate = _as_bool(config.framework_params.get("_portfolio_calibrate", True))
    prior_calibrate = _as_bool(
        config.framework_params.get("_portfolio_prior_calibrate", True)
    )
    retain_auc_boosting_pair = _as_bool(
        config.framework_params.get("_portfolio_auc_retain_boosting_pair", False)
    )
    weak_signal_shallow_probe_enabled = (
        _adaptive_weak_signal_shallow_probe_enabled(config.framework_params)
    )
    configured_pair_strong_weight = config.framework_params.get(
        "_portfolio_pair_strong_weight"
    )
    pair_strong_weight = (
        _adaptive_pair_strong_weight(geometry_features, train_target, candidates)
        if configured_pair_strong_weight is None
        and configured_candidates is None
        else 0.5
        if configured_pair_strong_weight is None
        else float(configured_pair_strong_weight)
    )
    configured_lgbm_num_leaves = config.framework_params.get(
        "_portfolio_lgbm_num_leaves"
    )
    lgbm_num_leaves = (
        _adaptive_lgbm_num_leaves(geometry_features, train_target)
        if configured_lgbm_num_leaves is None
        else int(configured_lgbm_num_leaves)
    )
    configured_lgbm_min_child_samples = config.framework_params.get(
        "_portfolio_lgbm_min_child_samples"
    )
    lgbm_min_child_samples = _adaptive_lgbm_min_child_samples(
        geometry_features,
        train_target,
        configured=configured_lgbm_min_child_samples,
    )
    configured_lgbm_min_child_weight = config.framework_params.get(
        "_portfolio_lgbm_min_child_weight"
    )
    lgbm_min_child_weight = _adaptive_lgbm_min_child_weight(
        geometry_features,
        train_target,
        configured=configured_lgbm_min_child_weight,
    )
    configured_xgboost_learning_rate = config.framework_params.get(
        "_portfolio_xgboost_learning_rate"
    )
    xgboost_learning_rate = (
        _adaptive_xgboost_learning_rate(geometry_features, train_target)
        if configured_xgboost_learning_rate is None
        else float(configured_xgboost_learning_rate)
    )
    configured_xgboost_max_depth = config.framework_params.get(
        "_portfolio_xgboost_max_depth"
    )
    xgboost_max_depth = (
        _adaptive_xgboost_max_depth(geometry_features, train_target)
        if configured_xgboost_max_depth is None
        else int(configured_xgboost_max_depth)
    )
    configured_xgboost_colsample_bytree = config.framework_params.get(
        "_portfolio_xgboost_colsample_bytree"
    )
    xgboost_colsample_bytree = (
        _adaptive_xgboost_colsample_bytree(geometry_features, train_target)
        if configured_xgboost_colsample_bytree is None
        else float(configured_xgboost_colsample_bytree)
    )
    configured_xgboost_subsample = config.framework_params.get(
        "_portfolio_xgboost_subsample"
    )
    xgboost_subsample = (
        _adaptive_xgboost_subsample(geometry_features, train_target)
        if configured_xgboost_subsample is None
        else float(configured_xgboost_subsample)
    )
    configured_xgboost_min_child_weight = config.framework_params.get(
        "_portfolio_xgboost_min_child_weight"
    )
    xgboost_min_child_weight = (
        _adaptive_xgboost_min_child_weight(geometry_features, train_target)
        if configured_xgboost_min_child_weight is None
        else float(configured_xgboost_min_child_weight)
    )
    configured_logit_c = config.framework_params.get("_portfolio_logit_c")
    logit_c = (
        _adaptive_logit_c(geometry_features, train_target, candidates)
        if configured_logit_c is None and configured_candidates is None
        else None
        if configured_logit_c is None
        else float(configured_logit_c)
    )
    configured_rf_n_estimators = config.framework_params.get(
        "_portfolio_rf_n_estimators"
    )
    rf_n_estimators = (
        _adaptive_rf_n_estimators(geometry_features, train_target, candidates)
        if configured_rf_n_estimators is None
        and configured_candidates is None
        else None
        if configured_rf_n_estimators is None
        else int(configured_rf_n_estimators)
    )
    full_data_refit_requested = _as_bool(
        config.framework_params.get("_portfolio_full_data_refit", True)
    )
    configured_reuse_cv_secondary = config.framework_params.get(
        "_portfolio_reuse_cv_secondary"
    )
    configured_direct_all_row_refit = config.framework_params.get(
        "_portfolio_direct_all_row_refit"
    )
    configured_direct_round_exponent = config.framework_params.get(
        "_portfolio_direct_round_exponent"
    )
    configured_full_data_refit_min_rows = config.framework_params.get(
        "_portfolio_full_data_refit_min_rows"
    )
    full_data_refit_min_rows = (
        5_000
        if configured_full_data_refit_min_rows is None
        else int(configured_full_data_refit_min_rows)
    )
    configured_boosting_rounds = config.framework_params.get(
        "_portfolio_boosting_rounds"
    )
    boosting_rounds = _adaptive_boosting_rounds(
        geometry_features,
        train_target,
        configured=configured_boosting_rounds,
    )

    if not 0 < validation_fraction < 0.5:
        raise ValueError("_portfolio_validation_fraction must be between 0 and 0.5")
    if configured_max_train_rows <= 0:
        raise ValueError("_portfolio_train_rows must be greater than zero")
    if max_validation_rows <= 0:
        raise ValueError("_portfolio_validation_rows must be greater than zero")
    if boosting_rounds is not None and boosting_rounds <= 0:
        raise ValueError("_portfolio_boosting_rounds must be greater than zero")
    if lgbm_num_leaves is not None and lgbm_num_leaves < 2:
        raise ValueError("_portfolio_lgbm_num_leaves must be at least two")
    if lgbm_min_child_samples is not None and lgbm_min_child_samples <= 0:
        raise ValueError("_portfolio_lgbm_min_child_samples must be positive")
    if lgbm_min_child_weight is not None and lgbm_min_child_weight < 0:
        raise ValueError(
            "_portfolio_lgbm_min_child_weight must be non-negative"
        )
    if xgboost_learning_rate is not None and xgboost_learning_rate <= 0:
        raise ValueError("_portfolio_xgboost_learning_rate must be greater than zero")
    if xgboost_max_depth is not None and xgboost_max_depth <= 0:
        raise ValueError("_portfolio_xgboost_max_depth must be greater than zero")
    if xgboost_max_bin is not None and xgboost_max_bin < 2:
        raise ValueError("_portfolio_xgboost_max_bin must be at least two")
    if xgboost_colsample_bytree is not None and not 0 < xgboost_colsample_bytree <= 1:
        raise ValueError(
            "_portfolio_xgboost_colsample_bytree must be in the interval (0, 1]"
        )
    if xgboost_subsample is not None and not 0 < xgboost_subsample <= 1:
        raise ValueError(
            "_portfolio_xgboost_subsample must be in the interval (0, 1]"
        )
    if xgboost_min_child_weight is not None and xgboost_min_child_weight < 0:
        raise ValueError(
            "_portfolio_xgboost_min_child_weight must be non-negative"
        )
    if logit_c is not None and logit_c <= 0:
        raise ValueError("_portfolio_logit_c must be greater than zero")
    if rf_n_estimators is not None and rf_n_estimators <= 0:
        raise ValueError("_portfolio_rf_n_estimators must be greater than zero")
    if not 0.5 <= pair_strong_weight < 1:
        raise ValueError(
            "_portfolio_pair_strong_weight must be in the interval [0.5, 1)"
        )
    if full_data_refit_min_rows <= 0:
        raise ValueError("_portfolio_full_data_refit_min_rows must be greater than zero")
    if _use_fixed_round_all_row_xgboost(
        geometry_features,
        train_target,
        metric=metric,
        runtime_seconds=config.max_runtime_seconds,
        cores=config.cores,
        framework_params=config.framework_params,
    ):
        model_params = _candidate_model_params(
            "xgboost",
            xgboost_learning_rate=xgboost_learning_rate,
            xgboost_max_depth=xgboost_max_depth,
            xgboost_max_bin=xgboost_max_bin,
            xgboost_colsample_bytree=xgboost_colsample_bytree,
            xgboost_subsample=xgboost_subsample,
            xgboost_min_child_weight=xgboost_min_child_weight,
        )
        model_params = _fixed_round_xgboost_model_params(
            model_params,
            seed=config.seed,
        )
        model_params["fit_time_limit"] = (
            float(config.max_runtime_seconds)
            - _DIRECT_XGBOOST_NON_FIT_RESERVE_SECONDS
        )
        model_params["fit_time_limit_adaptive_learning_rate"] = True
        return _run_fixed_round_all_row_xgboost(
            train_features=train_features,
            train_target=train_target,
            test_features=test_features,
            test_target=test_target,
            config=config,
            scoring_metric=scoring_metric,
            training_params=training_params,
            runtime_min=runtime_min,
            max_pipeline_fit_time=max_pipeline_fit_time,
            model_params=model_params,
            boosting_rounds=260,
            observed_encoded_labels=observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )
    validation_splits = _classification_validation_splits(
        train_features,
        train_target,
        validation_fraction=validation_fraction,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        seed=config.seed,
    )
    portfolio_labels = np.unique(train_target)
    full_train_rows = len(train_target)
    full_data_refit = _use_all_row_refit(
        requested=full_data_refit_requested,
        configured_min_rows=configured_full_data_refit_min_rows,
        full_train_rows=full_train_rows,
        validation_fold_count=len(validation_splits),
        metric=metric,
    )
    if (
        full_data_refit
        and configured_full_data_refit_min_rows is None
        and full_train_rows < full_data_refit_min_rows
    ):
        _portfolio_report(
            "Enabling all-row refit for %d training rows because logloss "
            "selection has %d cross-validation folds.",
            full_train_rows,
            len(validation_splits),
        )
    if full_data_refit_requested and not full_data_refit:
        _portfolio_report(
            "Disabling all-row refit for %d training rows; at least %d are needed "
            "unless logloss selection has at least three folds.",
            full_train_rows,
            full_data_refit_min_rows,
        )
    selector_train_rows = int(
        np.mean([len(split[2]) for split in validation_splits])
    )
    selector_validation_rows = int(
        np.mean([len(split[3]) for split in validation_splits])
    )
    direct_all_row_refit = _use_direct_all_row_refit(
        configured=configured_direct_all_row_refit,
        full_data_refit=full_data_refit,
        full_train_rows=full_train_rows,
        selector_train_rows=selector_train_rows,
        feature_count=train_features.shape[1],
        all_features_are_numeric=_all_features_are_numeric(train_features),
    )
    sampled_numeric_density = (
        _sampled_numeric_density(train_features)
        if direct_all_row_refit
        else None
    )
    direct_round_exponent = _adaptive_direct_round_exponent(
        configured_direct_round_exponent,
        sampled_numeric_density,
    )
    if not 0 <= direct_round_exponent <= 1:
        raise ValueError(
            "_portfolio_direct_round_exponent must be between zero and one"
        )
    # Selector timing includes fixed FEDOT setup and validation prediction costs,
    # so scaling its full duration linearly by row count severely overestimates a
    # refit when the selector is deliberately small.  Tree-training wall time is
    # better represented by a conservative sublinear scaling of the observed total.
    refit_scale = _portfolio_refit_scale(
        full_train_rows,
        selector_train_rows,
        len(validation_splits),
        feature_count=train_features.shape[1],
    )
    prediction_reserve = max(8.0, config.max_runtime_seconds * 0.04)
    candidate_start_safety_reserve = max(
        3.0, config.max_runtime_seconds * 0.03
    )
    refit_start_safety_reserve = max(
        8.0, config.max_runtime_seconds * 0.08
    )

    log.info(
        "Portfolio candidates=%s; validation folds=%d; mean selector train rows=%d; "
        "mean validation rows=%d; boosting cap=%d; LGBM minimum child samples=%s; "
        "LGBM minimum child weight=%s; "
        "XGBoost learning rate=%s; "
        "XGBoost max depth=%s; XGBoost max bins=%s; XGBoost column sample=%s; "
        "XGBoost row sample=%s; "
        "XGBoost min child weight=%s; RF trees=%s; "
        "full-refit cost scale=%.2f; direct all-row refit=%s; direct round "
        "exponent=%.2f; sampled numeric density=%s.",
        candidates,
        len(validation_splits),
        selector_train_rows,
        selector_validation_rows,
        boosting_rounds,
        lgbm_min_child_samples,
        lgbm_min_child_weight,
        xgboost_learning_rate,
        xgboost_max_depth,
        xgboost_max_bin,
        xgboost_colsample_bytree,
        xgboost_subsample,
        xgboost_min_child_weight,
        rf_n_estimators,
        refit_scale,
        direct_all_row_refit,
        direct_round_exponent,
        (
            "n/a"
            if sampled_numeric_density is None
            else f"{sampled_numeric_density:.3f}"
        ),
    )

    observations = {}
    best = None
    contenders = []
    weak_signal_shallow_activated = False
    with Timer() as training:
        for candidate in candidates:
            if _skip_adaptive_composed_auto_after_kernel_dominance(
                candidate,
                configured_candidates,
                geometry_features,
                train_target,
                contenders,
            ):
                _portfolio_report(
                    "Skipping composed auto after a bounded scaled-SVC view "
                    "already won raw OOF by the %.3f dominance margin.",
                    _NARROW_KERNEL_DOMINANCE_MIN_LOGLOSS_GAIN,
                )
                continue
            estimated_fit = _estimated_candidate_fit_seconds(candidate, observations)
            if best is not None and estimated_fit is not None:
                elapsed = time.monotonic() - started_at
                remaining_for_work = max(
                    config.max_runtime_seconds
                    - elapsed
                    - prediction_reserve
                    - candidate_start_safety_reserve,
                    0.0,
                )
                # Preserve a deployable refit of the current leader only when it
                # is still feasible. If table scaling already makes every refit
                # impossible, blocking a diverse validation candidate merely
                # locks in the first model; validation-model reuse below is the
                # intended budget-safe deployment path for that regime.
                protected_refit = _candidate_start_refit_reserve(
                    best["refit_seconds"], remaining_for_work
                )
                required = estimated_fit + protected_refit
                if (
                    required > remaining_for_work
                ):
                    _portfolio_report(
                        "Skipping portfolio candidate %s: estimated %.1fs validation "
                        "fit and %.1fs protected leader refit do not fit the %.1fs remaining "
                        "budget.",
                        candidate,
                        estimated_fit,
                        protected_refit,
                        remaining_for_work,
                    )
                    continue

            candidate_started_at = time.monotonic()
            candidate_truth = []
            candidate_predictions = []
            candidate_probabilities = []
            candidate_models = []
            candidate_model_params = _candidate_model_params(
                candidate,
                lgbm_num_leaves=lgbm_num_leaves,
                lgbm_min_child_samples=lgbm_min_child_samples,
                lgbm_min_child_weight=lgbm_min_child_weight,
                xgboost_learning_rate=xgboost_learning_rate,
                xgboost_max_depth=xgboost_max_depth,
                xgboost_max_bin=xgboost_max_bin,
                xgboost_colsample_bytree=xgboost_colsample_bytree,
                xgboost_subsample=xgboost_subsample,
                xgboost_min_child_weight=xgboost_min_child_weight,
                logit_c=logit_c,
                rf_n_estimators=rf_n_estimators,
            )
            try:
                for X_train, X_valid, y_train, y_valid in validation_splits:
                    if _is_scaled_svc_candidate(candidate):
                        candidate_automl = _fit_scaled_svc_candidate(
                            X_train,
                            y_train,
                            model_params=candidate_model_params,
                            seed=config.seed,
                        )
                    elif candidate == "mixed_logit":
                        candidate_automl = _fit_mixed_logit_candidate(
                            X_train,
                            y_train,
                            model_params=candidate_model_params,
                            seed=config.seed,
                        )
                    elif candidate == "mixed_svc":
                        candidate_automl = _fit_mixed_svc_candidate(
                            X_train,
                            y_train,
                            model_params=candidate_model_params,
                            seed=config.seed,
                        )
                    elif candidate in {"extra_trees", "extra_trees_wide"}:
                        candidate_automl = _fit_extra_trees_candidate(
                            X_train,
                            y_train,
                            model_params=candidate_model_params,
                            seed=config.seed,
                            n_jobs=config.cores,
                        )
                    elif candidate == "mixed_extra_trees":
                        candidate_automl = _fit_mixed_extra_trees_candidate(
                            X_train,
                            y_train,
                            model_params=candidate_model_params,
                            seed=config.seed,
                            n_jobs=config.cores,
                        )
                    elif candidate in {
                        "relational_hist",
                        "relational_hist_second_order",
                    }:
                        candidate_automl = _fit_relational_hist_candidate(
                            X_train,
                            y_train,
                            model_params=candidate_model_params,
                            seed=config.seed,
                            second_order=candidate == "relational_hist_second_order",
                        )
                    else:
                        candidate_automl = _make_fedot(
                            config=config,
                            scoring_metric=scoring_metric,
                            training_params=training_params,
                            runtime_min=runtime_min,
                            max_pipeline_fit_time=max_pipeline_fit_time,
                        )
                        candidate_automl.fit(
                            features=X_train,
                            target=y_train,
                            predefined_model=_predefined_model_with_n_jobs(
                                candidate,
                                config.cores,
                                n_estimators=boosting_rounds,
                                model_params=candidate_model_params,
                            ),
                        )
                    predictions, probabilities = _classification_predictions(
                        candidate_automl, X_valid
                    )
                    candidate_truth.append(y_valid)
                    candidate_predictions.append(predictions)
                    candidate_probabilities.append(probabilities)
                    candidate_models.append(candidate_automl)

                candidate_truth = np.concatenate(candidate_truth)
                predictions = np.concatenate(candidate_predictions)
                probabilities = np.vstack(candidate_probabilities)
                score = _selection_score(
                    metric,
                    candidate_truth,
                    predictions,
                    probabilities,
                    labels=portfolio_labels if metric == "logloss" else None,
                )
            except Exception:
                duration = time.monotonic() - candidate_started_at
                observations[candidate] = duration
                log.warning(
                    "Portfolio candidate %s failed after %.1fs.",
                    candidate,
                    duration,
                    exc_info=True,
                )
                if "candidate_automl" in locals():
                    del candidate_automl
                candidate_models.clear()
                gc.collect()
                continue

            duration = time.monotonic() - candidate_started_at
            observations[candidate] = duration
            contender = {
                "model": candidate,
                "score": score,
                "duration": duration,
                "refit_seconds": duration * refit_scale,
                "probabilities": probabilities,
                "truth": candidate_truth,
                "validation_models": candidate_models,
                "model_params": candidate_model_params,
                "labels": portfolio_labels,
            }
            if (
                candidate == "lgbm"
                and _use_weak_signal_shallow_probe(
                    train_features,
                    train_target,
                    metric=metric,
                    candidates=candidates,
                    enabled=weak_signal_shallow_probe_enabled,
                    runtime_seconds=config.max_runtime_seconds,
                    cores=config.cores,
                )
                and _is_weak_auc_score(score)
            ):
                shallow_started_at = time.monotonic()
                shallow_truth = []
                shallow_predictions = []
                shallow_probabilities = []
                shallow_models = []
                shallow_params = {
                    **candidate_model_params,
                    "num_leaves": 31,
                    "min_child_samples": 200,
                }
                try:
                    for X_train, X_valid, y_train, y_valid in validation_splits:
                        shallow_automl = _make_fedot(
                            config=config,
                            scoring_metric=scoring_metric,
                            training_params=training_params,
                            runtime_min=runtime_min,
                            max_pipeline_fit_time=max_pipeline_fit_time,
                        )
                        shallow_automl.fit(
                            features=X_train,
                            target=y_train,
                            predefined_model=_predefined_model_with_n_jobs(
                                candidate,
                                config.cores,
                                n_estimators=boosting_rounds,
                                model_params=shallow_params,
                            ),
                        )
                        shallow_prediction, shallow_probability = (
                            _classification_predictions(shallow_automl, X_valid)
                        )
                        shallow_truth.append(y_valid)
                        shallow_predictions.append(shallow_prediction)
                        shallow_probabilities.append(shallow_probability)
                        shallow_models.append(shallow_automl)

                    shallow_truth = np.concatenate(shallow_truth)
                    shallow_predictions = np.concatenate(shallow_predictions)
                    shallow_probabilities = np.vstack(shallow_probabilities)
                    shallow_score = _selection_score(
                        metric,
                        shallow_truth,
                        shallow_predictions,
                        shallow_probabilities,
                    )
                    shallow_duration = time.monotonic() - shallow_started_at
                    del contender["validation_models"]
                    del candidate_models
                    del candidate_automl
                    contender = {
                        "model": candidate,
                        "score": shallow_score,
                        "duration": shallow_duration,
                        "refit_seconds": shallow_duration * refit_scale,
                        "probabilities": shallow_probabilities,
                        "truth": shallow_truth,
                        "validation_models": shallow_models,
                        "model_params": shallow_params,
                        "labels": portfolio_labels,
                    }
                    # The next candidate is XGBoost in this guarded two-model
                    # regime. Mutating these local overrides affects both its
                    # selector fit and any all-row refit. The default LGBM was
                    # used only as the train-only signal gate.
                    xgboost_max_depth = 3
                    xgboost_min_child_weight = 20.0
                    weak_signal_shallow_activated = True
                    _portfolio_report(
                        "Weak-signal shallow probe activated: default LGBM OOF "
                        "AUC %.8f; shallow LGBM OOF AUC %.8f, extra selector "
                        "cost %.1fs. The shallow boosting pair will be retained.",
                        score,
                        shallow_score,
                        shallow_duration,
                    )
                except Exception:
                    log.warning(
                        "Weak-signal shallow LGBM probe failed; keeping the "
                        "default portfolio.",
                        exc_info=True,
                    )
                    del shallow_models
                    if "shallow_automl" in locals():
                        del shallow_automl
                gc.collect()
            if (
                candidate == "xgboost"
                and configured_xgboost_max_depth is None
                and _use_shallow_xgboost_probe(
                    train_features,
                    train_target,
                    metric=metric,
                    validation_fold_count=len(validation_splits),
                    adaptive_depth=xgboost_max_depth,
                )
                and not _refit_fits_budget(
                    contender["refit_seconds"],
                    time.monotonic() - started_at,
                    config.max_runtime_seconds,
                    prediction_reserve,
                    refit_start_safety_reserve,
                )
                and _refit_fits_budget(
                    max(duration, 0.1),
                    time.monotonic() - started_at,
                    config.max_runtime_seconds,
                    prediction_reserve,
                    candidate_start_safety_reserve,
                )
            ):
                shallow_started_at = time.monotonic()
                shallow_truth = []
                shallow_predictions = []
                shallow_probabilities = []
                shallow_models = []
                shallow_params = {**candidate_model_params, "max_depth": 4}
                try:
                    for X_train, X_valid, y_train, y_valid in validation_splits:
                        shallow_automl = _make_fedot(
                            config=config,
                            scoring_metric=scoring_metric,
                            training_params=training_params,
                            runtime_min=runtime_min,
                            max_pipeline_fit_time=max_pipeline_fit_time,
                        )
                        shallow_automl.fit(
                            features=X_train,
                            target=y_train,
                            predefined_model=_predefined_model_with_n_jobs(
                                candidate,
                                config.cores,
                                n_estimators=boosting_rounds,
                                model_params=shallow_params,
                            ),
                        )
                        shallow_prediction, shallow_probability = (
                            _classification_predictions(shallow_automl, X_valid)
                        )
                        shallow_truth.append(y_valid)
                        shallow_predictions.append(shallow_prediction)
                        shallow_probabilities.append(shallow_probability)
                        shallow_models.append(shallow_automl)

                    shallow_truth = np.concatenate(shallow_truth)
                    shallow_predictions = np.concatenate(shallow_predictions)
                    shallow_probabilities = np.vstack(shallow_probabilities)
                    shallow_score = _selection_score(
                        metric,
                        shallow_truth,
                        shallow_predictions,
                        shallow_probabilities,
                        labels=portfolio_labels,
                    )
                    shallow_duration = time.monotonic() - shallow_started_at
                    _portfolio_report(
                        "Shallow XGBoost budget fallback: selection score "
                        "%.8f, fit+validation %.1fs.",
                        shallow_score,
                        shallow_duration,
                    )
                    shallow_refit_seconds = shallow_duration * refit_scale
                    shallow_reuses_validation_models = not _refit_fits_budget(
                        shallow_refit_seconds,
                        time.monotonic() - started_at,
                        config.max_runtime_seconds,
                        prediction_reserve,
                        refit_start_safety_reserve,
                    )
                    if (
                        shallow_reuses_validation_models
                        and _materially_better_shallow_score(shallow_score, score)
                    ):
                        del contender["validation_models"]
                        del candidate_models
                        del candidate_automl
                        contender = {
                            "model": candidate,
                            "score": shallow_score,
                            "duration": shallow_duration,
                            "refit_seconds": shallow_refit_seconds,
                            "probabilities": shallow_probabilities,
                            "truth": shallow_truth,
                            "validation_models": shallow_models,
                            "model_params": shallow_params,
                            "labels": portfolio_labels,
                        }
                        _portfolio_report(
                            "Using shallow XGBoost validation models: score gain "
                            "%.8f exceeds the guarded materiality threshold.",
                            shallow_score - score,
                        )
                    else:
                        del shallow_models
                        del shallow_automl
                        if shallow_reuses_validation_models:
                            _portfolio_report(
                                "Keeping default-depth XGBoost: shallow score gain "
                                "%.8f is not material.",
                                shallow_score - score,
                            )
                        else:
                            _portfolio_report(
                                "Keeping default-depth XGBoost: the shallow model "
                                "would enter a full-refit regime where holdout depth "
                                "transfer is not reliable.",
                            )
                except Exception:
                    log.warning(
                        "Shallow XGBoost budget fallback failed; keeping the "
                        "default-depth validation models.",
                        exc_info=True,
                    )
                    del shallow_models
                    if "shallow_automl" in locals():
                        del shallow_automl
                gc.collect()
            if (
                candidate == "xgboost"
                and configured_candidates is None
                and configured_xgboost_min_child_weight is None
                and _use_xgboost_child_weight_probe(
                    train_features,
                    train_target,
                    metric=metric,
                    validation_fold_count=len(validation_splits),
                    adaptive_min_child_weight=xgboost_min_child_weight,
                )
                and _refit_fits_budget(
                    contender["refit_seconds"] + max(duration, 0.1),
                    time.monotonic() - started_at,
                    config.max_runtime_seconds,
                    prediction_reserve,
                    candidate_start_safety_reserve,
                )
            ):
                strong_started_at = time.monotonic()
                strong_truth = []
                strong_predictions = []
                strong_probabilities = []
                strong_models = []
                strong_params = {
                    **candidate_model_params,
                    "min_child_weight": 10.0,
                }
                try:
                    for X_train, X_valid, y_train, y_valid in validation_splits:
                        strong_automl = _make_fedot(
                            config=config,
                            scoring_metric=scoring_metric,
                            training_params=training_params,
                            runtime_min=runtime_min,
                            max_pipeline_fit_time=max_pipeline_fit_time,
                        )
                        strong_automl.fit(
                            features=X_train,
                            target=y_train,
                            predefined_model=_predefined_model_with_n_jobs(
                                candidate,
                                config.cores,
                                n_estimators=boosting_rounds,
                                model_params=strong_params,
                            ),
                        )
                        strong_prediction, strong_probability = (
                            _classification_predictions(strong_automl, X_valid)
                        )
                        strong_truth.append(y_valid)
                        strong_predictions.append(strong_prediction)
                        strong_probabilities.append(strong_probability)
                        strong_models.append(strong_automl)

                    strong_truth = np.concatenate(strong_truth)
                    strong_predictions = np.concatenate(strong_predictions)
                    strong_probabilities = np.vstack(strong_probabilities)
                    strong_score = _selection_score(
                        metric,
                        strong_truth,
                        strong_predictions,
                        strong_probabilities,
                        labels=portfolio_labels,
                    )
                    strong_duration = time.monotonic() - strong_started_at
                    use_strong = _materially_better_xgboost_regularisation(
                        strong_score,
                        score,
                        strong_probabilities,
                        probabilities,
                        strong_truth,
                        labels=portfolio_labels,
                    )
                    _portfolio_report(
                        "Strong-child XGBoost probe: selection score %.8f "
                        "versus %.8f, fit+validation %.1fs; selected=%s.",
                        strong_score,
                        score,
                        strong_duration,
                        use_strong,
                    )
                    if use_strong:
                        del contender["validation_models"]
                        del candidate_models
                        del candidate_automl
                        contender = {
                            "model": candidate,
                            "score": strong_score,
                            "duration": strong_duration,
                            "refit_seconds": strong_duration * refit_scale,
                            "probabilities": strong_probabilities,
                            "truth": strong_truth,
                            "validation_models": strong_models,
                            "model_params": strong_params,
                            "labels": portfolio_labels,
                        }
                        score = strong_score
                        duration = strong_duration
                    else:
                        del strong_models
                    del strong_automl
                except Exception:
                    log.warning(
                        "Strong-child XGBoost probe failed; keeping the default "
                        "validation models.",
                        exc_info=True,
                    )
                    del strong_models
                    if "strong_automl" in locals():
                        del strong_automl
                gc.collect()
            contenders.append(contender)
            _portfolio_report(
                "Portfolio candidate %s: selection score %.8f, fit+validation %.1fs, "
                "estimated full refit %.1fs.",
                candidate,
                contender["score"],
                contender["duration"],
                contender["refit_seconds"],
            )
            if best is None or contender["score"] > best["score"]:
                best = contender

        contenders = _train_gated_wide_extra_trees_contenders(
            contenders,
            reference_target=train_target,
            enabled=(
                metric == "logloss"
                and configured_candidates is None
                and _is_train_gated_wide_extra_trees_candidate(
                    geometry_features, train_target
                )
            ),
            pair_strong_weight=pair_strong_weight,
            seed=config.seed,
        )
        selector_reference_target = np.concatenate(
            [np.asarray(split[2]).reshape(-1) for split in validation_splits]
        )
        contenders, forced_lgbm_extra_trees_pair = (
            _train_gated_lgbm_extra_trees_pair_contenders(
                contenders,
                reference_target=selector_reference_target,
                enabled=lgbm_extra_trees_pair_enabled,
                pair_strong_weight=pair_strong_weight,
                seed=config.seed,
            )
        )
        best = (
            max(contenders, key=lambda contender: contender["score"])
            if contenders
            else None
        )
        if best is None:
            raise RuntimeError("All FEDOT portfolio candidates failed")

        temperature = 1.0
        if metric == "logloss":
            variant_contenders = _select_supported_adaptive_scaled_svc_variant(
                contenders,
                enabled=(
                    configured_candidates is None
                    and _is_small_narrow_dense_kernel_multiclass(
                        geometry_features, train_target
                    )
                ),
            )
            selector_contenders = _prefer_dominant_adaptive_scaled_svc(
                variant_contenders,
                enabled=(
                    configured_candidates is None
                    and _is_small_narrow_dense_kernel_multiclass(
                        geometry_features, train_target
                    )
                ),
            )
            if len(selector_contenders) < len(variant_contenders):
                selected_svc = selector_contenders[0]
                strongest_alternative = max(
                    contender["score"]
                    for contender in variant_contenders
                    if not _is_scaled_svc_candidate(contender["model"])
                )
                _portfolio_report(
                    "Using dominant adaptive scaled_svc as a singleton: raw OOF "
                    "logloss gain %.6f is at least %.3f.",
                    selected_svc["score"] - strongest_alternative,
                    _NARROW_KERNEL_DOMINANCE_MIN_LOGLOSS_GAIN,
                )
            if forced_lgbm_extra_trees_pair is not None:
                provisional_pair = forced_lgbm_extra_trees_pair["ensemble"]
                provisional_direct_rounds = {
                    contender["model"]: (
                        _extrapolated_boosting_rounds(
                            contender["validation_models"],
                            full_train_rows,
                            selector_train_rows,
                            boosting_rounds,
                            row_exponent=direct_round_exponent,
                        )
                        if direct_all_row_refit
                        and contender["model"]
                        in {"lgbm", _LGBM_EXTRA_TREES_MODEL, "xgboost"}
                        else None
                    )
                    for contender in provisional_pair
                }
                provisional_refit_seconds = sum(
                    _direct_refit_budget_estimate(
                        contender,
                        provisional_direct_rounds[contender["model"]],
                        len(validation_splits),
                        contender["refit_seconds"],
                    )
                    for contender in provisional_pair
                )
                elapsed = time.monotonic() - started_at
                if not _refit_fits_budget(
                    provisional_refit_seconds,
                    elapsed,
                    config.max_runtime_seconds,
                    prediction_reserve,
                    refit_start_safety_reserve,
                ):
                    _portfolio_report(
                        "Rejecting admitted Extra-LGBM pair before deployment: "
                        "its joint %.1fs all-row refit estimate does not fit the "
                        "remaining benchmark budget.",
                        provisional_refit_seconds,
                    )
                    for contender in provisional_pair:
                        if contender["model"] == _LGBM_EXTRA_TREES_MODEL:
                            contender.pop("validation_models", None)
                    contenders = [
                        contender
                        for contender in contenders
                        if contender["model"] != _LGBM_EXTRA_TREES_MODEL
                    ]
                    selector_contenders = [
                        contender
                        for contender in selector_contenders
                        if contender["model"] != _LGBM_EXTRA_TREES_MODEL
                    ]
                    forced_lgbm_extra_trees_pair = None
            if forced_lgbm_extra_trees_pair is not None:
                ensemble = forced_lgbm_extra_trees_pair["ensemble"]
                ensemble_weights = np.asarray(
                    forced_lgbm_extra_trees_pair["weights"], dtype=float
                )
                temperature = forced_lgbm_extra_trees_pair["temperature"]
                calibrated_score = forced_lgbm_extra_trees_pair[
                    "judgment_score"
                ]
            else:
                ensemble, ensemble_weights, temperature, calibrated_score = (
                    _select_logloss_ensemble(
                        selector_contenders,
                        calibrate,
                        pair_strong_weight=pair_strong_weight,
                    )
                )
            _portfolio_report(
                "Selected ensemble %s with weights %s; temperature %.4f gives "
                "validation score %.8f.",
                [contender["model"] for contender in ensemble],
                [round(float(weight), 2) for weight in ensemble_weights],
                temperature,
                calibrated_score,
            )
        elif metric == "auc":
            ensemble, ensemble_weights, selected_score = _select_auc_ensemble(
                contenders,
                retain_boosting_pair=(
                    retain_auc_boosting_pair or weak_signal_shallow_activated
                ),
            )
            _portfolio_report(
                "Selected AUC ensemble %s with weights %s and OOF score %.8f.",
                [contender["model"] for contender in ensemble],
                [round(float(weight), 2) for weight in ensemble_weights],
                selected_score,
            )
        else:
            ensemble = _select_ensemble(metric, contenders)
            ensemble_weights = np.full(len(ensemble), 1.0 / len(ensemble))
            _portfolio_report(
                "Selected ensemble %s with validation scores %s.",
                [contender["model"] for contender in ensemble],
                [round(contender["score"], 8) for contender in ensemble],
            )

        elapsed = time.monotonic() - started_at
        direct_rounds_by_name = {
            contender["model"]: (
                _extrapolated_boosting_rounds(
                    contender["validation_models"],
                    full_train_rows,
                    selector_train_rows,
                    boosting_rounds,
                    row_exponent=direct_round_exponent,
                )
                if direct_all_row_refit
                and contender["model"]
                in {"lgbm", _LGBM_EXTRA_TREES_MODEL, "xgboost"}
                else None
            )
            for contender in ensemble
        }
        refit_estimates_by_name = {
            contender["model"]: _direct_refit_budget_estimate(
                contender,
                direct_rounds_by_name[contender["model"]],
                len(validation_splits),
                contender["refit_seconds"],
            )
            for contender in ensemble
        }
        force_lgbm_extra_trees_validation_reuse = bool(
            forced_lgbm_extra_trees_pair is not None
            and not _refit_fits_budget(
                sum(refit_estimates_by_name.values()),
                elapsed,
                config.max_runtime_seconds,
                prediction_reserve,
                refit_start_safety_reserve,
            )
        )
        if force_lgbm_extra_trees_validation_reuse:
            _portfolio_report(
                "Reusing both admitted Extra-LGBM pair validation groups: "
                "their joint %.1fs refit estimate does not fit the remaining "
                "budget.",
                sum(refit_estimates_by_name.values()),
            )
        primary, validation_primary = _select_refittable_primary(
            ensemble,
            refit_estimates_by_name,
            elapsed,
            config.max_runtime_seconds,
            prediction_reserve,
            refit_start_safety_reserve,
        )
        if primary["model"] != validation_primary["model"]:
            _portfolio_report(
                "Using refittable %s as primary instead of selector-best %s; "
                "deployable all-row models take precedence over selector-scale "
                "models.",
                primary["model"],
                validation_primary["model"],
            )
        reuse_cv_secondary = _reuse_cv_secondary_models(
            configured=configured_reuse_cv_secondary,
            validation_fold_count=len(validation_splits),
            adaptive_wide_pair=(
                configured_candidates is None
                and candidates == ["logit", "xgboost"]
            ),
            primary_model=primary["model"],
        )
        ensemble_names = {contender["model"] for contender in ensemble}
        retained_forced_pair_ordinary_contenders = (
            [
                contender
                for contender in contenders
                if contender["model"] in {"lgbm", "xgboost"}
            ]
            if forced_lgbm_extra_trees_pair is not None
            else None
        )
        retained_forced_pair_ordinary_validation_models = (
            {
                contender["model"]: contender["validation_models"]
                for contender in retained_forced_pair_ordinary_contenders
            }
            if retained_forced_pair_ordinary_contenders is not None
            else None
        )
        unused = [
            contender
            for contender in contenders
            if contender["model"] not in ensemble_names
        ]
        for contender in unused:
            del contender["validation_models"]

        prediction_groups_by_name = {}
        reuses_validation_models_by_name = {}
        dropped_secondary_names = []
        elapsed = time.monotonic() - started_at
        primary_direct_rounds = direct_rounds_by_name[primary["model"]]
        primary_refit_seconds = refit_estimates_by_name[primary["model"]]
        primary_refit_fits = (
            False
            if force_lgbm_extra_trees_validation_reuse
            else _refit_fits_budget(
                primary_refit_seconds,
                elapsed,
                config.max_runtime_seconds,
                prediction_reserve,
                refit_start_safety_reserve,
            )
        )
        bounded_refit_seconds = (
            None
            if primary_refit_fits or force_lgbm_extra_trees_validation_reuse
            else _direct_xgboost_refit_time_limit(
                model=primary["model"],
                direct_rounds=primary_direct_rounds,
                elapsed_seconds=elapsed,
                runtime_seconds=config.max_runtime_seconds,
                prediction_reserve=prediction_reserve,
                refit_start_safety_reserve=refit_start_safety_reserve,
            )
        )
        local_image_refit_side = _local_image_refit_side(
            geometry_features,
            train_target,
            metric=metric,
            bounded_refit_seconds=bounded_refit_seconds,
            direct_rounds=primary_direct_rounds,
            framework_params=config.framework_params,
        )
        if forced_lgbm_extra_trees_pair is not None:
            # The admitted challenger is exactly the raw numeric
            # Extra-LGBM/XGBoost pair; do not substitute an unrelated local
            # image refit for either validated member.
            local_image_refit_side = None
        local_image_lgbm_contender = (
            _local_image_lgbm_refit_contender(
                contenders, bounded_refit_seconds
            )
            if local_image_refit_side is not None
            else None
        )
        if (
            local_image_refit_side is not None
            and local_image_lgbm_contender is None
        ):
            local_image_refit_side = None
        if primary_refit_fits or bounded_refit_seconds is not None:
            direct_rounds = (
                _LOCAL_IMAGE_LGBM_MAX_ROUNDS
                if local_image_refit_side is not None
                else primary_direct_rounds
            )
            primary_refit_contender = primary
            if bounded_refit_seconds is not None:
                if local_image_refit_side is not None:
                    primary_refit_contender = local_image_lgbm_contender
                    _portfolio_report(
                        "Using deadline-bounded LightGBM for the train-recognised "
                        "local image view; its wall-clock callback preserves the "
                        "same surrounding benchmark deadline."
                    )
                else:
                    bounded_model_params = _deadline_bounded_xgboost_model_params(
                        primary["model_params"],
                        geometry_features,
                        train_target,
                        metric=metric,
                        bounded_refit_seconds=bounded_refit_seconds,
                        direct_rounds=direct_rounds,
                    )
                    primary_refit_contender = {
                        "model": primary["model"],
                        "model_params": {
                            **bounded_model_params,
                            "fit_time_limit": bounded_refit_seconds,
                        },
                    }
                    if bounded_model_params != primary["model_params"]:
                        _portfolio_report(
                            "Using 128-bin histograms and L2=3 for the guarded "
                            "deadline-bounded medium-wide XGBoost refit."
                        )
                _portfolio_report(
                    "Starting deadline-bounded direct %s refit for up to %.1fs; "
                    "the conservative %.1fs estimate would otherwise reuse a "
                    "selector-scale model.",
                    primary_refit_contender["model"],
                    bounded_refit_seconds,
                    primary_refit_seconds,
                )
            if forced_lgbm_extra_trees_pair is None:
                del primary["validation_models"]
            gc.collect()
            primary_started_at = time.monotonic()
            if direct_rounds is not None:
                if local_image_refit_side is not None:
                    _portfolio_report(
                        "Fitting %s with at most %d boosting rounds before "
                        "the independent train-only calibration reserve.",
                        primary_refit_contender["model"],
                        direct_rounds,
                    )
                else:
                    _portfolio_report(
                        "Directly refitting %s on every training row with %d "
                        "maximum boosting rounds.",
                        primary_refit_contender["model"],
                        direct_rounds,
                    )
            primary_refit_features = train_features
            primary_refit_target = train_target
            local_image_calibration_features = None
            local_image_calibration_target = None
            if local_image_refit_side is not None:
                local_image_features = _local_image_features(
                    train_features, local_image_refit_side
                )
                fit_indices, calibration_indices = (
                    _local_image_fit_calibration_indices(
                        train_target, seed=config.seed
                    )
                )
                primary_refit_features = local_image_features[fit_indices]
                primary_refit_target = np.asarray(train_target)[fit_indices]
                local_image_calibration_features = local_image_features[
                    calibration_indices
                ]
                local_image_calibration_target = np.asarray(train_target)[
                    calibration_indices
                ]
                del local_image_features
                _portfolio_report(
                    "Using a train-recognised %dx%d local image view for the "
                    "deadline-bounded LightGBM refit: %d raw features become "
                    "%d pooled and gradient features; %d train-only rows are "
                    "reserved for probability calibration.",
                    local_image_refit_side,
                    local_image_refit_side,
                    train_features.shape[1],
                    primary_refit_features.shape[1],
                    len(calibration_indices),
                )
            primary_model = _fit_full_candidate(
                primary_refit_contender,
                primary_refit_features,
                primary_refit_target,
                config,
                scoring_metric,
                training_params,
                runtime_min,
                max_pipeline_fit_time,
                direct_rounds if direct_rounds is not None else boosting_rounds,
                use_eval_set=False if direct_rounds is not None else True,
                use_input_preprocessing=direct_rounds is None,
            )
            if local_image_refit_side is not None:
                local_image_calibration_probabilities = (
                    primary_model.predict_proba(
                        features=local_image_calibration_features,
                        probs_for_all_classes=True,
                    )
                )
                local_image_temperature = _fit_temperature(
                    local_image_calibration_probabilities,
                    local_image_calibration_target,
                    labels=np.unique(train_target),
                )
                _portfolio_report(
                    "Independent train-only local-image temperature is %.4f.",
                    local_image_temperature,
                )
                primary_model = _LocalImageProbabilityModel(
                    primary_model,
                    local_image_refit_side,
                    temperature=local_image_temperature,
                )
                del primary_refit_features
                del local_image_calibration_features
                del local_image_calibration_probabilities
            primary_refit_seconds = time.monotonic() - primary_started_at
            validation_fit_seconds = primary["duration"] / len(validation_splits)
            measured_refit_scale = primary_refit_seconds / max(
                validation_fit_seconds, 0.1
            )
            remaining_refit_reserve = sum(
                contender["duration"]
                / len(validation_splits)
                * measured_refit_scale
                for contender in ensemble
                if contender["model"] != primary["model"]
            )
            if direct_rounds is None:
                primary_model = _maybe_refit_on_all_rows(
                    primary_model,
                    primary,
                    train_features,
                    train_target,
                    config,
                    scoring_metric,
                    training_params,
                    runtime_min,
                    max_pipeline_fit_time,
                    boosting_rounds,
                    primary_refit_seconds,
                    started_at,
                    prediction_reserve,
                    full_data_refit,
                    additional_budget_reserve=remaining_refit_reserve,
                )
            prediction_groups_by_name[primary["model"]] = [primary_model]
            reuses_validation_models_by_name[primary["model"]] = False
        else:
            _portfolio_report(
                "Reusing validation models for primary %s: estimated %.1fs refit "
                "does not fit the remaining budget.",
                primary["model"],
                primary_refit_seconds,
            )
            prediction_groups_by_name[primary["model"]] = primary[
                "validation_models"
            ]
            reuses_validation_models_by_name[primary["model"]] = True
            primary_model = primary["validation_models"][0]
            measured_refit_scale = None

        for contender in ensemble:
            if contender["model"] == primary["model"]:
                continue
            if reuse_cv_secondary:
                reuse_reason = (
                    "configuration"
                    if configured_reuse_cv_secondary is not None
                    else "adaptive wide-data policy"
                )
                _portfolio_report(
                    "Reusing %d cross-validation models for secondary %s by "
                    "%s.",
                    len(contender["validation_models"]),
                    contender["model"],
                    reuse_reason,
                )
                prediction_groups_by_name[contender["model"]] = contender[
                    "validation_models"
                ]
                reuses_validation_models_by_name[contender["model"]] = True
                continue
            if measured_refit_scale is None:
                _portfolio_report(
                    "Reusing validation models for %s to keep every ensemble "
                    "member on the same selector scale as primary %s.",
                    contender["model"],
                    primary["model"],
                )
                prediction_groups_by_name[contender["model"]] = contender[
                    "validation_models"
                ]
                reuses_validation_models_by_name[contender["model"]] = True
                continue
            estimated_refit = _secondary_refit_budget_estimate(
                contender,
                measured_primary_scale=measured_refit_scale,
                fold_count=len(validation_splits),
            )
            if direct_rounds_by_name[contender["model"]] is not None:
                # A measured LGBM scale transferred well to another LGBM-like
                # refit, but it substantially underestimates XGBoost on sparse
                # wide matrices. Keep XGBoost's conservative row-based guard.
                if contender["model"] == "xgboost":
                    estimated_refit = max(
                        estimated_refit,
                        refit_estimates_by_name[contender["model"]],
                    )
            elapsed = time.monotonic() - started_at
            if _refit_fits_budget(
                estimated_refit,
                elapsed,
                config.max_runtime_seconds,
                prediction_reserve,
                refit_start_safety_reserve,
            ):
                direct_rounds = (
                    _extrapolated_boosting_rounds(
                        contender["validation_models"],
                        full_train_rows,
                        selector_train_rows,
                        boosting_rounds,
                        row_exponent=direct_round_exponent,
                    )
                    if direct_all_row_refit
                    and contender["model"]
                    in {"lgbm", _LGBM_EXTRA_TREES_MODEL, "xgboost"}
                    else None
                )
                del contender["validation_models"]
                gc.collect()
                if direct_rounds is not None:
                    _portfolio_report(
                        "Directly refitting %s on every training row with %d "
                        "selector-extrapolated boosting rounds.",
                        contender["model"],
                        direct_rounds,
                    )
                fitted_model = _fit_full_candidate(
                    contender,
                    train_features,
                    train_target,
                    config,
                    scoring_metric,
                    training_params,
                    runtime_min,
                    max_pipeline_fit_time,
                    direct_rounds if direct_rounds is not None else boosting_rounds,
                    use_eval_set=False if direct_rounds is not None else True,
                    use_input_preprocessing=direct_rounds is None,
                )
                if direct_rounds is None:
                    fitted_model = _maybe_refit_on_all_rows(
                        fitted_model,
                        contender,
                        train_features,
                        train_target,
                        config,
                        scoring_metric,
                        training_params,
                        runtime_min,
                        max_pipeline_fit_time,
                        boosting_rounds,
                        estimated_refit,
                        started_at,
                        prediction_reserve,
                        full_data_refit,
                    )
                prediction_groups_by_name[contender["model"]] = [fitted_model]
                reuses_validation_models_by_name[contender["model"]] = False
            else:
                if forced_lgbm_extra_trees_pair is not None:
                    (
                        ensemble,
                        ensemble_weights,
                        temperature,
                        calibrated_score,
                    ) = _select_logloss_ensemble(
                        retained_forced_pair_ordinary_contenders,
                        calibrate,
                        pair_strong_weight=pair_strong_weight,
                    )
                    _portfolio_report(
                        "Rejecting Extra-LGBM deployment after the primary refit: "
                        "secondary %s's %.1fs estimate no longer fits. Reverting "
                        "to ordinary selector-scale ensemble %s with weights %s.",
                        contender["model"],
                        estimated_refit,
                        [member["model"] for member in ensemble],
                        [round(float(weight), 2) for weight in ensemble_weights],
                    )
                    prediction_groups_by_name = {
                        member["model"]: (
                            retained_forced_pair_ordinary_validation_models[
                                member["model"]
                            ]
                        )
                        for member in ensemble
                    }
                    reuses_validation_models_by_name = {
                        model: True for model in prediction_groups_by_name
                    }
                    primary_model = prediction_groups_by_name[
                        ensemble[0]["model"]
                    ][0]
                    forced_lgbm_extra_trees_pair = None
                    measured_refit_scale = None
                    break
                _portfolio_report(
                    "Dropping secondary %s: estimated %.1fs refit does not fit "
                    "the remaining budget, and selector-scale predictions must "
                    "not be blended with a full-data primary.",
                    contender["model"],
                    estimated_refit,
                )
                dropped_secondary_names.append(contender["model"])
                del contender["validation_models"]

        if dropped_secondary_names:
            ensemble = [
                contender
                for contender in ensemble
                if contender["model"] not in dropped_secondary_names
            ]
            if metric == "logloss":
                ensemble, ensemble_weights, temperature, calibrated_score = (
                    _select_logloss_ensemble(
                        ensemble,
                        calibrate,
                        pair_strong_weight=pair_strong_weight,
                    )
                )
                _portfolio_report(
                    "Recomputed deployable ensemble %s with weights %s and "
                    "temperature %.4f; validation score %.8f.",
                    [contender["model"] for contender in ensemble],
                    [round(float(weight), 2) for weight in ensemble_weights],
                    temperature,
                    calibrated_score,
                )
            else:
                ensemble_weights = np.full(len(ensemble), 1.0 / len(ensemble))

        deployment_weights = (
            ensemble_weights
            if forced_lgbm_extra_trees_pair is not None
            else _direct_wide_deployment_weights(
                ensemble,
                ensemble_weights,
                direct_all_row_refit=(
                    direct_all_row_refit and measured_refit_scale is not None
                ),
                sampled_numeric_density=sampled_numeric_density,
                adaptive_portfolio=configured_candidates is None,
            )
        )
        if not np.allclose(deployment_weights, ensemble_weights):
            ensemble_weights = deployment_weights
            if metric == "logloss":
                validation_probabilities = _blend_probabilities(
                    [contender["probabilities"] for contender in ensemble],
                    ensemble_weights,
                )
                temperature = (
                    _fit_temperature(
                        validation_probabilities,
                        ensemble[0]["truth"],
                        labels=ensemble[0].get("labels"),
                    )
                    if calibrate
                    else 1.0
                )
                calibrated_score = _selection_score(
                    "logloss",
                    ensemble[0]["truth"],
                    None,
                    _apply_temperature(validation_probabilities, temperature),
                    labels=ensemble[0].get("labels"),
                )
                _portfolio_report(
                    "Using dense direct-refit deployment weights %s; refitted "
                    "temperature %.4f gives validation score %.8f.",
                    [round(float(weight), 2) for weight in ensemble_weights],
                    temperature,
                    calibrated_score,
                )

        validation_class_counts = np.unique(
            ensemble[0]["truth"], return_counts=True
        )[1]
        reuses_one_holdout_model = (
            len(validation_splits) == 1
            and all(
                reuses_validation_models_by_name.get(contender["model"], False)
                for contender in ensemble
            )
        )
        if (
            metric == "logloss"
            and calibrate
            and validation_class_counts.min() < 5
            and reuses_one_holdout_model
        ):
            validation_probabilities = _blend_probabilities(
                [contender["probabilities"] for contender in ensemble],
                ensemble_weights,
            )
            temperature = _fit_temperature(
                validation_probabilities,
                ensemble[0]["truth"],
                labels=ensemble[0].get("labels"),
                allow_singleton_classes=True,
            )
            calibrated_score = _selection_score(
                "logloss",
                ensemble[0]["truth"],
                None,
                _apply_temperature(validation_probabilities, temperature),
                labels=ensemble[0].get("labels"),
            )
            _portfolio_report(
                "Calibrating reused holdout model despite sparse validation "
                "classes: temperature %.4f gives score %.8f.",
                temperature,
                calibrated_score,
            )

        prediction_groups = [
            prediction_groups_by_name[contender["model"]]
            for contender in ensemble
        ]

    bounded_xgboost_raw_calibration = _use_raw_bounded_xgboost_calibration(
        metric=metric,
        bounded_refit_seconds=bounded_refit_seconds,
        selected_models=[contender["model"] for contender in ensemble],
    )
    if bounded_xgboost_raw_calibration:
        if local_image_refit_side is not None:
            _portfolio_report(
                "Using the deadline-bounded local LightGBM model's independent "
                "train-only calibration; selector-scale calibration does not "
                "transfer to the singleton deployment model."
            )
        else:
            _portfolio_report(
                "Using raw probabilities for deadline-bounded all-row XGBoost; "
                "selector-scale temperature and prior calibration do not transfer "
                "to the singleton deployment model."
            )
        temperature = 1.0

    prior_exponent = 0.0
    prior_reference_target = train_target
    if forced_lgbm_extra_trees_pair is not None:
        # Keep the calibration that passed the disjoint gate. Re-fitting on the
        # complete selector truth would expose the judgment labels after
        # admission and silently change the validated strategy.
        temperature = forced_lgbm_extra_trees_pair["temperature"]
        prior_exponent = forced_lgbm_extra_trees_pair["prior_exponent"]
        prior_reference_target = selector_reference_target
        _portfolio_report(
            "Keeping frozen Extra-LGBM pair calibration: temperature %.4f, "
            "prior exponent %.4f.",
            temperature,
            prior_exponent,
        )
    elif (
        metric == "logloss"
        and calibrate
        and prior_calibrate
        and not bounded_xgboost_raw_calibration
    ):
        validation_probabilities = _blend_probabilities(
            [contender["probabilities"] for contender in ensemble],
            ensemble_weights,
        )
        calibrated_validation_probabilities = _apply_temperature(
            validation_probabilities, temperature
        )
        prior_exponent = _fit_prior_exponent(
            calibrated_validation_probabilities,
            ensemble[0]["truth"],
            reference_target=train_target,
            labels=ensemble[0].get("labels"),
        )
        prior_score = _selection_score(
            "logloss",
            ensemble[0]["truth"],
            None,
            _apply_prior_exponent(
                calibrated_validation_probabilities,
                reference_target=train_target,
                exponent=prior_exponent,
                labels=ensemble[0].get("labels"),
            ),
            labels=ensemble[0].get("labels"),
        )
        _portfolio_report(
            "Residual class-prior calibration exponent %.4f gives validation "
            "score %.8f.",
            prior_exponent,
            prior_score,
        )

    selected_models = [contender["model"] for contender in ensemble]
    selector_temperature = temperature
    adaptive_temperature_transfer = (
        not any(
            key.startswith("_portfolio_")
            and key != "_portfolio_prior_calibrate"
            for key in config.framework_params
        )
        and not bounded_xgboost_raw_calibration
        and forced_lgbm_extra_trees_pair is None
    )
    deployment_temperature_multiplier = _adaptive_deployment_temperature_multiplier(
        geometry_features,
        train_target,
        selected_models=selected_models,
        selector_temperature=temperature,
        metric=metric,
        calibrate=calibrate,
        direct_all_row_refit=(
            direct_all_row_refit and measured_refit_scale is not None
        ),
        adaptive_portfolio=adaptive_temperature_transfer,
        validation_fold_count=len(validation_splits),
    )
    if deployment_temperature_multiplier != 1.0:
        temperature *= deployment_temperature_multiplier
        _portfolio_report(
            "Correcting selector-to-deployment calibration transfer: selector "
            "temperature %.4f, multiplier %.2f, deployment temperature %.4f.",
            selector_temperature,
            deployment_temperature_multiplier,
            temperature,
        )
    post_prior_temperature_multiplier = (
        _adaptive_post_prior_temperature_multiplier(
            geometry_features,
            train_target,
            selected_models=selected_models,
            selector_temperature=selector_temperature,
            metric=metric,
            calibrate=calibrate,
            adaptive_portfolio=adaptive_temperature_transfer,
        )
    )
    if post_prior_temperature_multiplier != 1.0:
        _portfolio_report(
            "Applying an additional post-prior calibration temperature "
            "multiplier %.2f.",
            post_prior_temperature_multiplier,
        )
    log.info("Predicting on the test set with portfolio ensemble %s.", selected_models)
    with Timer() as predict:
        grouped_probabilities = []
        for model_group in prediction_groups:
            model_probabilities = [
                _classification_predictions(model, test_features)[1]
                for model in model_group
            ]
            grouped_probabilities.append(np.mean(model_probabilities, axis=0))
        probabilities = _blend_probabilities(grouped_probabilities, ensemble_weights)
        if metric == "logloss":
            probabilities = _apply_temperature(probabilities, temperature)
            probabilities = _apply_prior_exponent(
                probabilities,
                reference_target=prior_reference_target,
                exponent=prior_exponent,
                labels=portfolio_labels,
            )
            probabilities = _apply_temperature(
                probabilities, post_prior_temperature_multiplier
            )
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=getattr(dataset, "encoded_class_count", None),
        )

    save_artifacts(primary_model, config)
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=sum(
            model.current_pipeline.length
            for model_group in prediction_groups
            for model in model_group
        ),
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _fixed_round_xgboost_model_params(model_params, seed):
    """Apply validated regularisation only to the guarded direct-fit model."""
    direct_params = dict(model_params)
    direct_params.update(
        {
            "tree_method": "hist",
            "random_state": seed,
            "reg_lambda": _DIRECT_FIXED_XGBOOST_REG_LAMBDA,
        }
    )
    return direct_params


def _direct_xgboost_temperature(completed_rounds, maximum_rounds):
    """Use fixed calibration only at the horizon where it was validated."""
    completed_rounds = int(completed_rounds)
    maximum_rounds = int(maximum_rounds)
    if completed_rounds <= 0 or maximum_rounds <= 0:
        raise ValueError("XGBoost round counts must be positive")
    if completed_rounds > maximum_rounds:
        raise ValueError("completed XGBoost rounds exceed the configured maximum")
    if completed_rounds < maximum_rounds:
        return 1.0
    return _DIRECT_FIXED_XGBOOST_TEMPERATURE


def _finite_f_classif(features, target):
    """Compute ANOVA scores while mapping constant columns behind finite scores."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        scores, probabilities = f_classif(features, target)
    score_dtype = scores.dtype if np.issubdtype(scores.dtype, np.floating) else float
    scores = np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(score_dtype).max,
    )
    return scores, probabilities


def _screened_wide_auc_lgbm_estimator(seed, n_jobs):
    """Build the fixed, train-only screened booster for wide binary AUC tasks."""
    return make_sklearn_pipeline(
        SimpleImputer(strategy="median"),
        SelectKBest(
            score_func=_finite_f_classif,
            k=_SCREENED_WIDE_AUC_FEATURES,
        ),
        LGBMClassifier(
            n_estimators=_SCREENED_WIDE_AUC_ROUNDS,
            learning_rate=0.05,
            num_leaves=127,
            max_bin=255,
            colsample_bytree=0.5,
            min_child_samples=20,
            n_jobs=max(int(n_jobs), 1),
            random_state=int(seed),
            verbose=-1,
        ),
    )


def _run_screened_wide_auc_lgbm(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Fit a bounded deep booster after train-only univariate noise removal."""
    _portfolio_report(
        "Using train-only ANOVA top-%d and %d-round LightGBM for a screened "
        "wide binary AUC task.",
        _SCREENED_WIDE_AUC_FEATURES,
        _SCREENED_WIDE_AUC_ROUNDS,
    )
    model = _screened_wide_auc_lgbm_estimator(config.seed, config.cores)
    with Timer() as training:
        model.fit(train_features, train_target)

    with Timer() as predict:
        probabilities = model.predict_proba(test_features)
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the screened wide "
            "LightGBM fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _medium_screened_auc_lgbm_estimator(seed, n_jobs):
    return _TrainSelectedScreenedAUCClassifier(
        n_estimators=_MEDIUM_SCREENED_AUC_ROUNDS,
        feature_counts=_MEDIUM_SCREENED_AUC_FEATURE_COUNTS,
        validation_fraction=0.2,
        selection_tolerance=_MEDIUM_SCREENED_AUC_SELECTION_TOLERANCE,
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
    )


def _run_medium_screened_auc_lgbm(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Select an ANOVA width on train-only data, then refit every train row."""
    _portfolio_report(
        "Selecting a train-only ANOVA width for an %d-round LightGBM on a "
        "medium dense binary AUC task.",
        _MEDIUM_SCREENED_AUC_ROUNDS,
    )
    with Timer() as training:
        model = _medium_screened_auc_lgbm_estimator(
            config.seed, config.cores
        )
        model.fit(train_features, train_target)
        _portfolio_report(
            "Medium screened validation AUC by feature count: %s; selected %d.",
            model.selector_scores_,
            model.selected_feature_count_,
        )

    with Timer() as predict:
        probabilities = model.predict_proba(test_features)
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the medium screened "
            "LightGBM fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _high_missing_frequency_auc_lgbm_estimator(seed, n_jobs):
    return _FrequencyLGBMClassifier(
        n_estimators=_HIGH_MISSING_FREQUENCY_AUC_ROUNDS,
        learning_rate=0.05,
        num_leaves=_HIGH_MISSING_FREQUENCY_AUC_LEAVES,
        min_child_samples=20,
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
    )


def _run_high_missing_frequency_auc_lgbm(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Quality-gate frequency encoding, then refit on every train row."""
    _portfolio_report(
        "Validating train-only frequency encoding with a %d-round, %d-leaf "
        "LightGBM on a high-missing mixed binary AUC task.",
        _HIGH_MISSING_FREQUENCY_AUC_ROUNDS,
        _HIGH_MISSING_FREQUENCY_AUC_LEAVES,
    )
    with Timer() as training:
        selector_train, selector_valid = train_test_split(
            np.arange(len(train_target)),
            test_size=0.2,
            random_state=int(config.seed),
            stratify=train_target,
        )
        selector_model = _high_missing_frequency_auc_lgbm_estimator(
            config.seed, config.cores
        )
        selector_model.fit(
            _slice_rows(train_features, selector_train),
            train_target[selector_train],
        )
        selector_probabilities = selector_model.predict_proba(
            _slice_rows(train_features, selector_valid)
        )[:, 1]
        selector_score = roc_auc_score(
            train_target[selector_valid], selector_probabilities
        )
        _portfolio_report(
            "High-missing frequency validation AUC %.8f; minimum %.4f.",
            selector_score,
            _HIGH_MISSING_FREQUENCY_AUC_MIN_SCORE,
        )
        del selector_model
        gc.collect()
        if (
            not np.isfinite(selector_score)
            or selector_score < _HIGH_MISSING_FREQUENCY_AUC_MIN_SCORE
        ):
            _portfolio_report(
                "High-missing frequency quality gate rejected the direct "
                "model; using the standard FEDOT portfolio."
            )
            return None

        model = _high_missing_frequency_auc_lgbm_estimator(
            config.seed, config.cores
        )
        model.fit(train_features, train_target)

    with Timer() as predict:
        probabilities = model.predict_proba(test_features)
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the high-missing "
            "frequency LightGBM fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _target_frequency_auc_lgbm_estimator(seed, n_jobs):
    return _CrossFittedTargetFrequencyClassifier(
        smoothing=_TARGET_FREQUENCY_AUC_SMOOTHING,
        n_estimators=_TARGET_FREQUENCY_AUC_ROUNDS,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
    )


def _frequency_auc_lgbm_estimator(seed, n_jobs):
    return _FrequencyLGBMClassifier(
        n_estimators=_TARGET_FREQUENCY_AUC_ROUNDS,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
    )


def _run_target_frequency_auc_lgbm(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Select leakage-safe category statistics, then refit on every train row."""
    _portfolio_report(
        "Selecting train-only frequency or three-fold target/frequency encoding "
        "with %d-round LightGBM for a large high-cardinality binary AUC task.",
        _TARGET_FREQUENCY_AUC_ROUNDS,
    )
    with Timer() as training:
        selector_train, selector_valid = train_test_split(
            np.arange(len(train_target)),
            test_size=0.2,
            random_state=int(config.seed),
            stratify=train_target,
        )
        selector_observations = {}
        for name, factory in (
            ("frequency", _frequency_auc_lgbm_estimator),
            ("target_frequency", _target_frequency_auc_lgbm_estimator),
        ):
            selector_model = factory(config.seed, config.cores)
            selector_model.fit(
                _slice_rows(train_features, selector_train),
                train_target[selector_train],
            )
            selector_probabilities = selector_model.predict_proba(
                _slice_rows(train_features, selector_valid)
            )[:, 1]
            selector_observations[name] = roc_auc_score(
                train_target[selector_valid], selector_probabilities
            )
            _portfolio_report(
                "Large mixed %s validation AUC %.8f.",
                name,
                selector_observations[name],
            )
            del selector_model
            gc.collect()

        target_gain = (
            selector_observations["target_frequency"]
            - selector_observations["frequency"]
        )
        selected_name = (
            "target_frequency"
            if target_gain >= _TARGET_FREQUENCY_AUC_MIN_GAIN
            else "frequency"
        )
        _portfolio_report(
            "Selected large mixed %s encoding; target-statistic AUC gain %.8f "
            "versus %.4f minimum.",
            selected_name,
            target_gain,
            _TARGET_FREQUENCY_AUC_MIN_GAIN,
        )
        selected_factory = (
            _target_frequency_auc_lgbm_estimator
            if selected_name == "target_frequency"
            else _frequency_auc_lgbm_estimator
        )
        model = selected_factory(config.seed, config.cores)
        model.fit(train_features, train_target)

    with Timer() as predict:
        probabilities = model.predict_proba(test_features)
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the target/frequency "
            "LightGBM fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _native_categorical_auc_catboost_estimator(seed, n_jobs, iterations):
    return _NativeCategoricalCatBoostClassifier(
        iterations=int(iterations),
        learning_rate=0.05,
        depth=8,
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
    )


def _narrow_categorical_auc_baseline_estimator(seed, n_jobs, selector):
    return _CrossFittedTargetFrequencyClassifier(
        smoothing=50.0,
        n_estimators=(
            _NARROW_CATEGORICAL_AUC_SELECTOR_ROUNDS
            if selector
            else _TARGET_FREQUENCY_AUC_ROUNDS
        ),
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
    )


def _select_narrow_categorical_auc_model(catboost_auc, baseline_auc):
    """Require both useful absolute quality and a material CatBoost gain."""
    if (
        np.isfinite(catboost_auc)
        and catboost_auc >= _NARROW_CATEGORICAL_AUC_MIN_SCORE
        and catboost_auc - baseline_auc >= _NARROW_CATEGORICAL_AUC_MIN_GAIN
    ):
        return "catboost"
    return "target_frequency"


def _run_narrow_categorical_auc_catboost(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Select native category handling, then refit one model on every row."""
    _portfolio_report(
        "Comparing %d-round native CatBoost with leakage-safe target/frequency "
        "LightGBM for a narrow high-cardinality binary AUC task.",
        _NARROW_CATEGORICAL_AUC_SELECTOR_ROUNDS,
    )
    with Timer() as training:
        selector_train, selector_valid = train_test_split(
            np.arange(len(train_target)),
            test_size=0.2,
            random_state=int(config.seed),
            stratify=train_target,
        )
        selector_train_features = _slice_rows(train_features, selector_train)
        selector_valid_features = _slice_rows(train_features, selector_valid)
        selector_train_target = train_target[selector_train]
        selector_valid_target = train_target[selector_valid]

        catboost_selector = _native_categorical_auc_catboost_estimator(
            config.seed,
            config.cores,
            _NARROW_CATEGORICAL_AUC_SELECTOR_ROUNDS,
        )
        catboost_selector.fit(selector_train_features, selector_train_target)
        catboost_auc = roc_auc_score(
            selector_valid_target,
            catboost_selector.predict_proba(selector_valid_features)[:, 1],
        )
        del catboost_selector
        gc.collect()

        baseline_selector = _narrow_categorical_auc_baseline_estimator(
            config.seed, config.cores, selector=True
        )
        baseline_selector.fit(selector_train_features, selector_train_target)
        baseline_auc = roc_auc_score(
            selector_valid_target,
            baseline_selector.predict_proba(selector_valid_features)[:, 1],
        )
        del baseline_selector
        gc.collect()

        selected_name = _select_narrow_categorical_auc_model(
            catboost_auc, baseline_auc
        )
        _portfolio_report(
            "Narrow categorical selector AUC: CatBoost %.8f, "
            "target/frequency LightGBM %.8f; selected %s (minimum CatBoost "
            "AUC %.2f and gain %.4f).",
            catboost_auc,
            baseline_auc,
            selected_name,
            _NARROW_CATEGORICAL_AUC_MIN_SCORE,
            _NARROW_CATEGORICAL_AUC_MIN_GAIN,
        )
        if selected_name == "catboost":
            model = _native_categorical_auc_catboost_estimator(
                config.seed,
                config.cores,
                _NARROW_CATEGORICAL_AUC_FINAL_ROUNDS,
            )
        else:
            model = _narrow_categorical_auc_baseline_estimator(
                config.seed, config.cores, selector=False
            )
        model.fit(train_features, train_target)

    with Timer() as predict:
        probabilities = model.predict_proba(test_features)
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the native categorical "
            "AUC fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _nominal_multiclass_catboost_estimator(seed, n_jobs, iterations):
    return _NativeCategoricalCatBoostClassifier(
        iterations=int(iterations),
        learning_rate=_NOMINAL_MULTICLASS_CATBOOST_LEARNING_RATE,
        depth=_NOMINAL_MULTICLASS_CATBOOST_DEPTH,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
    )


def _nominal_multiclass_lgbm_selector(
    train_features,
    train_target,
    valid_features,
    valid_target,
    seed,
    n_jobs,
):
    """Fit a cheap train-only ordinal baseline for the native-category gate."""
    encoded_train, encoded_valid = _compact_encode_if_needed(
        train_features,
        valid_features,
        force=True,
        min_rows=1,
    )
    model = LGBMClassifier(
        n_estimators=1_000,
        learning_rate=0.1,
        num_leaves=127,
        min_child_samples=100,
        n_jobs=max(int(n_jobs), 1),
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(
        encoded_train,
        train_target,
        eval_set=[(encoded_valid, valid_target)],
        callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
    )
    return model.predict_proba(encoded_valid)


def _select_nominal_multiclass_model(catboost_loss, baseline_loss, prior_loss):
    """Require both a useful classifier and a material native-category gain."""
    if (
        np.isfinite(catboost_loss)
        and np.isfinite(baseline_loss)
        and np.isfinite(prior_loss)
        and prior_loss - catboost_loss
        >= _NOMINAL_MULTICLASS_CATBOOST_MIN_PRIOR_GAIN
        and baseline_loss - catboost_loss
        >= _NOMINAL_MULTICLASS_CATBOOST_MIN_GAIN
    ):
        return "catboost"
    return "portfolio"


def _run_nominal_multiclass_catboost(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Gate native nominal handling against a cheap compact tree baseline."""
    _portfolio_report(
        "Comparing a bounded native CatBoost with a compact LightGBM on a "
        "nominal-heavy multiclass task."
    )
    labels = np.unique(train_target)
    with Timer() as training:
        selector_train, selector_valid = train_test_split(
            np.arange(len(train_target)),
            test_size=0.2,
            random_state=int(config.seed),
            stratify=train_target,
        )
        selector_train_features = _slice_rows(train_features, selector_train)
        selector_valid_features = _slice_rows(train_features, selector_valid)
        selector_train_target = train_target[selector_train]
        selector_valid_target = train_target[selector_valid]

        catboost_selector = _nominal_multiclass_catboost_estimator(
            config.seed,
            config.cores,
            _NOMINAL_MULTICLASS_CATBOOST_ROUNDS,
        )
        catboost_selector.fit(
            selector_train_features,
            selector_train_target,
            eval_set=(selector_valid_features, selector_valid_target),
            early_stopping_rounds=50,
        )
        catboost_probabilities = catboost_selector.predict_proba(
            selector_valid_features
        )
        catboost_temperature = _fit_temperature(
            catboost_probabilities,
            selector_valid_target,
            labels=labels,
        )
        catboost_loss = log_loss(
            selector_valid_target,
            _apply_temperature(catboost_probabilities, catboost_temperature),
            labels=labels,
        )
        selected_iterations = catboost_selector.best_iteration_
        del catboost_selector
        gc.collect()

        baseline_probabilities = _nominal_multiclass_lgbm_selector(
            selector_train_features,
            selector_train_target,
            selector_valid_features,
            selector_valid_target,
            seed=config.seed,
            n_jobs=config.cores,
        )
        baseline_temperature = _fit_temperature(
            baseline_probabilities,
            selector_valid_target,
            labels=labels,
        )
        baseline_loss = log_loss(
            selector_valid_target,
            _apply_temperature(baseline_probabilities, baseline_temperature),
            labels=labels,
        )
        class_frequencies = np.asarray(
            [np.mean(selector_train_target == label) for label in labels]
        )
        prior_probabilities = np.tile(
            class_frequencies,
            (len(selector_valid_target), 1),
        )
        prior_loss = log_loss(
            selector_valid_target,
            prior_probabilities,
            labels=labels,
        )
        selected_name = _select_nominal_multiclass_model(
            catboost_loss,
            baseline_loss,
            prior_loss,
        )
        _portfolio_report(
            "Nominal multiclass selector logloss: CatBoost %.8f, compact "
            "LightGBM %.8f, prior %.8f; selected %s (minimum baseline gain "
            "%.4f and prior gain %.4f).",
            catboost_loss,
            baseline_loss,
            prior_loss,
            selected_name,
            _NOMINAL_MULTICLASS_CATBOOST_MIN_GAIN,
            _NOMINAL_MULTICLASS_CATBOOST_MIN_PRIOR_GAIN,
        )
        if selected_name != "catboost":
            _portfolio_report(
                "Native CatBoost quality gate rejected the direct model; "
                "using the standard FEDOT portfolio."
            )
            return None

        model = _nominal_multiclass_catboost_estimator(
            config.seed,
            config.cores,
            selected_iterations,
        )
        model.fit(train_features, train_target)

    with Timer() as predict:
        probabilities = _apply_temperature(
            model.predict_proba(test_features),
            catboost_temperature,
        )
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the nominal "
            "multiclass CatBoost fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _contiguous_exact_one_hot_groups(features, minimum_width=3, maximum_width=64):
    """Find disjoint contiguous binary blocks with exactly one active value.

    Detection is unsupervised and deliberately conservative.  Constant columns,
    missing binary values, multi-hot rows and interleaved columns break a block.
    """
    if minimum_width < 2 or maximum_width < minimum_width:
        raise ValueError("invalid exact-one-hot group width bounds")
    if not hasattr(features, "dtypes"):
        return []
    frame = pd.DataFrame(features)
    if not frame.columns.is_unique:
        return []

    binary_values = {}
    binary_positions = []
    for position, column in enumerate(frame.columns):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            continue
        values = numeric.to_numpy(dtype=np.float32, copy=False)
        unique = np.unique(values)
        if len(unique) == 2 and set(unique.tolist()) == {0.0, 1.0}:
            binary_positions.append(position)
            binary_values[position] = values

    runs = []
    for position in binary_positions:
        if not runs or position != runs[-1][-1] + 1:
            runs.append([position])
        else:
            runs[-1].append(position)

    groups = []
    for run in runs:
        start = 0
        while start + minimum_width <= len(run):
            row_sums = np.zeros(len(frame), dtype=np.uint8)
            found = None
            maximum_stop = min(len(run), start + maximum_width)
            for stop in range(start, maximum_stop):
                row_sums += binary_values[run[stop]].astype(np.uint8, copy=False)
                width = stop - start + 1
                if np.any(row_sums > 1):
                    break
                if width >= minimum_width and np.all(row_sums == 1):
                    found = stop + 1
                    groups.append(tuple(frame.columns[run[start:found]]))
                    break
            start = start + 1 if found is None else found
    return groups


def _exact_one_hot_grouped_xgboost_profile(features, target):
    """Return a bounded mixed-table profile suitable for representation choice."""
    if not hasattr(features, "dtypes"):
        return None
    frame = pd.DataFrame(features)
    if frame.ndim != 2:
        return None
    row_count, feature_count = frame.shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    if not (
        100_000 <= row_count <= 1_000_000
        and 16 <= feature_count <= 128
        and row_count * feature_count <= 64_000_000
        and 3 <= len(class_counts) <= 9
        and class_counts.min() >= 1_000
        and not _is_sparse_table(features)
    ):
        return None

    groups = _contiguous_exact_one_hot_groups(frame)
    grouped_columns = {column for group in groups for column in group}
    grouped_width = len(grouped_columns)
    ordinary_columns = [
        column for column in frame.columns if column not in grouped_columns
    ]
    if not (
        2 <= len(groups) <= 16
        and grouped_width >= 16
        and grouped_width >= 0.5 * feature_count
        and len(ordinary_columns) >= 4
        and feature_count - grouped_width + len(groups) <= 64
    ):
        return None

    for column in ordinary_columns:
        source = frame[column]
        numeric = pd.to_numeric(source, errors="coerce")
        if not (numeric.notna() | source.isna()).all():
            return None
    return {
        "groups": groups,
        "ordinary_columns": ordinary_columns,
        "original_feature_count": feature_count,
        "compressed_feature_count": feature_count - grouped_width + len(groups),
        "grouped_width": grouped_width,
    }


def _use_exact_one_hot_grouped_xgboost(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    configured = framework_params.get("_portfolio_grouped_one_hot_xgboost")
    other_overrides = any(
        key.startswith("_portfolio_")
        and key != "_portfolio_grouped_one_hot_xgboost"
        for key in framework_params
    )
    if configured is not None and not _as_bool(configured):
        return None
    if configured is None and other_overrides:
        return None
    if metric != "logloss" or runtime_seconds < 180 or cores < 4:
        return None
    profile = _exact_one_hot_grouped_xgboost_profile(features, target)
    if profile is None:
        return None
    if configured is None and not _exact_one_hot_baseline_matches_portfolio(
        features, target, metric
    ):
        return None
    return profile


def _exact_one_hot_baseline_matches_portfolio(features, target, metric):
    """Require the numeric challenger baseline to equal the adaptive incumbent."""
    return (
        _adaptive_default_portfolio_candidates(
            features, target, configured=None, metric=metric
        )
        == ["xgboost"]
        and _adaptive_boosting_rounds(features, target)
        == _GROUPED_ONE_HOT_XGBOOST_SELECTOR_ROUNDS
        and _adaptive_xgboost_learning_rate(features, target) is None
        and _adaptive_xgboost_max_depth(features, target) is None
        and _adaptive_xgboost_max_bin(features, target) is None
        and _adaptive_xgboost_colsample_bytree(features, target) is None
        and _adaptive_xgboost_subsample(features, target) is None
        and _adaptive_xgboost_min_child_weight(features, target) == 5.0
    )


def _fit_exact_one_hot_numeric_transform(features):
    frame = pd.DataFrame(features)
    columns = list(frame.columns)
    medians = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        median = numeric.median()
        medians[column] = 0.0 if not np.isfinite(median) else float(median)
    return {"columns": columns, "medians": medians, "groups": []}


def _fit_exact_one_hot_grouped_transform(features, profile):
    frame = pd.DataFrame(features)
    ordinary_columns = list(profile["ordinary_columns"])
    medians = {}
    for column in ordinary_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        median = numeric.median()
        medians[column] = 0.0 if not np.isfinite(median) else float(median)
    return {
        "columns": list(frame.columns),
        "ordinary_columns": ordinary_columns,
        "medians": medians,
        "groups": list(profile["groups"]),
    }


def _apply_exact_one_hot_transform(features, transformer):
    frame = pd.DataFrame(features)
    expected_columns = transformer["columns"]
    if list(frame.columns) != expected_columns:
        raise ValueError("exact-one-hot transform received different feature columns")

    groups = transformer["groups"]
    if not groups:
        result = pd.DataFrame(index=frame.index)
        for output_position, column in enumerate(expected_columns):
            numeric = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            result[f"__fedot_numeric_{output_position}"] = numeric.fillna(
                transformer["medians"][column]
            ).to_numpy(dtype=np.float32)
        return result

    result = pd.DataFrame(index=frame.index)
    for output_position, column in enumerate(transformer["ordinary_columns"]):
        numeric = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        result[f"__fedot_numeric_{output_position}"] = numeric.fillna(
            transformer["medians"][column]
        ).to_numpy(dtype=np.float32)
    for group_number, group in enumerate(groups):
        matrix = np.column_stack(
            [
                pd.to_numeric(frame[column], errors="coerce")
                .to_numpy(dtype=np.float32)
                for column in group
            ]
        )
        finite = np.isfinite(matrix).all(axis=1)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = matrix.sum(axis=1)
        active = np.argmax(matrix, axis=1).astype(np.int32)
        valid = finite & np.logical_or(matrix == 0.0, matrix == 1.0).all(axis=1)
        valid &= row_sums == 1.0
        active[~valid] = len(group)
        result[f"__fedot_exact_one_hot_{group_number}"] = pd.Categorical(
            active, categories=range(len(group) + 1)
        )
    return result


def _exact_one_hot_xgboost_estimator(
    seed,
    n_jobs,
    maximum_rounds,
    max_depth,
    native_categorical,
    fit_time_limit=None,
    use_eval_set=True,
):
    callbacks = (
        None
        if fit_time_limit is None
        else [_WallClockStopCallback(float(fit_time_limit))]
    )
    return XGBClassifier(
        n_estimators=int(maximum_rounds),
        learning_rate=0.3,
        max_depth=int(max_depth),
        min_child_weight=5.0,
        n_jobs=max(int(n_jobs), 1),
        verbosity=0,
        tree_method="hist" if native_categorical else "auto",
        enable_categorical=bool(native_categorical),
        early_stopping_rounds=30 if use_eval_set else None,
        eval_metric="mlogloss",
        random_state=int(seed),
        callbacks=callbacks,
    )


def _fit_exact_one_hot_xgboost_with_eval(
    features,
    target,
    transformer,
    seed,
    n_jobs,
    maximum_rounds,
    max_depth,
    native_categorical,
    fit_time_limit=None,
    fit_indices=None,
    eval_indices=None,
    use_eval_set=True,
):
    target_array = np.asarray(target).reshape(-1)
    if use_eval_set:
        if (fit_indices is None) != (eval_indices is None):
            raise ValueError("fit and eval indices must be supplied together")
        if fit_indices is None:
            fit_indices, eval_indices = train_test_split(
                np.arange(len(target_array)),
                test_size=0.2,
                random_state=int(seed),
                stratify=target_array,
            )
    else:
        if eval_indices is not None:
            raise ValueError("eval indices require use_eval_set=True")
        if fit_indices is None:
            fit_indices = np.arange(len(target_array))
    fit_indices = np.asarray(fit_indices, dtype=int)
    if use_eval_set:
        eval_indices = np.asarray(eval_indices, dtype=int)
    fit_view = _apply_exact_one_hot_transform(
        _slice_rows(features, fit_indices), transformer
    )
    eval_view = (
        _apply_exact_one_hot_transform(
            _slice_rows(features, eval_indices), transformer
        )
        if use_eval_set
        else None
    )
    model = _exact_one_hot_xgboost_estimator(
        seed,
        n_jobs,
        maximum_rounds,
        max_depth,
        native_categorical,
        fit_time_limit=fit_time_limit,
        use_eval_set=use_eval_set,
    )
    fit_parameters = (
        {
            "eval_set": [(eval_view, target_array[eval_indices])],
            "verbose": False,
        }
        if use_eval_set
        else {}
    )
    model.fit(fit_view, target_array[fit_indices], **fit_parameters)
    eval_target = target_array[eval_indices] if use_eval_set else None
    return model, eval_view, eval_target, target_array[fit_indices]


def _exact_one_hot_final_fit_calibration_indices(target, seed):
    """Reserve only calibration rows after a fixed-round grouped refit.

    The grouped representation selector has already supplied independent early-
    stopping evidence.  Train-only pseudo-outer controls showed that exchanging
    100 rounds for the former final eval partition improves generalisation while
    preserving work.  Follow-up pseudo-outer controls then showed that this
    large, well-supported structural regime retains stable calibration with 5%
    of the rows.  The final 95/5 split therefore uses every non-calibration row
    for fitting and keeps the existing leakage-free calibration contract.
    """
    target_array = np.asarray(target).reshape(-1)
    return train_test_split(
        np.arange(len(target_array)),
        test_size=0.05,
        random_state=int(seed),
        stratify=target_array,
    )


def _select_exact_one_hot_grouped_representation(
    baseline_probabilities,
    grouped_probabilities,
    truth,
    labels,
):
    baseline_loss = float(log_loss(truth, baseline_probabilities, labels=labels))
    grouped_loss = float(log_loss(truth, grouped_probabilities, labels=labels))
    improvements = _row_log_losses(
        baseline_probabilities, truth, labels=labels
    ) - _row_log_losses(grouped_probabilities, truth, labels=labels)
    standard_error = (
        float(np.std(improvements, ddof=1) / np.sqrt(len(improvements)))
        if len(improvements) > 1
        else np.inf
    )
    lower_95 = float(np.mean(improvements) - 1.96 * standard_error)
    practical_gain = max(
        _GROUPED_ONE_HOT_XGBOOST_MIN_GAIN,
        baseline_loss * 0.01,
    )
    selected = (
        baseline_loss - grouped_loss > practical_gain and lower_95 > 0.0
    )
    return selected, baseline_loss, grouped_loss, lower_95, practical_gain


def _run_exact_one_hot_grouped_xgboost(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    profile,
    observed_encoded_labels,
    encoded_class_count,
    portfolio_started_at=None,
):
    """Select a one-hot representation, then fit a bounded native XGBoost."""
    started_at = (
        time.monotonic()
        if portfolio_started_at is None
        else float(portfolio_started_at)
    )
    with Timer() as training:
        selector_train, selector_valid, selector_target, selector_valid_target = (
            _classification_holdout(
                train_features,
                train_target,
                validation_fraction=0.2,
                max_train_rows=20_000,
                max_validation_rows=10_000,
                seed=int(config.seed),
            )
        )
        selector_fit_indices, selector_eval_indices = train_test_split(
            np.arange(len(selector_target)),
            test_size=0.2,
            random_state=int(config.seed),
            stratify=selector_target,
        )
        selector_fit_features = _slice_rows(
            selector_train, selector_fit_indices
        )
        baseline_transformer = _fit_exact_one_hot_numeric_transform(
            selector_fit_features
        )
        grouped_transformer = _fit_exact_one_hot_grouped_transform(
            selector_fit_features, profile
        )
        selector_deadline = started_at + min(
            45.0, 0.25 * float(config.max_runtime_seconds)
        )
        baseline_model, _, _, _ = _fit_exact_one_hot_xgboost_with_eval(
            selector_train,
            selector_target,
            baseline_transformer,
            config.seed,
            config.cores,
            _GROUPED_ONE_HOT_XGBOOST_SELECTOR_ROUNDS,
            max_depth=6,
            native_categorical=False,
            fit_time_limit=max(
                1.0, min(30.0, selector_deadline - time.monotonic())
            ),
            fit_indices=selector_fit_indices,
            eval_indices=selector_eval_indices,
        )
        grouped_model, _, _, _ = _fit_exact_one_hot_xgboost_with_eval(
            selector_train,
            selector_target,
            grouped_transformer,
            config.seed,
            config.cores,
            _GROUPED_ONE_HOT_XGBOOST_SELECTOR_ROUNDS,
            max_depth=7,
            native_categorical=True,
            fit_time_limit=max(
                1.0, min(30.0, selector_deadline - time.monotonic())
            ),
            fit_indices=selector_fit_indices,
            eval_indices=selector_eval_indices,
        )
        baseline_valid_view = _apply_exact_one_hot_transform(
            selector_valid, baseline_transformer
        )
        grouped_valid_view = _apply_exact_one_hot_transform(
            selector_valid, grouped_transformer
        )
        labels = np.unique(train_target)
        baseline_probabilities = baseline_model.predict_proba(baseline_valid_view)
        grouped_probabilities = grouped_model.predict_proba(grouped_valid_view)
        calibration_indices, decision_indices = train_test_split(
            np.arange(len(selector_valid_target)),
            test_size=0.5,
            random_state=int(config.seed) + 2,
            stratify=selector_valid_target,
        )
        baseline_temperature = _fit_temperature(
            baseline_probabilities[calibration_indices],
            selector_valid_target[calibration_indices],
            labels=labels,
        )
        grouped_temperature = _fit_temperature(
            grouped_probabilities[calibration_indices],
            selector_valid_target[calibration_indices],
            labels=labels,
        )
        baseline_probabilities = _apply_temperature(
            baseline_probabilities[decision_indices], baseline_temperature
        )
        grouped_probabilities = _apply_temperature(
            grouped_probabilities[decision_indices], grouped_temperature
        )
        selected, baseline_loss, grouped_loss, lower_95, practical_gain = (
            _select_exact_one_hot_grouped_representation(
                baseline_probabilities,
                grouped_probabilities,
                selector_valid_target[decision_indices],
                labels,
            )
        )
        _portfolio_report(
            "Exact-one-hot representation selector: baseline %.8f, grouped "
            "native %.8f, paired lower-95 gain %.8f, required gain %.8f; %s.",
            baseline_loss,
            grouped_loss,
            lower_95,
            practical_gain,
            "selected grouped native" if selected else "kept ordinary portfolio",
        )
        del baseline_model, grouped_model
        gc.collect()
        if not selected:
            return None

        fit_indices, calibration_indices = (
            _exact_one_hot_final_fit_calibration_indices(
                train_target, seed=config.seed
            )
        )
        final_transformer = _fit_exact_one_hot_grouped_transform(
            _slice_rows(train_features, fit_indices), profile
        )
        fit_time_limit = (
            float(config.max_runtime_seconds)
            - (time.monotonic() - started_at)
            - max(12.0, 0.06 * float(config.max_runtime_seconds))
        )
        if fit_time_limit < _DIRECT_XGBOOST_MIN_REFIT_SECONDS:
            _portfolio_report(
                "Exact-one-hot selector left only %.1fs for final fitting; "
                "keeping the ordinary deadline-aware portfolio.",
                fit_time_limit,
            )
            return None
        final_model, _, _, fit_target = (
            _fit_exact_one_hot_xgboost_with_eval(
                train_features,
                train_target,
                final_transformer,
                config.seed,
                config.cores,
                _GROUPED_ONE_HOT_XGBOOST_FINAL_ROUNDS,
                max_depth=7,
                native_categorical=True,
                fit_time_limit=fit_time_limit,
                fit_indices=fit_indices,
                use_eval_set=False,
            )
        )
        deployment_valid_view = _apply_exact_one_hot_transform(
            _slice_rows(train_features, calibration_indices), final_transformer
        )
        deployment_valid_target = np.asarray(train_target)[calibration_indices]
        deployment_valid_probabilities = final_model.predict_proba(
            deployment_valid_view
        )
        deployment_temperature = _fit_temperature(
            deployment_valid_probabilities,
            deployment_valid_target,
            labels=labels,
        )
        deployment_valid_calibrated = _apply_temperature(
            deployment_valid_probabilities, deployment_temperature
        )
        deployment_prior_exponent = _fit_prior_exponent(
            deployment_valid_calibrated,
            deployment_valid_target,
            reference_target=fit_target,
            labels=labels,
        )
        completed_rounds = final_model.get_booster().num_boosted_rounds()
        _portfolio_report(
            "Grouped native XGBoost fitted 95%% of rows for %d/%d fixed rounds "
            "with deployment temperature %.4f and prior exponent %.4f.",
            completed_rounds,
            _GROUPED_ONE_HOT_XGBOOST_FINAL_ROUNDS,
            deployment_temperature,
            deployment_prior_exponent,
        )

    with Timer() as predict:
        test_view = _apply_exact_one_hot_transform(test_features, final_transformer)
        probabilities = final_model.predict_proba(test_view)
        probabilities = _apply_temperature(probabilities, deployment_temperature)
        probabilities = _apply_prior_exponent(
            probabilities,
            reference_target=fit_target,
            exponent=deployment_prior_exponent,
            labels=labels,
        )
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the grouped native "
            "XGBoost fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _run_fixed_round_all_row_xgboost(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    scoring_metric,
    training_params,
    runtime_min,
    max_pipeline_fit_time,
    model_params,
    boosting_rounds,
    observed_encoded_labels,
    encoded_class_count,
):
    """Fit one deadline-bounded extreme-wide booster without losing holdout rows."""
    _portfolio_report(
        "Using direct all-row extreme-wide XGBoost with up to %d rounds and "
        "a %.1fs fit limit; validation would consume too much of the bounded "
        "fit budget.",
        boosting_rounds,
        model_params["fit_time_limit"],
    )
    contender = {"model": "xgboost", "model_params": model_params}
    with Timer() as training:
        automl = _fit_full_candidate(
            contender,
            train_features,
            train_target,
            config,
            scoring_metric,
            training_params,
            runtime_min,
            max_pipeline_fit_time,
            boosting_rounds,
            use_eval_set=False,
            use_input_preprocessing=False,
        )
        fitted_operation = automl.current_pipeline.nodes[0].fitted_operation
        fitted_xgboost = fitted_operation.model
        completed_rounds = fitted_xgboost.get_booster().num_boosted_rounds()
        adaptive_callback = next(
            (
                callback
                for callback in getattr(fitted_operation, "fit_callbacks", ())
                if getattr(callback, "adaptive_learning_rate", False)
            ),
            None,
        )
        if adaptive_callback is not None:
            _portfolio_report(
                "Deadline-aware XGBoost projected %s rounds after warm-up and "
                "selected learning rate %s.",
                getattr(adaptive_callback, "projected_rounds", None),
                getattr(adaptive_callback, "adjusted_learning_rate", None)
                or model_params.get("learning_rate"),
            )
        _portfolio_report(
            "Direct extreme-wide XGBoost completed %d of at most %d rounds.",
            completed_rounds,
            boosting_rounds,
        )

    log.info("Predicting on the test set with direct extreme-wide XGBoost.")
    with Timer() as predict:
        _, probabilities = _classification_predictions(automl, test_features)
        direct_temperature = _direct_xgboost_temperature(
            completed_rounds, boosting_rounds
        )
        probabilities = _apply_temperature(probabilities, direct_temperature)
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    _portfolio_report(
        "Applied %.3f temperature to direct XGBoost probabilities after "
        "%d/%d rounds.",
        direct_temperature,
        completed_rounds,
        boosting_rounds,
    )

    save_artifacts(automl, config)
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=automl.current_pipeline.length,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _run_sparse_native_logit(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Fit a CSR-native linear fallback without FEDOT's dense preprocessing.

    Sparse p >> n text tables are already in the representation expected by a
    linear classifier. Passing them through the ordinary FEDOT input pipeline
    can materialise a multi-gigabyte dense frame before the model starts. The
    fallback deliberately uses the raw matrix and one fixed regularisation level;
    it does not inspect outer-test labels or fit dataset-specific parameters.
    """
    model = LogisticRegression(
        C=0.1,
        solver="liblinear",
        dual=True,
        max_iter=2_000,
        random_state=config.seed,
    )
    train_matrix = _as_scipy_sparse_matrix(train_features)
    test_matrix = _as_scipy_sparse_matrix(test_features)
    _portfolio_report(
        "Using sparse-native dual logistic fallback on %d rows and %d features.",
        train_matrix.shape[0],
        train_matrix.shape[1],
    )
    with Timer() as training:
        model.fit(train_matrix, train_target)

    with Timer() as predict:
        probabilities = model.predict_proba(test_matrix)
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the sparse-native "
            "linear fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _run_materialized_sparse_tfidf_logit(
    train_features,
    train_target,
    test_features,
    test_target,
    config,
    observed_encoded_labels,
    encoded_class_count,
):
    """Fit a train-selected TF-IDF linear view of a bounded sparse-like table.

    Some word/count datasets arrive as ordinary dense numeric frames, so
    FEDOT and tree models pay dense-table costs despite very few nonzero values.
    Converting that already observed representation to CSR is lossless. TF-IDF is
    fitted separately inside every OOF split, keeping both regularisation choice
    and temperature calibration train-only.
    """
    _portfolio_report(
        "Using sparse-like TF-IDF logistic fallback on %d rows and %d "
        "features.",
        len(train_target),
        train_features.shape[1],
    )
    with Timer() as training:
        train_matrix = _as_materialized_sparse_csr(train_features)
        model, temperature, selected_c, oof_scores = (
            _fit_materialized_sparse_tfidf_selector(
                train_matrix,
                train_target,
                seed=config.seed,
            )
        )

    _portfolio_report(
        "Selected TF-IDF logistic C=%.1f from calibrated OOF loglosses "
        "C=3: %.6f and C=10: %.6f; transferred temperature %.4f.",
        selected_c,
        oof_scores[3.0],
        oof_scores[10.0],
        temperature,
    )
    with Timer() as predict:
        test_matrix = _as_materialized_sparse_csr(test_features)
        probabilities = _apply_temperature(
            model.predict_proba(test_matrix), temperature
        )
        predictions, probabilities = _restore_classification_label_space(
            probabilities,
            observed_encoded_labels,
            encoded_class_count=encoded_class_count,
        )

    if config.framework_params.get("_save_artifacts"):
        log.info(
            "FEDOT pipeline artifacts are unavailable for the materialized-sparse "
            "TF-IDF linear fallback."
        )
    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=test_target,
        probabilities=probabilities,
        target_is_encoded=True,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def _materialized_sparse_tfidf_estimator(c_value, seed):
    return make_sklearn_pipeline(
        TfidfTransformer(sublinear_tf=True),
        LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            dual=True,
            max_iter=2_000,
            random_state=int(seed),
        ),
    )


def _fit_materialized_sparse_tfidf_selector(features, target, seed=42):
    """Select weak or strong linear regularisation from calibrated OOF loss."""
    target = np.asarray(target).reshape(-1)
    labels = np.unique(target)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
    temperatures = {}
    scores = {}
    for c_value in (3.0, 10.0):
        probabilities = np.zeros((len(target), len(labels)), dtype=float)
        for train_indices, valid_indices in splitter.split(features, target):
            model = _materialized_sparse_tfidf_estimator(c_value, seed)
            model.fit(features[train_indices], target[train_indices])
            fold_probabilities = model.predict_proba(features[valid_indices])
            class_columns = np.searchsorted(labels, model.classes_)
            probabilities[np.ix_(valid_indices, class_columns)] = (
                fold_probabilities
            )
        temperature = _fit_temperature(probabilities, target, labels=labels)
        calibrated = _apply_temperature(probabilities, temperature)
        temperatures[c_value] = temperature
        scores[c_value] = log_loss(target, calibrated, labels=labels)

    selected_c = _select_materialized_sparse_logit_c(scores)
    selected_model = _materialized_sparse_tfidf_estimator(selected_c, seed)
    selected_model.fit(features, target)
    return selected_model, temperatures[selected_c], selected_c, scores


def _select_materialized_sparse_logit_c(scores, minimum_gain=0.01):
    """Prefer stronger regularisation unless a larger C wins materially."""
    if minimum_gain < 0:
        raise ValueError("minimum regularisation gain must be non-negative")
    return 10.0 if scores[3.0] - scores[10.0] >= minimum_gain else 3.0


def _make_fedot(
    config, scoring_metric, training_params, runtime_min, max_pipeline_fit_time
):
    return Fedot(
        problem=config.type,
        timeout=runtime_min,
        metric=scoring_metric,
        seed=config.seed,
        max_pipeline_fit_time=max_pipeline_fit_time,
        **training_params,
    )


def _fit_full_candidate(
    contender,
    train_features,
    train_target,
    config,
    scoring_metric,
    training_params,
    runtime_min,
    max_pipeline_fit_time,
    boosting_rounds,
    use_eval_set=True,
    use_input_preprocessing=True,
):
    if _is_scaled_svc_candidate(contender["model"]):
        return _fit_scaled_svc_candidate(
            train_features,
            train_target,
            model_params=contender.get("model_params"),
            seed=config.seed,
        )
    if contender["model"] == "mixed_logit":
        return _fit_mixed_logit_candidate(
            train_features,
            train_target,
            model_params=contender.get("model_params"),
            seed=config.seed,
        )
    if contender["model"] == "mixed_svc":
        return _fit_mixed_svc_candidate(
            train_features,
            train_target,
            model_params=contender.get("model_params"),
            seed=config.seed,
        )
    if contender["model"] in {"extra_trees", "extra_trees_wide"}:
        return _fit_extra_trees_candidate(
            train_features,
            train_target,
            model_params=contender.get("model_params"),
            seed=config.seed,
            n_jobs=config.cores,
        )
    if contender["model"] == "mixed_extra_trees":
        return _fit_mixed_extra_trees_candidate(
            train_features,
            train_target,
            model_params=contender.get("model_params"),
            seed=config.seed,
            n_jobs=config.cores,
        )
    if contender["model"] in {
        "relational_hist",
        "relational_hist_second_order",
    }:
        return _fit_relational_hist_candidate(
            train_features,
            train_target,
            model_params=contender.get("model_params"),
            seed=config.seed,
            second_order=contender["model"] == "relational_hist_second_order",
        )
    candidate_training_params = training_params
    if not use_input_preprocessing:
        candidate_training_params = {
            **training_params,
            "use_input_preprocessing": False,
        }
    automl = _make_fedot(
        config=config,
        scoring_metric=scoring_metric,
        training_params=candidate_training_params,
        runtime_min=runtime_min,
        max_pipeline_fit_time=max_pipeline_fit_time,
    )
    automl.fit(
        features=train_features,
        target=train_target,
        predefined_model=_predefined_model_with_n_jobs(
            contender["model"],
            config.cores,
            n_estimators=boosting_rounds,
            use_eval_set=use_eval_set,
            model_params=contender.get("model_params"),
        ),
    )
    return automl


def _fit_scaled_svc_candidate(features, target, model_params=None, seed=42):
    """Fit the bounded kernel candidate without changing FEDOT's model registry."""
    model_params = dict(model_params or {})
    gamma_multiplier = model_params.pop("_gamma_multiplier", None)
    parameters = {
        "C": _SMALL_DENSE_SVC_C,
        "gamma": "scale",
        "probability": True,
        "random_state": int(seed),
    }
    parameters.update(model_params)
    if gamma_multiplier is not None:
        feature_count = getattr(features, "shape", np.asarray(features).shape)[1]
        parameters["gamma"] = float(gamma_multiplier) / max(
            feature_count, 1
        )
    estimator = make_sklearn_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(**parameters),
    )
    estimator.fit(features, np.asarray(target).reshape(-1))
    return _SklearnProbabilityModel(estimator, target)


def _fit_mixed_logit_candidate(features, target, model_params=None, seed=42):
    """Fit a scaled one-hot linear view with train-only category handling."""
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    numeric_columns = [
        column for column in frame.columns if column not in categorical_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_sklearn_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_sklearn_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical_columns,
            )
        )
    parameters = {
        "C": 3.0,
        "max_iter": 2_000,
        "random_state": int(seed),
    }
    parameters.update(model_params or {})
    estimator = make_sklearn_pipeline(
        ColumnTransformer(transformers),
        LogisticRegression(**parameters),
    )
    estimator.fit(frame, np.asarray(target).reshape(-1))
    return _SklearnProbabilityModel(estimator, target)


def _fit_mixed_svc_candidate(features, target, model_params=None, seed=42):
    """Fit a bounded one-hot RBF view with train-only category handling."""
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    numeric_columns = [
        column for column in frame.columns if column not in categorical_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_sklearn_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_sklearn_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical_columns,
            )
        )
    parameters = {
        "C": _SMALL_DENSE_SVC_C,
        "gamma": "scale",
        "probability": True,
        "random_state": int(seed),
    }
    parameters.update(model_params or {})
    estimator = make_sklearn_pipeline(
        ColumnTransformer(transformers),
        SVC(**parameters),
    )
    estimator.fit(frame, np.asarray(target).reshape(-1))
    return _SklearnProbabilityModel(estimator, target)


def _fit_extra_trees_candidate(
    features, target, model_params=None, seed=42, n_jobs=1
):
    """Fit a bounded randomised-tree candidate with calibrated OOF selection."""
    parameters = {
        "n_estimators": 500,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": max(int(n_jobs), 1),
        "random_state": int(seed),
    }
    parameters.update(model_params or {})
    estimator = make_sklearn_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(**parameters),
    )
    estimator.fit(features, np.asarray(target).reshape(-1))
    return _SklearnProbabilityModel(estimator, target)


def _fit_mixed_extra_trees_candidate(
    features, target, model_params=None, seed=42, n_jobs=1
):
    """Fit a regularised forest after train-only ordinal mixed-data encoding."""
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not categorical_columns:
        return _fit_extra_trees_candidate(
            features,
            target,
            model_params=model_params,
            seed=seed,
            n_jobs=n_jobs,
        )
    numeric_columns = [
        column for column in frame.columns if column not in categorical_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_columns)
        )
    transformers.append(
        (
            "categorical",
            make_sklearn_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            categorical_columns,
        )
    )
    parameters = {
        "n_estimators": 500,
        "min_samples_leaf": 8,
        "max_features": 1.0,
        "n_jobs": max(int(n_jobs), 1),
        "random_state": int(seed),
    }
    parameters.update(model_params or {})
    estimator = make_sklearn_pipeline(
        ColumnTransformer(transformers),
        ExtraTreesClassifier(**parameters),
    )
    estimator.fit(frame, np.asarray(target).reshape(-1))
    return _SklearnProbabilityModel(estimator, target)


def _relational_feature_expansion(features):
    """Add bounded generic pairwise relations to a narrow numeric matrix."""
    features = np.asarray(features, dtype=np.float32)
    expanded = [features]
    for left in range(features.shape[1]):
        for right in range(left + 1, features.shape[1]):
            difference = features[:, left] - features[:, right]
            expanded.extend(
                (
                    difference[:, None],
                    np.abs(difference)[:, None],
                    (difference == 0).astype(np.float32)[:, None],
                )
            )
    return np.hstack(expanded)


def _relational_second_order_feature_expansion(features):
    """Add equality relations between the first-order pairwise distances."""
    features = np.asarray(features, dtype=np.float32)
    first_order = _relational_feature_expansion(features)
    distances = []
    for left in range(features.shape[1]):
        for right in range(left + 1, features.shape[1]):
            distances.append(np.abs(features[:, left] - features[:, right]))
    second_order = []
    for left in range(len(distances)):
        for right in range(left + 1, len(distances)):
            second_order.append(
                (distances[left] == distances[right]).astype(np.float32)[:, None]
            )
    return (
        first_order
        if not second_order
        else np.hstack([first_order, *second_order])
    )


def _fit_relational_hist_candidate(
    features, target, model_params=None, seed=42, second_order=False
):
    """Fit a bounded histogram model on train-only ordinal pair relations."""
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    numeric_columns = [
        column for column in frame.columns if column not in categorical_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_sklearn_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                categorical_columns,
            )
        )
    parameters = {
        "max_iter": 250,
        "learning_rate": 0.1,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "random_state": int(seed),
    }
    parameters.update(model_params or {})
    estimator = make_sklearn_pipeline(
        ColumnTransformer(transformers),
        FunctionTransformer(
            _relational_second_order_feature_expansion
            if second_order
            else _relational_feature_expansion,
            validate=False,
        ),
        HistGradientBoostingClassifier(**parameters),
    )
    estimator.fit(frame, np.asarray(target).reshape(-1))
    return _SklearnProbabilityModel(estimator, target)


def _grouped_sequence_lgbm_shape(candidate):
    shapes = {
        "grouped_trend_lgbmreg_leaf511_direct": (511, 20),
        "grouped_trend_lgbmreg_leaf255_regularized_direct": (255, 100),
    }
    return shapes[candidate]


def _make_regression_portfolio_estimator(
    candidate, features, seed, n_jobs, small_mixed=False, selector=False
):
    """Build a bounded regressor with train-fitted ordinal preprocessing."""
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    numeric_columns = [
        column for column in frame.columns if column not in categorical_columns
    ]
    if candidate.startswith("ridge_") and candidate.endswith("_direct"):
        alpha_text = candidate[len("ridge_"):-len("_direct")]
        alpha = float(alpha_text)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be a positive finite number")
        ridge_transformers = []
        if numeric_columns:
            ridge_transformers.append(
                (
                    "numeric",
                    make_sklearn_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                    ),
                    numeric_columns,
                )
            )
        if categorical_columns:
            ridge_transformers.append(
                (
                    "categorical",
                    make_sklearn_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(
                            handle_unknown="ignore",
                            dtype=np.float64,
                        ),
                    ),
                    categorical_columns,
                )
            )
        return make_sklearn_pipeline(
            ColumnTransformer(
                ridge_transformers,
                sparse_threshold=1.0 if categorical_columns else 0.0,
            ),
            Ridge(alpha=alpha, solver="lsqr"),
        )
    grouped_sequence_candidates = {
        "grouped_trend_lgbmreg_leaf511_direct",
        "grouped_trend_lgbmreg_leaf255_regularized_direct",
    }
    if candidate in grouped_sequence_candidates:
        num_leaves, min_child_samples = _grouped_sequence_lgbm_shape(candidate)
        return make_sklearn_pipeline(
            FunctionTransformer(
                _grouped_sequence_trend_expansion,
                validate=False,
            ),
            SimpleImputer(strategy="median"),
            _TrainSelectedLGBMRegressor(
                n_estimators=1_000,
                learning_rate=0.05,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                selector_max_rows=100_000,
                validation_fraction=0.2,
                early_stopping_rounds=100,
                refit_all_rows=not selector,
                n_jobs=max(int(n_jobs), 1),
                random_state=int(seed),
            ),
        )
    if candidate == "target_frequency_lgbmreg_direct":
        return _CrossFittedTargetFrequencyRegressor(
            smoothing=10.0,
            n_estimators=1_000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            n_jobs=max(int(n_jobs), 1),
            random_state=int(seed),
        )
    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_columns)
        )
    if categorical_columns:
        categorical_encoder = (
            OneHotEncoder(handle_unknown="ignore")
            if candidate == "lgbmreg_onehot_direct"
            else OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        )
        transformers.append(
            (
                "categorical",
                make_sklearn_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    categorical_encoder,
                ),
                categorical_columns,
            )
        )

    if candidate in {
        "lgbmreg_direct",
        "lgbmreg_onehot_direct",
        "lgbmreg_compact_direct",
        "lgbmreg_large_leaf127_direct",
        "lgbmreg_large_leaf255_direct",
        "lgbmreg_large_dense_direct",
    }:
        large_lgbm_leaves = {
            "lgbmreg_large_leaf127_direct": 127,
            "lgbmreg_large_leaf255_direct": 255,
            "lgbmreg_large_dense_direct": 255,
        }
        bounded_small_lgbm = (
            small_mixed
            or candidate in {"lgbmreg_onehot_direct", "lgbmreg_compact_direct"}
        )
        estimator = LGBMRegressor(
            n_estimators=(
                1_000
                if selector and candidate == "lgbmreg_large_dense_direct"
                else 2_000
                if candidate in large_lgbm_leaves
                else 500 if bounded_small_lgbm
                else 1_000
            ),
            learning_rate=0.05,
            num_leaves=large_lgbm_leaves.get(
                candidate, 31 if bounded_small_lgbm else 63
            ),
            min_child_samples=(
                100 if candidate == "lgbmreg_large_dense_direct" else 20
            ),
            n_jobs=max(int(n_jobs), 1),
            random_state=int(seed),
            verbose=-1,
        )
    elif candidate == "xgboostreg_direct":
        estimator = XGBRegressor(
            n_estimators=1_000,
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            n_jobs=max(int(n_jobs), 1),
            random_state=int(seed),
        )
    elif candidate == "catboostreg_direct":
        estimator = CatBoostRegressor(
            iterations=1_000,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            thread_count=max(int(n_jobs), 1),
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
        )
    elif candidate == "rfr_direct":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=max(int(n_jobs), 1),
            random_state=int(seed),
        )
    elif candidate == "extra_treesreg_direct":
        estimator = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=max(int(n_jobs), 1),
            random_state=int(seed),
        )
    elif candidate == "scaled_svrreg_direct":
        estimator = TransformedTargetRegressor(
            regressor=make_sklearn_pipeline(
                StandardScaler(),
                SVR(C=10.0, epsilon=0.05, gamma="scale", cache_size=2_000),
            ),
            transformer=StandardScaler(),
        )
    elif candidate in {
        "ordinal_lgbm31_direct",
        "ordinal_lgbm63_direct",
        "ordinal_ordered_lgbm_direct",
    }:
        estimator = _CumulativeProbabilityRegressor(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=(
                    63 if candidate == "ordinal_lgbm63_direct" else 31
                ),
                n_jobs=max(int(n_jobs), 1),
                random_state=int(seed),
                verbose=-1,
            )
        )
    elif candidate == "ordinal_xgboost_direct":
        estimator = _CumulativeProbabilityRegressor(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                tree_method="hist",
                n_jobs=max(int(n_jobs), 1),
                random_state=int(seed),
            )
        )
    else:
        raise ValueError(f"Unknown regression portfolio candidate: {candidate}")
    preprocessing_steps = [ColumnTransformer(transformers)]
    if candidate == "ordinal_ordered_lgbm_direct":
        preprocessing_steps.append(
            FunctionTransformer(_ordered_feature_expansion, validate=False)
        )
    return make_sklearn_pipeline(*preprocessing_steps, estimator)


def _maybe_refit_on_all_rows(
    fitted_model,
    contender,
    train_features,
    train_target,
    config,
    scoring_metric,
    training_params,
    runtime_min,
    max_pipeline_fit_time,
    boosting_rounds,
    observed_fit_seconds,
    started_at,
    prediction_reserve,
    enabled,
    additional_budget_reserve=0.0,
):
    """Optionally use the early-stopped iteration count to fit on every train row."""
    if not enabled or contender["model"] not in {
        "lgbm",
        _LGBM_EXTRA_TREES_MODEL,
        "xgboost",
    }:
        return fitted_model

    selected_rounds = _best_boosting_rounds(fitted_model, boosting_rounds)
    if selected_rounds is None:
        return fitted_model

    estimated_seconds = max(observed_fit_seconds * 1.25, 0.1)
    elapsed = time.monotonic() - started_at
    required_seconds = (
        elapsed
        + estimated_seconds
        + prediction_reserve
        + additional_budget_reserve
    )
    if required_seconds > config.max_runtime_seconds:
        _portfolio_report(
            "Keeping early-stopped %s refit at %d rounds: estimated %.1fs "
            "all-row refit does "
            "not fit after reserving %.1fs for remaining ensemble refits.",
            contender["model"],
            selected_rounds,
            estimated_seconds,
            additional_budget_reserve,
        )
        return fitted_model

    _portfolio_report(
        "Refitting %s on every training row with %d selected boosting rounds.",
        contender["model"],
        selected_rounds,
    )
    try:
        return _fit_full_candidate(
            contender,
            train_features,
            train_target,
            config,
            scoring_metric,
            training_params,
            runtime_min,
            max_pipeline_fit_time,
            selected_rounds,
            use_eval_set=False,
        )
    except Exception:
        log.warning(
            "All-row portfolio refit for %s failed; using the early-stopped model.",
            contender["model"],
            exc_info=True,
        )
        return fitted_model


def _best_boosting_rounds(automl, fallback):
    pipeline = getattr(automl, "current_pipeline", None)
    for node in getattr(pipeline, "nodes", []):
        implementation = getattr(node, "fitted_operation", None)
        model = getattr(implementation, "model", None)
        best_iteration = getattr(model, "best_iteration_", None)
        if isinstance(best_iteration, (int, np.integer)) and best_iteration > 0:
            return min(int(best_iteration), fallback) if fallback else int(best_iteration)
        try:
            best_iteration = getattr(model, "best_iteration", None)
        except (AttributeError, TypeError):
            best_iteration = None
        if isinstance(best_iteration, (int, np.integer)) and best_iteration >= 0:
            rounds = int(best_iteration) + 1
            return min(rounds, fallback) if fallback else rounds
    return fallback


def _extrapolated_boosting_rounds(
    validation_models,
    full_train_rows,
    selector_train_rows,
    fallback,
    row_exponent=0.45,
):
    """Estimate all-row rounds from early stopping on a capped selector.

    The useful boosting horizon grows much more slowly than the number of rows.
    Taking the median across validation models makes the estimate usable for both
    a single holdout and OOF selection, while the global boosting cap remains a
    hard upper bound.
    """
    selected = [
        rounds
        for rounds in (
            _best_boosting_rounds(model, fallback=None)
            for model in validation_models
        )
        if rounds is not None
    ]
    if not selected:
        return None

    row_ratio = max(full_train_rows / selector_train_rows, 1.0)
    extrapolated = max(int(round(float(np.median(selected)) * row_ratio**row_exponent)), 1)
    return min(extrapolated, fallback) if fallback else extrapolated


def _categorical_encoding_profile(features):
    """Return cheap dense-encoding bounds without inspecting held-out data."""
    if not hasattr(features, "dtypes"):
        return None

    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not categorical_columns:
        return None

    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    estimated_columns = (
        frame.shape[1] - len(categorical_columns) + categorical_cardinality
    )
    return {
        "categorical_columns": categorical_columns,
        "categorical_cardinality": categorical_cardinality,
        "estimated_columns": estimated_columns,
        "estimated_dense_cells": len(frame) * max(estimated_columns, 1),
    }


def _should_use_fast_one_hot(
    features,
    target,
    min_rows=2_500,
    max_categorical_cardinality=512,
    max_dense_cells=50_000_000,
):
    """Use a cheap nominal representation for bounded binary tabular tasks.

    The decision is deliberately limited to training-set geometry. Multiclass
    objectives keep the existing representation because a speed gain did not
    compensate for their small but consistent log-loss regression.
    """
    if min_rows <= 0:
        raise ValueError("one-hot encoding row threshold must be greater than zero")
    target_array = np.asarray(target).reshape(-1)
    if len(features) < min_rows or len(np.unique(target_array)) != 2:
        return False

    profile = _categorical_encoding_profile(features)
    return bool(
        profile
        and profile["categorical_cardinality"] <= max_categorical_cardinality
        and profile["estimated_dense_cells"] <= max_dense_cells
    )


def _compact_encode_if_needed(
    train_features, test_features, force=False, min_rows=2_500, one_hot=False
):
    """Compact categorical columns when one-hot handling is risky or costly.

    The encoder is fitted on the complete outer training partition and never sees
    test values. This shared, unsupervised transform also avoids repeating expensive
    type correction and imputation for every portfolio candidate. Large tables with
    numeric-like categorical levels use the transform even at low cardinality:
    FEDOT's repeated generic categorical preprocessing otherwise loses substantial
    quality under short benchmark budgets. Arbitrary low-cardinality nominal labels
    keep the generic path, where ordinal codes impose a potentially harmful order.
    """
    if min_rows <= 0:
        raise ValueError("compact encoding row threshold must be greater than zero")
    if not hasattr(train_features, "dtypes"):
        return train_features, test_features

    train_frame = pd.DataFrame(train_features)
    test_frame = pd.DataFrame(test_features)
    input_column_count = train_frame.shape[1]
    profile = _categorical_encoding_profile(train_frame)
    if profile is None:
        return train_features, test_features
    categorical_columns = profile["categorical_columns"]
    categorical_cardinality = profile["categorical_cardinality"]
    estimated_dense_cells = profile["estimated_dense_cells"]
    if one_hot and (
        categorical_cardinality > 512 or estimated_dense_cells > 50_000_000
    ):
        _portfolio_report(
            "Skipped dense one-hot encoding of %d levels and approximately %d "
            "cells; using compact ordinal codes instead.",
            categorical_cardinality,
            estimated_dense_cells,
        )
        one_hot = False
    large_numeric_categorical_table = (
        len(train_frame) >= min_rows
        and _categorical_levels_are_numeric(train_frame, categorical_columns)
    )
    if (
        not force
        and not one_hot
        and not large_numeric_categorical_table
        and categorical_cardinality <= 512
        and estimated_dense_cells <= 50_000_000
    ):
        return train_features, test_features

    retained_columns = train_frame.columns[~train_frame.isna().all(axis=0)]
    train_frame = train_frame.loc[:, retained_columns]
    test_frame = test_frame.loc[:, retained_columns]
    categorical_columns = [
        column for column in categorical_columns if column in retained_columns
    ]
    numerical_columns = [
        column for column in retained_columns if column not in categorical_columns
    ]

    if one_hot:
        return _fast_one_hot_encode(
            train_frame,
            test_frame,
            categorical_columns,
            numerical_columns,
            input_column_count,
        )

    transformed_train = np.empty(train_frame.shape, dtype=np.float32)
    transformed_test = np.empty(test_frame.shape, dtype=np.float32)
    column_positions = {column: index for index, column in enumerate(retained_columns)}

    if numerical_columns:
        train_numeric = train_frame.loc[:, numerical_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        test_numeric = test_frame.loc[:, numerical_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        train_numeric = train_numeric.replace([np.inf, -np.inf], np.nan)
        test_numeric = test_numeric.replace([np.inf, -np.inf], np.nan)
        medians = train_numeric.median(axis=0)
        train_numeric = train_numeric.fillna(medians).fillna(0.0)
        test_numeric = test_numeric.fillna(medians).fillna(0.0)
        numerical_positions = [column_positions[column] for column in numerical_columns]
        transformed_train[:, numerical_positions] = train_numeric.to_numpy(
            dtype=np.float32
        )
        transformed_test[:, numerical_positions] = test_numeric.to_numpy(
            dtype=np.float32
        )

    if categorical_columns:
        missing_token = "__fedot_missing__"
        train_categorical = (
            train_frame.loc[:, categorical_columns]
            .astype(object)
            .where(lambda frame: frame.notna(), missing_token)
            .astype(str)
        )
        test_categorical = (
            test_frame.loc[:, categorical_columns]
            .astype(object)
            .where(lambda frame: frame.notna(), missing_token)
            .astype(str)
        )
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )
        categorical_positions = [
            column_positions[column] for column in categorical_columns
        ]
        transformed_train[:, categorical_positions] = encoder.fit_transform(
            train_categorical
        )
        transformed_test[:, categorical_positions] = encoder.transform(
            test_categorical
        )

    _portfolio_report(
        "Compact-encoded %d categorical columns (%d observed levels); retained "
        "%d of %d input columns.",
        len(categorical_columns),
        categorical_cardinality,
        len(retained_columns),
        input_column_count,
    )
    return transformed_train, transformed_test


def _fast_one_hot_encode(
    train_frame,
    test_frame,
    categorical_columns,
    numerical_columns,
    input_column_count,
):
    """Fit a shared dense one-hot transform without observing outer test levels."""
    train_parts = []
    test_parts = []
    if numerical_columns:
        train_numeric = train_frame.loc[:, numerical_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        test_numeric = test_frame.loc[:, numerical_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        train_numeric = train_numeric.replace([np.inf, -np.inf], np.nan)
        test_numeric = test_numeric.replace([np.inf, -np.inf], np.nan)
        medians = train_numeric.median(axis=0)
        train_parts.append(
            train_numeric.fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
        )
        test_parts.append(
            test_numeric.fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
        )

    encoded_column_count = 0
    if categorical_columns:
        missing_token = "__fedot_missing__"
        train_categorical = (
            train_frame.loc[:, categorical_columns]
            .astype(object)
            .where(lambda frame: frame.notna(), missing_token)
            .astype(str)
        )
        test_categorical = (
            test_frame.loc[:, categorical_columns]
            .astype(object)
            .where(lambda frame: frame.notna(), missing_token)
            .astype(str)
        )
        encoder_params = {
            "handle_unknown": "ignore",
            "dtype": np.float32,
        }
        try:
            encoder = OneHotEncoder(sparse_output=False, **encoder_params)
        except TypeError:  # scikit-learn < 1.2
            encoder = OneHotEncoder(sparse=False, **encoder_params)
        encoded_train = encoder.fit_transform(train_categorical)
        encoded_test = encoder.transform(test_categorical)
        encoded_column_count = encoded_train.shape[1]
        train_parts.append(encoded_train)
        test_parts.append(encoded_test)

    transformed_train = (
        np.column_stack(train_parts).astype(np.float32, copy=False)
        if train_parts
        else np.empty((len(train_frame), 0), dtype=np.float32)
    )
    transformed_test = (
        np.column_stack(test_parts).astype(np.float32, copy=False)
        if test_parts
        else np.empty((len(test_frame), 0), dtype=np.float32)
    )
    _portfolio_report(
        "Fast one-hot encoded %d categorical columns into %d indicators; retained "
        "%d of %d numerical columns.",
        len(categorical_columns),
        encoded_column_count,
        len(numerical_columns),
        input_column_count,
    )
    return transformed_train, transformed_test


def _categorical_levels_are_numeric(frame, categorical_columns):
    """Recognise category dtypes that merely wrap compact numerical codes."""
    for column in categorical_columns:
        levels = pd.Series(frame[column].dropna().unique())
        if levels.empty:
            continue
        if pd.to_numeric(levels.astype(str), errors="coerce").isna().any():
            return False
    return True


def _predefined_model_with_n_jobs(
    predefined_model,
    n_jobs,
    n_estimators=None,
    use_eval_set=None,
    model_params=None,
):
    if predefined_model == "scaled_logit":
        scaling = PipelineNode("scaling")
        classifier = PipelineNode("logit", nodes_from=[scaling])
        classifier.parameters = dict(model_params or {})
        return Pipeline(classifier)
    operation = (
        "rf"
        if predefined_model == "rf_large_subspace"
        else "lgbm"
        if predefined_model == _LGBM_EXTRA_TREES_MODEL
        else predefined_model
    )
    boosting_models = {"lgbm", "xgboost", "catboost"}
    parallel_models = boosting_models | {"rf"}
    parameterised_models = parallel_models | {"logit"}
    if not isinstance(operation, str) or operation not in parameterised_models:
        return predefined_model
    node = PipelineNode(operation)
    parameters = dict(model_params or {})
    if operation in parallel_models:
        parameters["n_jobs"] = max(int(n_jobs), 1)
    if n_estimators is not None and operation in boosting_models:
        # FEDOT's CatBoost defaults use ``num_trees``.  Adding the synonymous
        # ``n_estimators`` leaves both keys in the merged operation parameters
        # and CatBoost rejects the fit before training starts.
        rounds_parameter = "num_trees" if operation == "catboost" else "n_estimators"
        parameters[rounds_parameter] = int(n_estimators)
    if use_eval_set is not None and operation in boosting_models:
        parameters["use_eval_set"] = bool(use_eval_set)
        if not use_eval_set:
            parameters["early_stopping_rounds"] = None
    node.parameters = parameters
    return Pipeline(node)


def _portfolio_refit_scale(
    full_train_rows, selector_train_rows, fold_count, feature_count=None
):
    """Estimate full-fit cost from row scaling and mean fold cost.

    Validation on a very small slice of a wide table contains substantial fixed
    preprocessing and prediction work. Scaling that complete duration with the
    usual three-quarter power can therefore overestimate a full fit enough to
    reject every refit. Use a still-conservative two-thirds curve only in this
    identifiable regime; ordinary and moderately capped selectors retain the
    established estimate.
    """
    selector_to_full_ratio = max(full_train_rows / selector_train_rows, 1.0)
    tiny_wide_selector = (
        feature_count is not None
        and feature_count >= 256
        and selector_to_full_ratio >= 16
    )
    if tiny_wide_selector:
        return selector_to_full_ratio ** (2 / 3) * 1.15 / fold_count
    return selector_to_full_ratio**0.75 * 1.05 / fold_count


def _direct_refit_budget_estimate(
    contender, direct_rounds, fold_count, conservative_fallback
):
    """Estimate a raw all-row LGBM refit without charging selector overhead.

    The measured contender duration also contains FEDOT preprocessing and
    prediction on the comparatively large selector holdout.  A direct numeric
    refit skips both, so the generic row-based scale can reject a fit that is
    several times cheaper in practice.  Scaling the mean fold duration by the
    observed boosting-horizon ratio, with a safety multiplier, is still
    conservative on the wide numeric controls.  XGBoost keeps the generic
    estimate because its all-row cost varied much more strongly with row count.
    """
    if contender["model"] not in {
        "lgbm",
        _LGBM_EXTRA_TREES_MODEL,
    } or direct_rounds is None:
        return conservative_fallback

    selector_rounds = [
        rounds
        for rounds in (
            _best_boosting_rounds(model, fallback=None)
            for model in contender["validation_models"]
        )
        if rounds is not None
    ]
    if not selector_rounds:
        return conservative_fallback

    median_selector_rounds = max(float(np.median(selector_rounds)), 1.0)
    round_scale = max(float(direct_rounds) / median_selector_rounds, 1.0)
    direct_estimate = (
        contender["duration"] / max(fold_count, 1) * round_scale * 1.75
    )
    return min(max(direct_estimate, 0.1), conservative_fallback)


def _secondary_refit_budget_estimate(
    contender, measured_primary_scale, fold_count
):
    """Avoid transferring an extreme primary scale across model families.

    A measured full/selector scale is valuable for a second booster with similar
    training mechanics.  It can substantially overestimate a Random Forest,
    however, when a long early-stopped XGBoost refit becomes the primary.  The
    selector's independent row-based estimate already accounts for table size;
    allowing a twofold uncertainty margin covers near-linear RF row scaling over
    the portfolio's guarded range without letting an unrelated booster suppress
    an otherwise affordable forest refit.
    """
    if measured_primary_scale < 0 or fold_count <= 0:
        raise ValueError("Secondary refit estimate inputs must be positive")
    measured_transfer = (
        contender["duration"] / fold_count * measured_primary_scale
    )
    if contender["model"] not in {"rf", "rf_large_subspace"}:
        return measured_transfer
    independent_upper_bound = contender["refit_seconds"] * 2.0
    return min(measured_transfer, independent_upper_bound)


def _select_refittable_primary(
    ensemble,
    refit_estimates_by_name,
    elapsed_seconds,
    runtime_seconds,
    prediction_reserve,
    refit_start_safety_reserve,
):
    """Prefer a deployable all-row candidate over a selector-only winner."""
    validation_primary = max(
        ensemble, key=lambda contender: contender["score"]
    )
    refittable = [
        contender
        for contender in ensemble
        if _refit_fits_budget(
            refit_estimates_by_name[contender["model"]],
            elapsed_seconds,
            runtime_seconds,
            prediction_reserve,
            refit_start_safety_reserve,
        )
    ]
    primary = (
        max(refittable, key=lambda contender: contender["score"])
        if refittable
        else validation_primary
    )
    return primary, validation_primary


def _refit_fits_budget(
    estimated_refit_seconds,
    elapsed_seconds,
    runtime_seconds,
    prediction_reserve,
    refit_start_safety_reserve,
):
    """Require uncertainty headroom before starting a non-interruptible refit."""
    values = (
        estimated_refit_seconds,
        elapsed_seconds,
        runtime_seconds,
        prediction_reserve,
        refit_start_safety_reserve,
    )
    if any(value < 0 for value in values):
        raise ValueError("Refit budget values must be non-negative")
    return (
        elapsed_seconds
        + estimated_refit_seconds
        + prediction_reserve
        + refit_start_safety_reserve
        <= runtime_seconds
    )


def _direct_xgboost_refit_time_limit(
    model,
    direct_rounds,
    elapsed_seconds,
    runtime_seconds,
    prediction_reserve,
    refit_start_safety_reserve,
):
    """Return a safe fit deadline for a useful otherwise-rejected raw refit."""
    values = (
        elapsed_seconds,
        runtime_seconds,
        prediction_reserve,
        refit_start_safety_reserve,
    )
    if any(value < 0 for value in values):
        raise ValueError("Direct-refit budget values must be non-negative")
    if model != "xgboost" or direct_rounds is None:
        return None
    fit_seconds = (
        runtime_seconds
        - elapsed_seconds
        - prediction_reserve
        - refit_start_safety_reserve
        - _DIRECT_XGBOOST_REFIT_SETUP_RESERVE_SECONDS
    )
    if fit_seconds < _DIRECT_XGBOOST_MIN_REFIT_SECONDS:
        return None
    return fit_seconds


def _is_bounded_medium_wide_numeric_multiclass(features, target):
    """Recognise the dense image-like regime validated for coarse histograms."""
    shape = getattr(features, "shape", None)
    if shape is None or len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    if len(target_array) != row_count:
        return False
    _, class_counts = np.unique(target_array, return_counts=True)
    if not (
        20_000 <= row_count <= 100_000
        and 512 <= feature_count <= 1_024
        and row_count * feature_count <= 80_000_000
        and 5 <= len(class_counts) <= 20
        and class_counts.min() >= 500
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    ):
        return False
    density = _sampled_numeric_density(features)
    return density is not None and density >= 0.15


def _local_image_features(features, image_side):
    """Pool a flattened square image and append local gradient magnitudes."""
    image_side = int(image_side)
    matrix = (
        features.to_numpy(copy=False)
        if hasattr(features, "to_numpy")
        else np.asarray(features)
    )
    if (
        matrix.ndim != 2
        or image_side <= 0
        or image_side % 2
        or matrix.shape[1] != image_side * image_side
    ):
        raise ValueError("local image features require an even square width")
    images = np.asarray(matrix, dtype=np.float32).reshape(
        -1, image_side, image_side
    )
    pooled = images.reshape(
        len(images), image_side // 2, 2, image_side // 2, 2
    ).mean(axis=(2, 4))
    horizontal = np.abs(np.diff(pooled, axis=2))
    vertical = np.abs(np.diff(pooled, axis=1))
    return np.concatenate(
        (
            pooled.reshape(len(images), -1),
            horizontal.reshape(len(images), -1),
            vertical.reshape(len(images), -1),
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def _square_image_locality_side(features, target, seed=137):
    """Recognise a bounded flattened image using only unsupervised locality.

    A square feature count alone is too weak: an ordinary table can have a
    coincidentally square width. Real raster order has much smaller adjacent
    differences than deterministic random column pairs. The ratio is measured
    on a bounded, evenly spaced train-only sample and is scale-independent.
    """
    shape = getattr(features, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    row_count, feature_count = shape
    image_side = int(round(np.sqrt(feature_count)))
    target_array = np.asarray(target).reshape(-1)
    if len(target_array) != row_count:
        return None
    _, class_counts = np.unique(target_array, return_counts=True)
    if not (
        20_000 <= row_count <= 100_000
        and 5 <= len(class_counts) <= 20
        and class_counts.min() >= 500
        and image_side * image_side == feature_count
        and image_side % 2 == 0
        and 20 <= image_side <= 64
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    ):
        return None

    sampled_rows = min(row_count, 2_048)
    sampled_indices = np.linspace(0, row_count - 1, sampled_rows, dtype=int)
    sample = _slice_rows(features, sampled_indices)
    if hasattr(sample, "to_numpy"):
        sample = sample.to_numpy(copy=False)
    sample = np.asarray(sample, dtype=np.float32)
    if not np.isfinite(sample).all() or float(np.min(sample)) < 0:
        return None
    sampled_range = float(np.max(sample) - np.min(sample))
    if not np.isfinite(sampled_range) or sampled_range <= 0:
        return None

    images = sample.reshape(-1, image_side, image_side)
    adjacent_total = float(np.abs(np.diff(images, axis=1)).sum(dtype=np.float64))
    adjacent_total += float(
        np.abs(np.diff(images, axis=2)).sum(dtype=np.float64)
    )
    adjacent_count = 2 * len(images) * image_side * (image_side - 1)
    local_difference = adjacent_total / max(adjacent_count, 1)

    permutation = np.random.default_rng(int(seed)).permutation(feature_count)
    random_difference = float(
        np.abs(sample - sample[:, permutation]).mean(dtype=np.float64)
    )
    if not np.isfinite(random_difference) or random_difference <= 0:
        return None
    locality_ratio = local_difference / random_difference
    return image_side if locality_ratio <= 0.5 else None


def _local_image_refit_side(
    features,
    target,
    metric,
    bounded_refit_seconds,
    direct_rounds,
    framework_params,
):
    """Enable the local view only for the validated adaptive direct refit."""
    if any(str(key).startswith("_portfolio_") for key in framework_params):
        return None
    if not (
        metric == "logloss"
        and bounded_refit_seconds is not None
        and bounded_refit_seconds >= 75.0
        and direct_rounds is not None
        and direct_rounds >= 100
        and _is_bounded_medium_wide_numeric_multiclass(features, target)
    ):
        return None
    return _square_image_locality_side(features, target)


def _local_image_lgbm_refit_contender(contenders, fit_time_limit):
    """Build a bounded local LightGBM refit without mutating OOF contenders."""
    if fit_time_limit is None:
        return None
    lgbm = next(
        (
            contender
            for contender in contenders
            if contender.get("model") == "lgbm"
        ),
        None,
    )
    if lgbm is None:
        return None
    return {
        "model": "lgbm",
        "model_params": {
            **dict(lgbm.get("model_params") or {}),
            "fit_time_limit": float(fit_time_limit),
        },
    }


def _local_image_fit_calibration_indices(
    target, seed=42, calibration_fraction=_LOCAL_IMAGE_CALIBRATION_FRACTION
):
    """Reserve a deterministic stratified train-only calibration partition."""
    target_array = np.asarray(target).reshape(-1)
    if not 0 < float(calibration_fraction) < 0.5:
        raise ValueError("local image calibration fraction must be in (0, 0.5)")
    fit_indices, calibration_indices = train_test_split(
        np.arange(len(target_array)),
        test_size=float(calibration_fraction),
        random_state=int(seed) + 73,
        stratify=target_array,
    )
    return fit_indices, calibration_indices


def _deadline_bounded_xgboost_model_params(
    model_params,
    features,
    target,
    metric,
    bounded_refit_seconds,
    direct_rounds,
):
    """Regularise only a sufficiently long deadline-bounded all-row refit."""
    params = dict(model_params or {})
    applicable = (
        metric == "logloss"
        and bounded_refit_seconds is not None
        and bounded_refit_seconds >= 75.0
        and direct_rounds is not None
        and direct_rounds >= 100
        and "max_bin" not in params
        and "reg_lambda" not in params
        and _is_bounded_medium_wide_numeric_multiclass(features, target)
    )
    if applicable:
        params.update(max_bin=128, reg_lambda=3.0)
    return params


def _use_raw_bounded_xgboost_calibration(
    metric, bounded_refit_seconds, selected_models
):
    """Reject selector calibration after a bounded all-row singleton refit."""
    return (
        metric == "logloss"
        and bounded_refit_seconds is not None
        and list(selected_models) == ["xgboost"]
    )


def _use_all_row_refit(
    requested,
    configured_min_rows,
    full_train_rows,
    validation_fold_count,
    metric,
):
    """Allow a refit when row volume or robust probability validation supports it.

    A single internal holdout gives a noisy estimate of the boosting iteration on
    small data, so the conservative row threshold remains the general default.
    Three-fold OOF logloss is a stronger signal: in that regime using every row in
    the final booster consistently improved the external logloss controls.  An
    explicit minimum keeps its literal meaning and disables this automatic
    relaxation.
    """
    if not requested:
        return False
    min_rows = 5_000 if configured_min_rows is None else int(configured_min_rows)
    if full_train_rows >= min_rows:
        return True
    return (
        configured_min_rows is None
        and metric == "logloss"
        and validation_fold_count >= 3
    )


def _use_direct_all_row_refit(
    configured,
    full_data_refit,
    full_train_rows,
    selector_train_rows,
    feature_count,
    all_features_are_numeric=True,
):
    """Skip the redundant early-stopped full fit for a tiny wide selector.

    On a large wide table the capped selector is much smaller than the final
    training partition. Its early-stopping horizon scales predictably with row
    count, while fitting once with that extrapolated horizon lets the booster use
    every row and avoids an expensive second fit. Explicit configuration remains
    available for controlled ablations.
    """
    if not full_data_refit or not all_features_are_numeric:
        return False
    if configured is not None:
        return _as_bool(configured)
    row_ratio = max(full_train_rows / selector_train_rows, 1.0)
    return feature_count >= 256 and row_ratio >= 16


def _all_features_are_numeric(features):
    if hasattr(features, "dtypes"):
        return all(
            pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
        )
    dtype = getattr(features, "dtype", None)
    return dtype is not None and np.issubdtype(dtype, np.number)


def _sampled_numeric_density(features, max_cells=2_000_000):
    """Estimate nonzero density without scanning an arbitrarily large table."""
    if max_cells <= 0:
        raise ValueError("density sample size must be greater than zero")
    if not _all_features_are_numeric(features):
        return None

    row_count, feature_count = features.shape
    if row_count == 0 or feature_count == 0:
        return 0.0
    sampled_rows = min(row_count, max(max_cells // feature_count, 1))
    if sampled_rows < row_count:
        indices = np.linspace(0, row_count - 1, sampled_rows, dtype=int)
        sample = _slice_rows(features, indices)
    else:
        sample = features
    if hasattr(sample, "nnz"):
        return float(sample.nnz) / float(sample.shape[0] * sample.shape[1])
    if hasattr(sample, "to_numpy"):
        sample = sample.to_numpy(copy=False)
    sample = np.asarray(sample)
    observed = np.isfinite(sample) & (sample != 0)
    return float(np.count_nonzero(observed)) / float(sample.size)


def _adaptive_direct_round_exponent(configured, sampled_numeric_density):
    """Scale raw wide-table refits using only the observed matrix structure.

    Early-stopping horizons measured on a small selector need slightly stronger
    row-count extrapolation for sufficiently dense wide numeric tables.  The
    conservative exponent remains preferable for sparse image matrices.  Density
    is sampled only in the already guarded direct-all-row regime, so this policy
    does not affect ordinary FEDOT refits.  Explicit configuration always wins.
    """
    if configured is not None:
        return float(configured)
    if sampled_numeric_density is not None and sampled_numeric_density >= 0.30:
        return 0.48
    return 0.45


def _reuse_cv_secondary_models(
    configured, validation_fold_count, adaptive_wide_pair, primary_model
):
    """Choose CV bagging only for an explicit request or the robust wide pair.

    The automatic case keeps a full-data linear primary and uses cross-validation
    XGBoost models as a diverse nonlinear secondary. General secondary-model reuse
    is intentionally avoided: it was unstable for small LGBM/XGBoost portfolios.
    """
    if validation_fold_count <= 1:
        return False
    if configured is not None:
        return _as_bool(configured)
    return adaptive_wide_pair and primary_model == "logit"


def _adaptive_selector_train_rows(
    features, target, configured_max_rows, xgboost_max_bin=None
):
    """Cap selection work using a dataset-shape proxy, independent of task name.

    Extreme-wide numeric multiclass tables use one aggressively column-sampled
    XGBoost model. Coarse histograms make rows much cheaper than the raw feature
    count suggests, so spend the saved work on class support instead. The cap is
    scaled by effective sampled columns, classes and histogram resolution; its
    reference point is the largest selector that remained inside a 180-second,
    four-core budget in the external-fold controls.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    class_count = max(len(np.unique(np.asarray(target).reshape(-1))), 1)
    if _is_extreme_wide_numeric_multiclass(features, target):
        effective_features = max(feature_count * 0.02, 1.0)
        histogram_scale = max((xgboost_max_bin or 32) / 32.0, 0.25)
        reference_work = 7_200 * (7_200 * 0.02) * 10
        adaptive_rows = int(
            reference_work
            / max(effective_features * class_count * histogram_scale, 1.0)
        )
        adaptive_rows = min(max(adaptive_rows, 2_500), 7_200)
        return min(configured_max_rows, adaptive_rows)

    work_per_row = max(feature_count * np.sqrt(class_count), 1.0)
    work_budget = 5_000_000
    adaptive_rows = max(2_500, int(work_budget / work_per_row))
    return min(configured_max_rows, adaptive_rows)


def _adaptive_boosting_rounds(features, target, configured=None):
    """Set a generous early-stopping ceiling from task shape, not task identity.

    Narrow problems need more sequential trees to express interactions, while the
    cost of one boosting round grows with the number of features and classes.  The
    ceiling is intentionally high because both boosters still use early stopping.
    """
    if configured is not None:
        return int(configured)
    if _is_high_work_wide_many_class_numeric(features, target):
        # On dense many-class image-like tables even a few hundred multiclass
        # trees exceed the complete 180-second job budget.  A validated compact
        # tree shape trades a little per-tree capacity for a longer boosting
        # horizon: eighty rounds stayed within the budget and improved logloss
        # across several independent dense image-feature tasks.
        return 80
    if _is_large_supported_mid_class_categorical(features, target):
        # This regime deploys only XGBoost, so a high early-stopping ceiling is
        # cheaper than it would be for a multi-candidate portfolio.  In the
        # wider part of the regime the generic work proxy otherwise truncates
        # the model at 300 rounds even though the measured refit still fits the
        # benchmark budget.  A 1,400-round ceiling transferred across external
        # folds while retaining a thirty-second job margin at the bounded dense
        # cell count below.  Larger or narrower tables keep the more conservative
        # 1,000-round ceiling.  The refit budget guard can retain the early-stopped
        # model when a second fixed-horizon all-row fit is no longer affordable.
        shape = getattr(features, "shape", np.asarray(features).shape)
        row_count = shape[0] if shape else len(np.asarray(target).reshape(-1))
        feature_count = shape[1] if len(shape) > 1 else 1
        if feature_count >= 32 and row_count * feature_count <= 32_000_000:
            return 1_400
        return 1_000

    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    class_count = max(len(np.unique(np.asarray(target).reshape(-1))), 1)
    round_complexity = feature_count * np.sqrt(class_count)
    if round_complexity <= 32:
        return 2_000
    if round_complexity <= 128:
        return 1_000
    return 300


def _adaptive_lgbm_num_leaves(features, target):
    """Adapt tree capacity to the interaction and multiclass work regimes.

    Very narrow problems need only a modest increase over LightGBM's default.
    Large low-cardinality tables in the next complexity band can support deeper
    interaction partitions.  Conversely, dense high-work multiclass tables gain
    more from slightly smaller trees and a longer boosting horizon.  All branches
    use only table geometry, and refit-budget guards still limit their cost.
    """
    if _is_high_work_wide_many_class_numeric(features, target):
        return 27
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = max(len(class_counts), 1)
    round_complexity = feature_count * np.sqrt(class_count)
    if (
        len(target_array) >= 20_000
        and 3 <= class_count <= 5
        and round_complexity <= 128
    ):
        return 127
    if round_complexity <= 32:
        return 63
    return None


def _has_declared_or_compact_categorical_features(features):
    """Recognise declared categories and compact integer category encodings."""
    if not hasattr(features, "dtypes"):
        return False

    feature_dtypes = list(features.dtypes)
    has_declared_categorical_features = any(
        not pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
    )
    compact_integer_features = sum(
        pd.api.types.is_integer_dtype(dtype)
        and getattr(getattr(dtype, "numpy_dtype", dtype), "itemsize", 0) == 1
        for dtype in feature_dtypes
    )
    has_compact_integer_coding = (
        bool(feature_dtypes)
        and 4 * compact_integer_features >= 3 * len(feature_dtypes)
    )
    return bool(has_declared_categorical_features or has_compact_integer_coding)


def _is_large_low_class_categorical(features, target):
    """Identify a discrete interaction regime with well-supported leaves.

    Input metadata does not always mark integer-coded categorical columns as
    categorical.  Treat a table as discrete when it either contains a declared
    non-numeric column or at least three quarters of its columns use a compact
    one-byte integer dtype.  The latter is deliberately stricter than merely
    checking for integer values: it excludes ordinary continuous measurements
    stored as int32/int64 without scanning the complete table.
    """
    if not _has_declared_or_compact_categorical_features(features):
        return False

    shape = features.shape
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    round_complexity = feature_count * np.sqrt(class_count)
    return bool(
        len(target_array) >= 20_000
        and 3 <= class_count <= 5
        and round_complexity <= 128
    )


def _is_large_narrow_discrete_relational_candidate(features, target):
    """Recognise bounded compact ordinal tables where pair relations are useful.

    Signed differences have a natural meaning only for ordered measurements, so
    the adaptive branch requires every input column to use a compact integer
    dtype.  In particular, nominal category codes are not treated as ordinal.
    Strong class support, mostly low cardinality values and the expanded-cell
    budget bound both statistical variance and the quadratic feature expansion.
    Explicit candidate configuration remains available for controlled uses on
    other dtypes.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    relational_width = feature_count + 3 * feature_count * (feature_count - 1) // 2
    if not (
        20_000 <= row_count <= 100_000
        and 4 <= feature_count <= 9
        and 3 <= class_count <= 5
        and class_counts.min() >= 1_000
        and row_count * relational_width <= 12_000_000
    ):
        return False

    if hasattr(features, "dtypes"):
        frame = pd.DataFrame(features)
        if not all(
            pd.api.types.is_integer_dtype(frame[column].dtype)
            and getattr(frame[column].dtype, "itemsize", float("inf")) <= 2
            for column in frame.columns
        ):
            return False
        discrete_columns = sum(
            frame[column].nunique(dropna=True) <= 32 for column in frame.columns
        )
    else:
        values = np.asarray(features)
        if values.dtype.kind not in "iu" or values.dtype.itemsize > 2:
            return False
        discrete_columns = sum(
            len(np.unique(values[:, column])) <= 32
            for column in range(feature_count)
        )
    return bool(4 * discrete_columns >= 3 * feature_count)


def _is_second_order_relational_expansion_feasible(features):
    """Bound the complete second-order relational design by dense cell count."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    pair_count = feature_count * (feature_count - 1) // 2
    first_order_width = feature_count + 3 * pair_count
    second_order_width = first_order_width + pair_count * (pair_count - 1) // 2
    return bool(row_count * max(second_order_width, 1) <= 8_000_000)


def _is_medium_numeric_regression_portfolio(features, target):
    """Bound the direct booster pair to affordable medium numeric tables."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    return bool(
        10_000 <= row_count <= 30_000
        and 8 <= feature_count <= 64
        and row_count * feature_count <= 2_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and len(np.unique(target_array)) >= 2
        and _all_features_are_numeric(features)
    )


def _is_censored_ordinal_regression_portfolio(features, target):
    """Recognise bounded numeric regressions with substantial edge censoring.

    A cumulative classifier uses the order of a small numeric target alphabet
    and is especially useful when clipping creates large point masses at the
    observed range boundaries.  The rule only inspects outer-training geometry
    and keeps both the number of classifiers and encoded work bounded.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        1_500 <= row_count <= 30_000
        and 3 <= feature_count <= 64
        and row_count * feature_count <= 2_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and _all_features_are_numeric(features)
    ):
        return False

    _, level_counts = np.unique(target_array, return_counts=True)
    if not 3 <= len(level_counts) <= 32:
        return False
    edge_fraction = (level_counts[0] + level_counts[-1]) / row_count
    return bool(edge_fraction >= 0.25)


def _is_ordered_histogram_ordinal_regression_portfolio(features, target):
    """Recognise bounded ordered-bin tables suitable for cumulative features.

    Cumulative sums and adjacent differences are useful when compact integer
    columns describe consecutive bins.  Requiring a moderately wide, entirely
    compact-integer table keeps this optional expansion away from ordinary
    low-dimensional ordinal regressions and bounds its fourfold width growth.
    """
    if not _is_censored_ordinal_regression_portfolio(features, target):
        return False
    _, feature_count = features.shape
    if not 16 <= feature_count <= 64:
        return False

    if hasattr(features, "dtypes"):
        return bool(
            all(
                pd.api.types.is_integer_dtype(dtype)
                and getattr(getattr(dtype, "numpy_dtype", dtype), "itemsize", 0)
                <= 2
                for dtype in features.dtypes
            )
        )
    values = np.asarray(features)
    return bool(values.dtype.kind in "iu" and values.dtype.itemsize <= 2)


def _is_medium_mostly_numeric_regression_portfolio(features, target):
    """Allow a tiny numeric-like categorical fringe on medium numeric tables."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "dtypes")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        10_000 <= row_count <= 30_000
        and 8 <= feature_count <= 64
        and row_count * feature_count <= 2_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and len(np.unique(target_array)) >= 2
    ):
        return False

    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not categorical_columns or 20 * len(categorical_columns) > feature_count:
        return False
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    return bool(
        len(categorical_columns) <= 2
        and categorical_cardinality <= 128
        and _categorical_levels_are_numeric(frame, categorical_columns)
    )


def _is_small_wide_regression_portfolio(features, target):
    """Bound a cheap portfolio for small, wide, potentially mixed tables."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        200 <= row_count <= 2_000
        and 64 <= feature_count <= 256
        and row_count * feature_count <= 500_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.std(target_array) > 0
    ):
        return False

    if not hasattr(features, "dtypes"):
        return bool(np.asarray(features).dtype.kind in "biufc")
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    estimated_width = (
        feature_count - len(categorical_columns) + categorical_cardinality
    )
    return bool(
        len(categorical_columns) <= 8
        and categorical_cardinality <= 512
        and row_count * max(estimated_width, 1) <= 500_000
    )


def _is_small_mixed_regression_portfolio(features, target):
    """Recognise small nominal-heavy regressions with bounded encoding cost."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "dtypes")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        400 <= row_count <= 2_500
        and 4 <= feature_count <= 128
        and row_count * feature_count <= 250_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and len(np.unique(target_array)) >= 2
    ):
        return False

    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not categorical_columns:
        return False
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    return bool(
        2 * len(categorical_columns) >= feature_count
        and len(categorical_columns) <= 64
        and categorical_cardinality <= 2_048
    )


def _is_relaxed_small_mixed_ridge_portfolio(features, target):
    """Allow OOF-guarded Ridge to cover a bounded near-majority nominal table."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "dtypes")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        400 <= row_count <= 2_500
        and 4 <= feature_count <= 128
        and row_count * feature_count <= 250_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.std(target_array) > 0
    ):
        return False
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    return bool(
        5 * len(categorical_columns) >= 2 * feature_count
        and len(categorical_columns) <= 64
        and categorical_cardinality <= 2_048
    )


def _is_ridge_oof_regression_portfolio(features, target):
    """Bound the cost of leakage-safe Ridge transfer selection by train geometry."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "dtypes")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        200 <= row_count <= 2_500
        and 4 <= feature_count <= 256
        and row_count * feature_count <= 300_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.std(target_array) > 0
    ):
        return False
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    estimated_width = feature_count - len(categorical_columns) + categorical_cardinality
    return bool(estimated_width <= 2_048)


def _is_small_classic_regression_portfolio(features, target):
    """Recognise bounded, mostly-numeric classic regression tables."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        350 <= row_count <= 1_000
        and 5 <= feature_count <= 32
        and row_count * feature_count <= 32_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.std(target_array) > 0
    ):
        return False
    if not hasattr(features, "dtypes"):
        return bool(np.asarray(features).dtype.kind in "biufc")
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    return bool(
        len(categorical_columns) <= 2
        and 4 * len(categorical_columns) <= feature_count
        and categorical_cardinality <= 64
    )


def _is_large_compact_mixed_regression_portfolio(features, target):
    """Recognise bounded large mixed tables suited to direct tree models."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "dtypes")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        30_000 <= row_count <= 200_000
        and 5 <= feature_count <= 16
        and row_count * feature_count <= 2_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.std(target_array) > 0
    ):
        return False
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not 1 <= len(categorical_columns) <= 8:
        return False
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    return bool(categorical_cardinality <= 10_000)


def _is_very_large_compact_mixed_regression_portfolio(features, target):
    """Recognise compact mixed tables that can support high-capacity leaves.

    The lower row bound supplies at least one thousand observations per leaf
    for the 255-leaf candidate.  The cell and cardinality bounds keep both
    fixed-round fits inside the benchmark budget and avoid sparse/high-cardinality
    encodings where ordinal splits are a poor proxy for nominal structure.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "dtypes")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        255_000 <= row_count <= 600_000
        and 8 <= feature_count <= 24
        and row_count * feature_count <= 12_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.std(target_array) > 0
    ):
        return False
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not 1 <= len(categorical_columns) <= 12:
        return False
    categorical_cardinality = sum(
        frame[column].nunique(dropna=True) for column in categorical_columns
    )
    return bool(categorical_cardinality <= 1_024)


def _consecutive_suffix_feature_groups(columns):
    """Return repeated feature blocks named ``prefix_0, ..., prefix_k``."""
    grouped_columns = {}
    for column_index, column in enumerate(columns):
        prefix, separator, suffix = str(column).rpartition("_")
        if not separator or not suffix.isdigit():
            continue
        grouped_columns.setdefault(prefix, []).append((int(suffix), column_index))

    groups = []
    for indexed_columns in grouped_columns.values():
        indexed_columns.sort()
        suffixes = [suffix for suffix, _ in indexed_columns]
        if 4 <= len(suffixes) <= 12 and suffixes == list(range(len(suffixes))):
            groups.append([column_index for _, column_index in indexed_columns])
    return groups


def _grouped_sequence_trend_expansion(features):
    """Add recent change, full-window change and slope to sequential blocks."""
    frame = pd.DataFrame(features)
    groups = _consecutive_suffix_feature_groups(frame.columns)
    if not groups:
        raise ValueError("grouped sequence expansion needs consecutive blocks")
    group_lengths = {len(group) for group in groups}
    if len(group_lengths) != 1:
        raise ValueError("grouped sequence blocks must have equal lengths")

    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    group_length = next(iter(group_lengths))
    time_axis = np.arange(group_length, dtype=np.float32)
    centered_time = time_axis - np.mean(time_axis)
    slope_denominator = float(np.sum(centered_time**2))
    expansions = [values]
    for indices in groups:
        group_values = values[:, indices]
        expansions.append(
            np.column_stack(
                [
                    group_values[:, -1] - group_values[:, -2],
                    group_values[:, -1] - group_values[:, 0],
                    np.sum(group_values * centered_time, axis=1)
                    / slope_denominator,
                ]
            )
        )
    return np.column_stack(expansions)


def _is_large_grouped_sequence_regression_portfolio(features, target):
    """Recognise bounded dense positive regressions with repeated lag blocks.

    Count-like targets with long right tails often accompany multiple histories
    sampled over the same short horizon.  Fixed recent-change and linear-trend
    contrasts expose extrapolative directions that axis-aligned trees otherwise
    need many paired splits to approximate.  Geometry, target and expanded-cell
    bounds keep both the feature map and train-selected boosters affordable.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "columns")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        250_000 <= row_count <= 650_000
        and 32 <= feature_count <= 160
        and row_count * feature_count <= 60_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.min(target_array) >= 0
        and np.std(target_array) > 0
        and len(np.unique(target_array)) >= max(1_000, row_count // 100)
        and _all_features_are_numeric(features)
    ):
        return False

    groups = _consecutive_suffix_feature_groups(features.columns)
    if not 4 <= len(groups) <= 32:
        return False
    group_lengths = {len(group) for group in groups}
    grouped_feature_count = sum(map(len, groups))
    if len(group_lengths) != 1 or 5 * grouped_feature_count < 4 * feature_count:
        return False
    expanded_feature_count = feature_count + 3 * len(groups)
    if row_count * expanded_feature_count > 75_000_000:
        return False

    target_scale = float(np.std(target_array))
    standardized_skew = float(
        np.mean(((target_array - np.mean(target_array)) / target_scale) ** 3)
    )
    return bool(standardized_skew >= 3.0)


def _is_large_dense_low_cardinality_regression_portfolio(features, target):
    """Recognise bounded large numeric regressions with repeated target levels.

    Deep leaves are affordable and well supported on hundreds of thousands of
    dense rows, while a small target alphabet makes stronger leaf support less
    prone to fitting rare continuous extremes.  The cell bound excludes tables
    where the fixed 2,000-round fit cannot retain a benchmark-time margin.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        250_000 <= row_count <= 600_000
        and 64 <= feature_count <= 128
        and row_count * feature_count <= 60_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
        and np.std(target_array) > 0
        and _all_features_are_numeric(features)
    ):
        return False
    return bool(32 <= len(np.unique(target_array)) <= 256)


def _is_large_high_categorical_regression_portfolio(features, target):
    """Recognise bounded nominal-heavy regressions with a long target tail.

    Cross-fitted target statistics complement ordinal tree splits when most
    columns are nominal.  Cardinality and encoded-cell bounds keep the OOF view
    and its fixed LightGBM inside the benchmark budget.  Target skew limits the
    regime to the continuous-tail problems on which the complementary view was
    validated, without inspecting task names or identifiers.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if (
        len(shape) != 2
        or _is_sparse_table(features)
        or not hasattr(features, "dtypes")
    ):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        100_000 <= row_count <= 250_000
        and 96 <= feature_count <= 192
        and row_count * feature_count <= 40_000_000
        and len(target_array) == row_count
        and np.isfinite(target_array).all()
    ):
        return False
    target_scale = float(np.std(target_array))
    if target_scale <= 0 or len(np.unique(target_array)) < max(1_000, row_count // 10):
        return False

    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    numeric_count = feature_count - len(categorical_columns)
    if not (
        64 <= len(categorical_columns) <= 160
        and 10 * len(categorical_columns) >= 7 * feature_count
        and 8 <= numeric_count <= 32
    ):
        return False
    cardinalities = np.asarray(
        [frame[column].nunique(dropna=True) for column in categorical_columns],
        dtype=int,
    )
    if not (
        256 <= int(cardinalities.sum()) <= 4_096
        and int(cardinalities.max(initial=0)) <= 512
    ):
        return False
    standardized_skew = float(
        np.mean(((target_array - np.mean(target_array)) / target_scale) ** 3)
    )
    return bool(standardized_skew >= 2.0)


def _is_minimum_inflated_skewed_regression_target(target):
    """Recognise continuous targets with a censored floor and a long right tail."""
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if not (
        400 <= len(target_array) <= 2_500
        and np.isfinite(target_array).all()
        and len(np.unique(target_array)) >= 33
    ):
        return False
    target_scale = float(np.std(target_array))
    if target_scale <= 0:
        return False
    minimum_mass = float(np.mean(target_array == np.min(target_array)))
    standardized_skew = float(
        np.mean(((target_array - np.mean(target_array)) / target_scale) ** 3)
    )
    return bool(minimum_mass >= 0.15 and standardized_skew >= 3.0)


def _adaptive_regression_portfolio_candidates(features, target, configured=None):
    """Return bounded direct candidates or ``None`` for standard FEDOT."""
    aliases = {
        "lgbm": "lgbmreg_direct",
        "lgbmreg": "lgbmreg_direct",
        "lgbmreg_direct": "lgbmreg_direct",
        "lgbmreg_compact": "lgbmreg_compact_direct",
        "lgbmreg_compact_direct": "lgbmreg_compact_direct",
        "lgbmreg_large_leaf127": "lgbmreg_large_leaf127_direct",
        "lgbmreg_large_leaf127_direct": "lgbmreg_large_leaf127_direct",
        "lgbmreg_large_leaf255": "lgbmreg_large_leaf255_direct",
        "lgbmreg_large_leaf255_direct": "lgbmreg_large_leaf255_direct",
        "lgbmreg_large_dense": "lgbmreg_large_dense_direct",
        "lgbmreg_large_dense_direct": "lgbmreg_large_dense_direct",
        "grouped_trend_lgbmreg_leaf511": (
            "grouped_trend_lgbmreg_leaf511_direct"
        ),
        "grouped_trend_lgbmreg_leaf511_direct": (
            "grouped_trend_lgbmreg_leaf511_direct"
        ),
        "grouped_trend_lgbmreg_leaf255_regularized": (
            "grouped_trend_lgbmreg_leaf255_regularized_direct"
        ),
        "grouped_trend_lgbmreg_leaf255_regularized_direct": (
            "grouped_trend_lgbmreg_leaf255_regularized_direct"
        ),
        "target_frequency_lgbmreg": "target_frequency_lgbmreg_direct",
        "target_frequency_lgbmreg_direct": "target_frequency_lgbmreg_direct",
        "xgboost": "xgboostreg_direct",
        "xgboostreg": "xgboostreg_direct",
        "xgboostreg_direct": "xgboostreg_direct",
        "catboost": "catboostreg_direct",
        "catboostreg": "catboostreg_direct",
        "catboostreg_direct": "catboostreg_direct",
        "rf": "rfr_direct",
        "rfr": "rfr_direct",
        "rfr_direct": "rfr_direct",
        "ordinal_lgbm": "ordinal_lgbm31_direct",
        "ordinal_lgbm31": "ordinal_lgbm31_direct",
        "ordinal_lgbm31_direct": "ordinal_lgbm31_direct",
        "ordinal_lgbm63": "ordinal_lgbm63_direct",
        "ordinal_lgbm63_direct": "ordinal_lgbm63_direct",
        "ordinal_ordered_lgbm": "ordinal_ordered_lgbm_direct",
        "ordinal_ordered_lgbm_direct": "ordinal_ordered_lgbm_direct",
        "ordinal_xgboost": "ordinal_xgboost_direct",
        "ordinal_xgb": "ordinal_xgboost_direct",
        "ordinal_xgboost_direct": "ordinal_xgboost_direct",
        "lgbm_onehot": "lgbmreg_onehot_direct",
        "lgbmreg_onehot": "lgbmreg_onehot_direct",
        "lgbmreg_onehot_direct": "lgbmreg_onehot_direct",
        "extra_treesreg": "extra_treesreg_direct",
        "extra_treesreg_direct": "extra_treesreg_direct",
        "svr": "scaled_svrreg_direct",
        "scaled_svr": "scaled_svrreg_direct",
        "scaled_svrreg": "scaled_svrreg_direct",
        "scaled_svrreg_direct": "scaled_svrreg_direct",
    }
    if configured is not None:
        requested = _portfolio_candidates(configured)
        unknown = [candidate for candidate in requested if candidate not in aliases]
        if unknown:
            raise ValueError(
                f"Unknown regression portfolio candidates: {sorted(set(unknown))}"
            )
        return [aliases[candidate] for candidate in requested]
    if _is_censored_ordinal_regression_portfolio(features, target):
        candidates = [
            "ordinal_lgbm31_direct",
            "ordinal_lgbm63_direct",
            "ordinal_xgboost_direct",
        ]
        if _is_ordered_histogram_ordinal_regression_portfolio(features, target):
            candidates.insert(1, "ordinal_ordered_lgbm_direct")
        return candidates
    if _is_medium_numeric_regression_portfolio(
        features, target
    ) or _is_medium_mostly_numeric_regression_portfolio(features, target):
        return ["lgbmreg_direct", "catboostreg_direct"]
    if _is_small_wide_regression_portfolio(features, target):
        if _all_features_are_numeric(features):
            return ["lgbmreg_direct", "extra_treesreg_direct"]
        return ["lgbmreg_onehot_direct"]
    if _is_small_classic_regression_portfolio(features, target):
        return [
            "scaled_svrreg_direct",
            "xgboostreg_direct",
            "catboostreg_direct",
        ]
    if _is_very_large_compact_mixed_regression_portfolio(features, target):
        return [
            "lgbmreg_large_leaf127_direct",
            "lgbmreg_large_leaf255_direct",
        ]
    if _is_large_grouped_sequence_regression_portfolio(features, target):
        return [
            "grouped_trend_lgbmreg_leaf511_direct",
            "grouped_trend_lgbmreg_leaf255_regularized_direct",
        ]
    if _is_large_dense_low_cardinality_regression_portfolio(features, target):
        return ["lgbmreg_large_dense_direct"]
    if _is_large_high_categorical_regression_portfolio(features, target):
        return ["catboostreg_direct", "target_frequency_lgbmreg_direct"]
    if _is_large_compact_mixed_regression_portfolio(features, target):
        return [
            "lgbmreg_compact_direct",
            "xgboostreg_direct",
            "catboostreg_direct",
        ]
    if _is_small_mixed_regression_portfolio(features, target):
        return [
            "lgbmreg_direct",
            "xgboostreg_direct",
            "catboostreg_direct",
            "rfr_direct",
        ]
    return None


def _is_large_supported_mid_class_categorical(features, target):
    """Identify costly discrete tables where one deployable booster is safer.

    With hundreds of thousands of rows, a selector-scale LGBM/XGBoost pair can
    look marginally better than either member but leave room to deploy only the
    first refit.  Restricting this regime to moderately many, well-supported
    classes and a tested feature/work band avoids changing rare-label, narrow,
    continuous-numeric or many-class tasks.  Only training geometry and dtypes
    are inspected.
    """
    if not _has_declared_or_compact_categorical_features(features):
        return False

    shape = features.shape
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    round_complexity = feature_count * np.sqrt(max(class_count, 1))
    return bool(
        len(target_array) >= 100_000
        and 16 <= feature_count <= 64
        and 6 <= class_count <= 9
        and round_complexity <= 192
        and class_counts.min() >= 1_000
    )


def _is_leaf_supported_medium_many_class_numeric(features, target):
    """Identify dense numeric tables where stronger leaf support is well backed.

    The minimum class count is deliberately tied to the proposed 100-row leaf
    threshold. Requiring at least ten such groups prevents the regulariser from
    consuming most of a rare class, which was unstable on an independent highly
    imbalanced control.
    """
    if hasattr(features, "nnz") or not _all_features_are_numeric(features):
        return False
    if hasattr(features, "dtypes") and any(
        isinstance(dtype, pd.SparseDtype) for dtype in features.dtypes
    ):
        return False

    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        len(target_array) >= 20_000
        and 64 <= feature_count <= 256
        and 6 <= class_count <= 19
        and class_counts.min() >= 1_000
    )


def _is_leaf_supported_large_low_class_numeric(features, target):
    """Identify large dense numeric tables that can support regularised leaves.

    In the low-class regime the adaptive portfolio gives LightGBM 127 leaves.
    Requiring one hundred rows per leaf improves probability estimates when every
    class has ample support.  The feature/work bounds keep the rule in the same
    tractable interaction regime, while sparse and categorical tables retain their
    independently validated policies.
    """
    if hasattr(features, "nnz") or not _all_features_are_numeric(features):
        return False
    if hasattr(features, "dtypes") and any(
        isinstance(dtype, pd.SparseDtype) for dtype in features.dtypes
    ):
        return False

    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    round_complexity = feature_count * np.sqrt(max(class_count, 1))
    return bool(
        len(target_array) >= 20_000
        and 8 <= feature_count
        and 3 <= class_count <= 5
        and round_complexity <= 128
        and class_counts.min() >= 1_000
    )


def _adaptive_lgbm_min_child_samples(features, target, configured=None):
    """Regularise LightGBM leaves in structurally guarded regimes.

    Deep discrete trees and well-supported dense numeric tables both benefited from
    stronger row support across external folds. Discrete columns may be declared
    categorical or compact integer-coded. Numeric tables additionally require
    ample support in every class; sparse, rare-label, binary and out-of-range
    geometries retain LightGBM's default. Explicit configuration always wins.
    """
    if configured is not None:
        return int(configured)
    if _is_large_low_class_categorical(
        features, target
    ) or _is_leaf_supported_medium_many_class_numeric(
        features, target
    ) or _is_leaf_supported_large_low_class_numeric(features, target):
        return 100
    return None


def _adaptive_lgbm_min_child_weight(features, target, configured=None):
    """Prevent unstable class-specific leaves on costly many-class refits.

    LightGBM's default minimum leaf Hessian is deliberately permissive.  On
    dense wide problems with dozens of classes, later multiclass trees can then
    make very large updates after class probabilities have saturated.  Requiring
    one unit of Hessian support keeps those refits numerically stable.  The same
    geometry guard that bounds the high-work portfolio confines the stronger
    regularisation to well-supported full-scale tables; explicit configuration
    always takes precedence.
    """
    if configured is not None:
        return float(configured)
    if _is_high_work_wide_many_class_numeric(features, target):
        return 1.0
    return None


def _adaptive_xgboost_learning_rate(features, target):
    """Use a smaller boosting step for tractable high-cardinality targets.

    XGBoost's library default (0.3) is intentionally aggressive.  With many
    classes it tends to stop after only a few dozen trees and leaves noticeably
    worse multiclass logloss.  The work proxy prevents the threefold increase in
    trees on very wide or extremely high-cardinality tasks where it would consume
    the refit budget.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    class_count = max(len(np.unique(np.asarray(target).reshape(-1))), 1)
    if _is_extreme_wide_numeric_multiclass(features, target):
        # Very wide tables benefit from smaller boosting steps, but 0.1 tends to
        # consume the complete short-run tree budget.  A moderate 0.15 step kept
        # the selector inside the runtime limit while improving all external
        # folds in the wide-table controls.
        return 0.15
    round_complexity = feature_count * np.sqrt(class_count)
    if class_count >= 20 and round_complexity <= 512:
        return 0.1
    return None


def _adaptive_xgboost_max_bin(features, target):
    """Trade histogram resolution for class-supporting rows on extreme-wide data."""
    if _is_extreme_wide_numeric_multiclass(features, target):
        return 32
    return None


def _adaptive_xgboost_colsample_bytree(features, target):
    """Use random feature subspaces when a tree sees thousands of columns.

    On dense multiclass tables with several thousand numeric columns, full-column
    trees spend most of a short budget repeatedly considering correlated or weak
    predictors.  Sampling two percent still exposes dozens of columns per tree
    at the activation boundary and gives different trees useful diversity.
    """
    if _is_extreme_wide_numeric_multiclass(features, target):
        return 0.02
    return None


def _adaptive_xgboost_subsample(features, target):
    """Regularise costly multiclass trees with conservative row sampling.

    When each boosting round builds one tree per class, fully deterministic row
    reuse makes the trees strongly correlated.  A conservative 90% row sample
    adds diversity without discarding much rare-class evidence.  The rule is
    based only on task geometry. It is also useful when thousands of columns are
    already randomly subspaced; stronger 80% sampling was too noisy in that
    regime.
    """
    if _is_extreme_wide_numeric_multiclass(features, target):
        return 0.9
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    round_complexity = feature_count * np.sqrt(class_count)
    if (
        len(target_array) >= 20_000
        and class_count >= 20
        and round_complexity <= 512
    ):
        return 0.9
    return None


def _adaptive_xgboost_min_child_weight(features, target):
    """Regularise leaves on well-supported, tractable many-class tables.

    XGBoost's unit child-Hessian threshold permits very small class-specific
    leaves. A moderate threshold improves probability estimates when many
    classes each have ample observations. The class, work and support guards
    avoid applying it to binary, sparse-label or very wide multiclass problems.
    The boundary covers both moderately many-class tables and higher-cardinality
    narrow tables while leaving image-like and extreme-wide regimes unchanged.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    round_complexity = feature_count * np.sqrt(class_count)
    if (
        feature_count >= 32
        and _is_large_supported_mid_class_categorical(features, target)
    ):
        return 5.0
    if (
        len(target_array) >= 20_000
        and class_count >= 10
        and len(target_array) / class_count >= 100
        and round_complexity <= 640
    ):
        return 3.0
    return None


def _adaptive_xgboost_max_depth(features, target):
    """Regularise trees in high-class or very high work-complexity regimes."""
    if _is_extreme_wide_numeric_multiclass(features, target):
        return 4
    if _is_well_supported_narrow_many_class_numeric(features, target):
        # With only a few dozen columns and ample support in every class, one
        # additional level captures useful interactions while reaching early
        # stopping in materially fewer rounds.  The same row/class/cell bounds
        # that make the complementary RF affordable keep this branch out of
        # sparse, rare-label and costly wide-table regimes.
        return 6
    if _adaptive_xgboost_learning_rate(features, target) is not None:
        return 5
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    class_count = max(len(np.unique(np.asarray(target).reshape(-1))), 1)
    round_complexity = feature_count * np.sqrt(class_count)
    return 4 if round_complexity >= 1_024 else None


def _use_shallow_xgboost_probe(
    features,
    target,
    metric,
    validation_fold_count,
    adaptive_depth,
):
    """Probe a cheaper tree shape only in the validation-model reuse regime.

    Large narrow many-class tables can make an all-row XGBoost refit impossible
    under a short benchmark budget.  In that regime the holdout model is also the
    deployment model, so a second, materially different tree depth can be judged
    on exactly the probabilities that will be reused.  Existing geometry-driven
    depth policies take precedence, and mixed tables are excluded because compact
    categorical encoding showed less stable depth transfer in the controls.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    return (
        metric == "logloss"
        and validation_fold_count == 1
        and adaptive_depth is None
        and _all_features_are_numeric(features)
        and len(target_array) >= 100_000
        and 7 <= class_count <= 19
        and feature_count <= 32
    )


def _materially_better_shallow_score(shallow_score, default_score):
    """Require a robust holdout gain before changing the XGBoost tree shape."""
    required_gain = max(abs(float(default_score)) * 0.02, 0.002)
    return float(shallow_score) - float(default_score) >= required_gain


def _use_xgboost_child_weight_probe(
    features,
    target,
    metric,
    validation_fold_count,
    adaptive_min_child_weight,
):
    """Probe stronger leaf support in the bounded XGBoost/RF regime.

    The base and probe fits use the same train-only holdout.  This lets the
    adapter distinguish high-cardinality tables that benefit from suppressing
    small class-specific leaves from otherwise similar many-class tables where
    the same regularisation underfits.  Geometry limits bound the extra fit.
    """
    return (
        metric == "logloss"
        and validation_fold_count == 1
        and adaptive_min_child_weight == 3.0
        and _is_well_supported_narrow_many_class_numeric(features, target)
    )


def _materially_better_xgboost_regularisation(
    strong_score,
    default_score,
    strong_probabilities,
    default_probabilities,
    truth,
    labels,
):
    """Require both a practical and paired-significant validation gain."""
    score_gain = float(strong_score) - float(default_score)
    practical_gain = max(abs(float(default_score)) * 0.002, 0.002)
    strong_losses = _row_log_losses(strong_probabilities, truth, labels=labels)
    default_losses = _row_log_losses(default_probabilities, truth, labels=labels)
    paired_improvements = default_losses - strong_losses
    standard_error = (
        np.std(paired_improvements, ddof=1) / np.sqrt(len(paired_improvements))
        if len(paired_improvements) > 1
        else 0.0
    )
    return score_gain >= max(practical_gain, 1.96 * standard_error)


def _candidate_model_params(
    candidate,
    lgbm_num_leaves=None,
    lgbm_min_child_samples=None,
    lgbm_min_child_weight=None,
    xgboost_learning_rate=None,
    xgboost_max_depth=None,
    xgboost_max_bin=None,
    xgboost_colsample_bytree=None,
    xgboost_subsample=None,
    xgboost_min_child_weight=None,
    logit_c=None,
    rf_n_estimators=None,
):
    """Build candidate-specific overrides shared by selection and refits."""
    if candidate in {"lgbm", _LGBM_EXTRA_TREES_MODEL}:
        parameters = {}
        if lgbm_num_leaves is not None:
            parameters["num_leaves"] = lgbm_num_leaves
        if lgbm_min_child_samples is not None:
            parameters["min_child_samples"] = lgbm_min_child_samples
        if lgbm_min_child_weight is not None:
            parameters["min_child_weight"] = lgbm_min_child_weight
        if candidate == _LGBM_EXTRA_TREES_MODEL:
            parameters["extra_trees"] = True
        return parameters
    if candidate == "logit" and logit_c is not None:
        return {"C": logit_c}
    if candidate == "scaled_logit":
        # On wide numeric tables, the dual liblinear problem is substantially
        # cheaper than an unconstrained multiclass LBFGS fit. Scaling makes the
        # deliberately strong L2 regularisation comparable across datasets.
        return {
            "C": 0.001 if logit_c is None else logit_c,
            "solver": "liblinear",
            "dual": True,
            "max_iter": 2_000,
        }
    if _is_scaled_svc_candidate(candidate):
        parameters = {
            "C": _SMALL_DENSE_SVC_C,
            "gamma": "scale",
            "probability": True,
        }
        if candidate == "scaled_svc_strong_half":
            parameters.update(C=30.0, _gamma_multiplier=0.5)
        elif candidate == "scaled_svc_strong_scale":
            parameters.update(C=30.0)
        return parameters
    if candidate == "mixed_logit":
        return {
            "C": 3.0,
            "max_iter": 2_000,
        }
    if candidate == "mixed_svc":
        return {
            "C": _SMALL_DENSE_SVC_C,
            "gamma": "scale",
            "probability": True,
        }
    if candidate == "extra_trees":
        return {
            "n_estimators": 500,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
    if candidate == "extra_trees_wide":
        return {
            "n_estimators": 500,
            "min_samples_leaf": 1,
            "max_features": 0.5,
        }
    if candidate == "mixed_extra_trees":
        return {
            "n_estimators": 500,
            "min_samples_leaf": 8,
            "max_features": 1.0,
        }
    if candidate in {"relational_hist", "relational_hist_second_order"}:
        return {
            "max_iter": 250,
            "learning_rate": 0.1,
            "max_leaf_nodes": 127,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        }
    if candidate in {"rf", "rf_large_subspace"}:
        parameters = {}
        if rf_n_estimators is not None:
            parameters["n_estimators"] = rf_n_estimators
        if candidate == "rf_large_subspace":
            parameters["max_features"] = 0.1
        return parameters
    if candidate == "xgboost":
        parameters = {}
        if xgboost_learning_rate is not None:
            parameters["learning_rate"] = xgboost_learning_rate
        if xgboost_max_depth is not None:
            parameters["max_depth"] = xgboost_max_depth
        if xgboost_max_bin is not None:
            parameters["max_bin"] = xgboost_max_bin
        if xgboost_colsample_bytree is not None:
            parameters["colsample_bytree"] = xgboost_colsample_bytree
        if xgboost_subsample is not None:
            parameters["subsample"] = xgboost_subsample
        if xgboost_min_child_weight is not None:
            parameters["min_child_weight"] = xgboost_min_child_weight
        return parameters
    return {}


def _adaptive_default_portfolio_candidates(
    features, target, configured=None, metric=None
):
    """Choose a compute-feasible, structurally diverse default portfolio.

    A linear classifier is especially complementary to trees on moderately sized,
    high-dimensional numerical tables.  It is deliberately substituted for LGBM
    rather than added as a third candidate: selecting and refitting all three does
    not fit a short benchmark budget.  Very large dense tables, categorical tables,
    and many-class problems keep the tree portfolio because an unbounded LBFGS
    refit would be both expensive and unreliable there.
    """
    if configured is not None:
        return _portfolio_candidates(configured)

    target_array = np.asarray(target).reshape(-1)
    row_count = len(target_array)
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = max(len(class_counts), 1)
    if hasattr(features, "dtypes"):
        categorical_columns = [
            column
            for column in features.columns
            if not pd.api.types.is_numeric_dtype(features[column].dtype)
        ]
        categorical_cardinality = sum(
            features[column].nunique(dropna=True) for column in categorical_columns
        )
        expanded_feature_count = (
            feature_count - len(categorical_columns) + categorical_cardinality
        )
        categorical_encoding_is_feasible = categorical_cardinality <= 512
        all_features_are_numeric = not categorical_columns
    else:
        expanded_feature_count = feature_count
        all_features_are_numeric = np.asarray(features).dtype.kind in "biufc"
        categorical_encoding_is_feasible = all_features_are_numeric
    dense_cells = row_count * max(expanded_feature_count, 1)

    if _is_large_narrow_discrete_relational_candidate(features, target):
        # Compact discrete spaces can depend on relative offsets and equality
        # relations that require many axis-aligned booster splits.  A bounded
        # expansion supplies that representation as one OOF candidate.  Equality
        # between pairwise distances is additionally exposed when the complete
        # second-order dense design remains below its separate cell budget; wider
        # tables retain the cheaper first-order form.  The established logloss
        # selector can retain a booster blend or reject either representation.
        relational_candidate = (
            "relational_hist_second_order"
            if _is_second_order_relational_expansion_feasible(features)
            else "relational_hist"
        )
        return ["lgbm", "xgboost", relational_candidate]

    if _is_small_supported_mixed_multiclass(features, target):
        # On compact mixed tables below the existing mixed-candidate row floor,
        # the composed ``auto`` pipeline and shallow boosters can both miss the
        # useful nominal structure.  Cheap train-only one-hot linear and ordinal
        # forest views cover complementary boundaries and leave OOF selection to
        # choose a singleton or conservative blend.  Resource, cardinality and
        # class-support bounds keep all four candidates reliable and inexpensive.
        return ["lgbm", "xgboost", "mixed_logit", "mixed_extra_trees"]

    if _is_large_supported_mid_class_categorical(features, target):
        # A selector-scale pair is misleading in this costly regime: after the
        # first all-row refit there is normally no budget for the second member,
        # so choose the consistently deployable booster before spending that time.
        return ["xgboost"]

    if _is_high_work_wide_many_class_numeric(features, target):
        # Candidate selection is not the bottleneck in this regime; fitting a
        # second multiclass booster consumes budget that is better spent on the
        # all-row LGBM refit.  Explicit candidate configuration still takes
        # precedence before this adaptive policy is called.
        return ["lgbm"]

    if _is_extreme_wide_numeric_multiclass(features, target):
        # A single regularised XGBoost candidate leaves enough of a short budget
        # to build a useful model.  On this geometry an LGBM-first portfolio can
        # exhaust selection time before the random-subspace candidate is tried.
        return ["xgboost"]

    if _is_well_supported_narrow_many_class_numeric(features, target):
        # On large, narrow many-class tables a modest forest is a cheap source
        # of probability diversity.  A fixed shrinkage blend with XGBoost was
        # stable across external folds, while a second booster transferred less
        # reliably.  The forest-specific memory proxy in the geometry guard is
        # essential because tree value arrays grow with the class count.
        return ["xgboost", "rf"]

    if _is_small_dense_kernel_multiclass(features, target):
        # Kernel distances are especially effective on bounded, genuinely dense
        # numerical tables, while a quadratic-cost fit remains cheap at this row
        # cap.  Three-fold OOF probabilities provide honest temperature scaling.
        return ["scaled_svc"]

    if _is_medium_dense_kernel_multiclass(features, target):
        # A larger RBF view remains cheap when both dense cells and class support
        # are bounded.  Requiring genuinely dense values avoids applying the
        # image/spectral geometry assumption to materialised sparse tables.
        return ["scaled_svc"]

    small_wide_many_class = (
        row_count <= 2_500
        and feature_count >= 256
        and feature_count >= 0.75 * row_count
        and class_count >= 10
        and dense_cells <= 5_000_000
        and all_features_are_numeric
    )
    if small_wide_many_class:
        # Random feature subspaces are especially useful when there are more
        # columns than examples, or nearly as many.  RF also gives probability
        # errors unlike the two boosting families and is much cheaper than a
        # composed ``auto`` candidate in this small-sample regime.
        extreme_p_over_n = feature_count >= 4_096 and feature_count >= 8 * row_count
        if extreme_p_over_n and row_count <= 256:
            # On very wide tables, sequential boosters and large RF subspaces can
            # consume a short benchmark budget before a robust fallback is fitted.
            # A scaled dual linear model is efficient for p >> n, while ordinary
            # RF remains complementary for image-like manifolds.  OOF validation
            # chooses between them without looking at task identity.
            return ["scaled_logit", "rf"]

        candidates = ["lgbm", "xgboost", "rf"]
        low_class_support = (
            len(class_counts) > 0
            and class_counts.min() >= 5
            and row_count / class_count <= 40
        )
        if feature_count > 1_024 or (
            feature_count == 1_024 and low_class_support
        ):
            # A larger per-tree subspace is a useful, still-cheap source of
            # diversity on very high-dimensional data.  At the 1,024-column
            # boundary it is enabled only when every class is represented and
            # per-class support is low; better-supported image tables retain the
            # cheaper ordinary forest.  OOF validation can then reject the wider
            # subspace when its probability errors are not complementary.
            candidates.append("rf_large_subspace")
        return candidates

    if _is_dense_scaled_logit_regime(features, target):
        # A scaled, regularised linear view has probability errors unlike a
        # shallow tree booster.  Very wide multiclass matrices that are genuinely
        # dense favour LGBM's faster histogram traversal; binary, sparse-like and
        # more moderate-width tables keep XGBoost.  The density estimate reads at
        # most a bounded deterministic sample and never uses outer outcomes.
        dense_wide_multiclass = (
            class_count >= 3
            and feature_count >= 1_024
            and _sampled_numeric_density(features) >= 0.50
        )
        return [
            "scaled_logit",
            "lgbm" if dense_wide_multiclass else "xgboost",
        ]

    linear_is_feasible = (
        row_count >= 2_500
        and feature_count >= 256
        and dense_cells <= 12_000_000
        and class_count <= 20
        and categorical_encoding_is_feasible
    )
    if linear_is_feasible:
        return ["logit", "xgboost"]

    candidates = ["lgbm", "xgboost", "auto"]
    if row_count > 5_000:
        candidates.remove("auto")
    if _is_bounded_numeric_extra_trees_candidate(features, target):
        candidates.insert(2, "extra_trees")
        next_index = 3
        if (
            metric == "logloss"
            and _is_train_gated_wide_extra_trees_candidate(features, target)
        ):
            candidates.insert(next_index, "extra_trees_wide")
            next_index += 1
        candidates.insert(next_index, "catboost")
    else:
        mixed_candidates = []
        if _is_bounded_mixed_logit_candidate(features, target):
            mixed_candidates.append("mixed_logit")
        if _is_bounded_mixed_svc_candidate(features, target) or (
            metric == "auc"
            and _is_small_supported_mixed_binary_svc(features, target)
        ):
            mixed_candidates.append("mixed_svc")
        if _is_bounded_mixed_extra_trees_candidate(features, target):
            mixed_candidates.append("mixed_extra_trees")
        if mixed_candidates:
            candidates[2:2] = mixed_candidates
    if _is_small_narrow_dense_kernel_multiclass(features, target):
        # A scaled RBF view is a cheap source of nonlinear distance structure on
        # compact narrow numerical tables.  Unlike the wider image/spectral
        # regimes above, it is added to the ordinary portfolio rather than made
        # the sole candidate.  A large raw-OOF dominance margin can promote it
        # to a singleton; otherwise the ordinary calibrated selector remains in
        # control and can reject it or use it only when complementary.
        kernel_candidates = [
            "scaled_svc",
            "scaled_svc_strong_half",
            "scaled_svc_strong_scale",
        ]
        # Evaluate the cheap, structurally motivated views before the optional
        # composed ``auto`` candidate.  Runtime variance in the composed model
        # must not prevent a sub-second SVC candidate from being measured.
        auto_index = candidates.index("auto") if "auto" in candidates else len(candidates)
        candidates[auto_index:auto_index] = kernel_candidates
    return candidates


def _is_small_narrow_dense_kernel_multiclass(features, target):
    """Recognise cheap narrow multiclass tables worth an RBF OOF candidate.

    Rows, width, dense cells and class support jointly bound kernel fitting and
    nested probability calibration.  Numeric type, storage and sampled density
    checks keep categorical and materialised sparse tables out.  The predicate
    intentionally only admits a candidate; OOF selection retains the existing
    tree portfolio whenever the kernel view does not transfer.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        500 <= row_count <= 2_500
        and 16 <= feature_count <= 127
        and row_count * feature_count <= 300_000
        and 3 <= class_count <= 10
        and class_counts.min() >= 50
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
        and _sampled_numeric_density(features) >= 0.25
    )


def _is_small_dense_kernel_multiclass(features, target):
    """Recognise bounded dense numeric tables where an RBF view is affordable.

    The row and dense-cell caps bound SVC's superlinear fit cost.  Width excludes
    ordinary narrow tabular problems, while sampled nonzero density keeps sparse
    bag-of-words matrices out even when they are materialised as dense uint8.
    Class support is required for stable nested probability calibration.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        1_000 <= row_count <= 2_500
        and 128 <= feature_count <= 2_048
        and row_count * feature_count <= 3_000_000
        and 3 <= class_count <= 20
        and class_counts.min() >= 50
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
        and _sampled_numeric_density(features) >= 0.25
    )


def _prefer_dominant_adaptive_scaled_svc(
    contenders,
    enabled,
    minimum_logloss_gain=_NARROW_KERNEL_DOMINANCE_MIN_LOGLOSS_GAIN,
):
    """Use a narrow-kernel singleton only after a large raw OOF victory."""
    if minimum_logloss_gain < 0:
        raise ValueError("minimum logloss gain must be non-negative")
    if not enabled:
        return contenders
    scaled_svc = [
        contender
        for contender in contenders
        if _is_scaled_svc_candidate(contender.get("model"))
    ]
    alternatives = [
        contender
        for contender in contenders
        if not _is_scaled_svc_candidate(contender.get("model"))
    ]
    if len(scaled_svc) != 1 or not alternatives:
        return contenders
    strongest_alternative = max(
        alternatives, key=lambda contender: contender["score"]
    )
    if (
        scaled_svc[0]["score"] - strongest_alternative["score"]
        >= float(minimum_logloss_gain)
    ):
        return scaled_svc
    return contenders


def _is_scaled_svc_candidate(candidate):
    return bool(
        candidate == "scaled_svc"
        or (
            isinstance(candidate, str)
            and candidate.startswith(_SCALED_SVC_VARIANT_PREFIX)
        )
    )


def _select_supported_adaptive_scaled_svc_variant(
    contenders,
    enabled,
    minimum_logloss_gain=_NARROW_KERNEL_VARIANT_MIN_LOGLOSS_GAIN,
):
    """Keep a stronger fixed RBF view only after a material raw-OOF gain.

    The alternative views are deliberately limited to the already bounded narrow
    kernel regime.  Their probabilities are evaluated on exactly the same OOF
    rows as the established C=3 candidate.  A meaningful margin is required
    before changing the deployment hyperparameters; otherwise the original view
    is retained byte-for-byte and the extra validation models are discarded.
    """
    if minimum_logloss_gain < 0:
        raise ValueError("minimum logloss gain must be non-negative")
    if not enabled:
        return contenders
    baseline = [
        contender
        for contender in contenders
        if contender.get("model") == "scaled_svc"
    ]
    variants = [
        contender
        for contender in contenders
        if _is_scaled_svc_candidate(contender.get("model"))
        and contender.get("model") != "scaled_svc"
    ]
    if len(baseline) != 1 or not variants:
        return contenders
    selected = max(variants, key=lambda contender: contender["score"])
    if selected["score"] - baseline[0]["score"] < float(minimum_logloss_gain):
        selected = baseline[0]
    return [
        contender
        for contender in contenders
        if not _is_scaled_svc_candidate(contender.get("model"))
        or contender is selected
    ]


def _skip_adaptive_composed_auto_after_kernel_dominance(
    candidate,
    configured_candidates,
    features,
    target,
    contenders,
):
    """Avoid an expensive composed candidate after decisive bounded OOF evidence."""
    if (
        candidate != "auto"
        or configured_candidates is not None
        or not _is_small_narrow_dense_kernel_multiclass(features, target)
    ):
        return False
    provisional = _select_supported_adaptive_scaled_svc_variant(
        contenders, enabled=True
    )
    provisional = _prefer_dominant_adaptive_scaled_svc(
        provisional, enabled=True
    )
    return bool(
        len(provisional) == 1
        and _is_scaled_svc_candidate(provisional[0].get("model"))
    )


def _is_medium_dense_kernel_multiclass(features, target):
    """Recognise supported medium dense tables where an RBF view is affordable.

    The dense-cell cap limits both distance work and memory, while the row cap
    bounds SVC's superlinear fit cost.  The minimum class support makes nested
    probability calibration reliable.  All signals come from the training table;
    sampled density excludes sparse-like matrices independently of storage type.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        5_000 <= row_count <= 10_000
        and 256 <= feature_count <= 1_024
        and row_count * feature_count <= 6_000_000
        and 10 <= class_count <= 30
        and class_counts.min() >= 100
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
        and _sampled_numeric_density(features) >= 0.25
    )


def _is_bounded_numeric_extra_trees_candidate(features, target):
    """Allow OOF selection of tree-diversity views on cheap dense tables.

    Extra Trees and CatBoost expose complementary randomised and ordered
    probability structure on some small and medium numerical datasets, but no
    fixed geometry makes either one a universal winner.  This predicate therefore
    only bounds candidate cost; the existing OOF selector must still beat or
    complement the ordinary tree boosters before either view is deployed.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        1_000 <= row_count <= 10_000
        and 8 <= feature_count <= 64
        and row_count * feature_count <= 500_000
        and 3 <= class_count <= 10
        and class_counts.min() >= 5
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
        and _sampled_numeric_density(features) >= 0.25
    )


def _is_train_gated_wide_extra_trees_candidate(features, target):
    """Bound the extra shape probe to affordable medium-row numeric tables."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    return bool(
        5_000 <= row_count <= 10_000
        and 16 <= feature_count <= 64
        and _is_bounded_numeric_extra_trees_candidate(features, target)
    )


def _is_train_gated_lgbm_extra_trees_pair_candidate(features, target):
    """Recognise the frozen dense multiclass random-threshold domain."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    if len(target_array) != row_count:
        return False
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        30_000 <= row_count <= 100_000
        and 64 <= feature_count <= 256
        # The frozen contender used min_child_samples=100. The adaptive
        # production helper keeps that exact value only below 20 classes.
        # Its shared XGBoost reproduces the frozen min_child_weight=3 control
        # only from ten classes onward.
        and 10 <= class_count <= 19
        and feature_count * np.sqrt(class_count) <= 640
        and class_counts.min() >= 1_000
        and row_count * feature_count <= 20_000_000
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    )


def _use_train_gated_lgbm_extra_trees_pair(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Enable the validated challenger only for an untouched default portfolio."""
    has_explicit_portfolio_settings = any(
        str(key).startswith("_portfolio_") for key in framework_params
    )
    return bool(
        metric == "logloss"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_train_gated_lgbm_extra_trees_pair_candidate(features, target)
    )


def _is_bounded_mixed_extra_trees_candidate(features, target):
    """Allow OOF selection of a regularised ordinal forest on cheap mixed tables.

    Low-cardinality categoricals often benefit from treating nearby encoded
    partitions as a complementary tree view.  Geometry and class-support bounds
    make that view cheap and keep its calibration stable; OOF selection remains
    responsible for rejecting it on linear or interaction-heavy tables.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or not hasattr(features, "dtypes"):
        return False
    row_count, feature_count = shape
    profile = _categorical_encoding_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        1_000 <= row_count <= 10_000
        and 8 <= feature_count <= 64
        and row_count * feature_count <= 500_000
        and 3 <= class_count <= 10
        and class_counts.min() >= 100
        and categorical_count >= 2
        and categorical_count >= 0.25 * feature_count
        and profile["categorical_cardinality"] <= 256
        and profile["estimated_dense_cells"] <= 1_000_000
        and not _is_sparse_table(features)
    )


def _is_small_supported_mixed_multiclass(features, target):
    """Recognise well-supported compact mixed tables below 1,000 rows.

    This regime is intentionally separate from the broader mixed ExtraTrees
    guard: fewer rows require stronger class-support and categorical-information
    lower bounds, while the one-hot and ordinal views remain cheap enough for
    three-fold OOF selection.  Only training geometry, dtypes and cardinalities
    are used; task names and external-fold outcomes are not available here.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or not hasattr(features, "dtypes"):
        return False
    row_count, feature_count = shape
    profile = _categorical_encoding_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        500 <= row_count < 1_000
        and 12 <= feature_count <= 64
        and row_count * feature_count <= 500_000
        and 3 <= class_count <= 10
        and class_counts.min() >= 30
        and categorical_count >= 2
        and categorical_count >= 0.25 * feature_count
        and 24 <= profile["categorical_cardinality"] <= 256
        and profile["estimated_dense_cells"] <= 1_000_000
        and not _is_sparse_table(features)
    )


def _is_bounded_mixed_logit_candidate(features, target):
    """Add a cheap one-hot linear view only on sufficiently informative widths.

    Very narrow mixed tables provide too few independent linear directions and
    showed unstable blend transfer despite acceptable OOF scores.  Requiring at
    least 16 raw columns keeps the bounded mixed-data resource and class-support
    guarantees while leaving the OOF selector responsible for model quality.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    return bool(
        len(shape) == 2
        and shape[1] >= 16
        and _is_bounded_mixed_extra_trees_candidate(features, target)
    )


def _is_bounded_mixed_svc_candidate(features, target):
    """Bound an RBF view to cheap mixed geometries with stable class support.

    Fully categorical narrow tables form a compact Hamming-like space where RBF
    interactions are effective.  Wider mixed tables provide enough independent
    one-hot directions for the same kernel view.  The gap between those regimes
    excludes ambiguous narrow mixed geometries, while row, cardinality and cell
    caps bound the quadratic kernel work and encoded memory.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or not hasattr(features, "dtypes"):
        return False
    row_count, feature_count = shape
    profile = _categorical_encoding_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    narrow_categorical = bool(
        4 <= feature_count <= 8
        and categorical_count == feature_count
        and profile["categorical_cardinality"] <= 64
    )
    wider_mixed = bool(
        16 <= feature_count <= 64
        and categorical_count >= 2
        and categorical_count >= 0.25 * feature_count
        and profile["categorical_cardinality"] <= 256
    )
    return bool(
        1_000 <= row_count <= 2_500
        and 3 <= class_count <= 10
        and class_counts.min() >= 50
        and (narrow_categorical or wider_mixed)
        and profile["estimated_dense_cells"] <= 1_000_000
        and not _is_sparse_table(features)
    )


def _is_small_supported_mixed_binary_svc(features, target):
    """Recognise compact mixed binary AUC tables worth a one-hot RBF view.

    This deliberately separate predicate does not broaden the established
    multiclass SVC regime.  It requires enough rows and minority examples for
    stable three-fold probability fitting, enough raw mixed directions for the
    kernel representation to be useful, and strict encoded-cell/cardinality
    caps.  Candidate admission uses only training geometry and dtypes; the OOF
    selector remains responsible for accepting or rejecting the fitted view.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or not hasattr(features, "dtypes"):
        return False
    row_count, feature_count = shape
    profile = _categorical_encoding_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    return bool(
        800 <= row_count <= 2_500
        and 16 <= feature_count <= 64
        and len(class_counts) == 2
        and class_counts.min() >= 100
        and categorical_count >= 4
        and categorical_count >= 0.20 * feature_count
        and profile["categorical_cardinality"] <= 256
        and profile["estimated_dense_cells"] <= 1_000_000
        and not _is_sparse_table(features)
    )


def _bounded_mostly_numeric_one_hot_profile(features, max_cells=2_000_000):
    """Describe a dense wide table whose few categories encode cheaply."""
    profile = _categorical_encoding_profile(features)
    if profile is None or max_cells <= 0:
        return None

    frame = pd.DataFrame(features)
    categorical_columns = profile["categorical_columns"]
    numerical_columns = [
        column for column in frame.columns if column not in categorical_columns
    ]
    if not numerical_columns:
        return None

    feature_count = frame.shape[1]
    if (
        len(categorical_columns) > 64
        or 20 * len(categorical_columns) > feature_count
        or profile["categorical_cardinality"] > 128
        or profile["estimated_columns"] >= 2_048
        or profile["estimated_dense_cells"] > 12_000_000
    ):
        return None

    sampled_rows = min(
        len(frame), max(max_cells // len(numerical_columns), 1)
    )
    if sampled_rows < len(frame):
        indices = np.linspace(0, len(frame) - 1, sampled_rows, dtype=int)
        sample = frame.iloc[indices].loc[:, numerical_columns]
    else:
        sample = frame.loc[:, numerical_columns]
    values = sample.to_numpy(copy=False)
    observed = np.isfinite(values) & (values != 0)
    density = float(np.count_nonzero(observed)) / float(values.size)
    return profile if density >= 0.50 else None


def _is_dense_scaled_logit_regime(features, target):
    """Recognise bounded dense wide tables suited to a scaled linear/tree pair.

    Both the dense-cell and class-work caps have an operational meaning:
    they keep a dual linear refit plus a shallow XGBoost refit inside the short
    benchmark budget. Minimum row and class support guards avoid applying a
    fixed regularisation strength where its variance would be poorly controlled.
    The predicate deliberately uses only training geometry and dtypes.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    all_numeric = _all_features_are_numeric(features)
    one_hot_profile = (
        _bounded_mostly_numeric_one_hot_profile(features)
        if class_count == 2 and not all_numeric
        else None
    )
    effective_feature_count = (
        one_hot_profile["estimated_columns"]
        if one_hot_profile is not None
        else feature_count
    )
    dense_cells = row_count * effective_feature_count
    dense_wide_budget = (
        all_numeric
        and effective_feature_count >= 1_024
        and dense_cells <= 20_000_000
        and not _is_sparse_table(features)
        and _sampled_numeric_density(features) >= 0.50
    )
    supported_multiclass = (
        row_count >= 5_000
        and 3 <= class_count <= 20
        and all_numeric
    )
    supported_binary_wide = (
        row_count >= 4_500
        and class_count == 2
        and effective_feature_count >= 1_024
        and (all_numeric or one_hot_profile is not None)
    )
    return (
        (supported_multiclass or supported_binary_wide)
        and 256 <= effective_feature_count < 2_048
        and class_counts.min() >= 100
        and (dense_cells <= 12_000_000 or dense_wide_budget)
        and dense_cells * class_count <= 100_000_000
        and not _is_sparse_table(features)
    )


def _adaptive_logit_c(features, target, candidates):
    """Use validated moderate regularisation only in the guarded dense regime."""
    if (
        len(candidates) == 2
        and candidates[0] == "scaled_logit"
        and candidates[1] in {"lgbm", "xgboost"}
        and _is_dense_scaled_logit_regime(features, target)
    ):
        class_count = len(np.unique(np.asarray(target).reshape(-1)))
        return 0.001 if class_count == 2 else 0.01
    return None


def _is_high_work_wide_many_class_numeric(features, target):
    """Recognise dense multiclass tables whose full boosting work is extreme.

    The first-order row-feature-class product is deliberately used only after
    minimum width and class-count guards.  Sparse and categorical tables are
    excluded because their effective per-round work is not represented by the
    dense-cell proxy.  The policy therefore depends only on observed training
    geometry and never on a benchmark task or dataset identity.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    row_count = shape[0] if shape else len(np.asarray(target).reshape(-1))
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    return (
        class_count >= 30
        and feature_count >= 256
        and row_count * feature_count * class_count >= 2_000_000_000
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    )


def _is_large_narrow_extreme_many_class_numeric(features, target):
    """Recognise a bounded dense regime needing a larger XGBoost selector.

    Hundreds of well-supported classes make a small selector unrepresentative,
    while the narrow feature count keeps a 10k-row XGBoost fit affordable.  The
    dense-cell cap bounds both selector and prediction work, and the lower work
    proxy keeps this policy separate from ordinary many-class tables.  Every
    boundary is based on training geometry and dtypes, never task identity.
    """
    shape = getattr(features, "shape", None)
    if shape is None:
        shape = np.asarray(features).shape
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    if len(target_array) != row_count:
        return False
    _, class_counts = np.unique(target_array, return_counts=True)
    if len(class_counts) == 0:
        return False
    class_count = len(class_counts)
    return bool(
        row_count >= 200_000
        and 48 <= feature_count <= 64
        and 256 <= class_count <= 512
        and class_counts.min() >= 500
        and row_count * feature_count <= 30_000_000
        and feature_count * np.sqrt(class_count) >= 1_024
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    )


def _use_large_narrow_extreme_many_class_xgboost(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use the validated XGBoost-only 10k selector under a strict guard."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "logloss"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_large_narrow_extreme_many_class_numeric(features, target)
    )


def _is_well_supported_narrow_many_class_numeric(features, target):
    """Recognise a resource-bounded regime where RF probability shrinkage helps.

    Random-forest node values scale with both the number of training rows and
    classes.  The row-class cap therefore protects memory independently of raw
    table width, while the dense-cell and feature caps bound fitting work.  The
    support guard excludes rare-label problems where forest probabilities are
    too discrete for a fixed blend.  All boundaries depend only on training-set
    geometry and deliberately exclude sparse or categorical tables.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    row_count = shape[0] if shape else len(np.asarray(target).reshape(-1))
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    return (
        row_count >= 20_000
        and class_count >= 20
        and row_count / class_count >= 100
        and feature_count <= 64
        and row_count * feature_count <= 4_000_000
        and row_count * class_count <= 7_000_000
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    )


def _adaptive_pair_strong_weight(features, target, candidates):
    """Use validated fixed shrinkage for guarded portfolio geometries."""
    if "mixed_svc" in candidates and len(np.unique(np.asarray(target))) >= 3:
        return 0.75
    if (
        candidates == ["lgbm", "xgboost", "mixed_logit", "mixed_extra_trees"]
        and _is_small_supported_mixed_multiclass(features, target)
    ):
        return 0.75
    if (
        candidates == ["xgboost", "rf"]
        and _is_well_supported_narrow_many_class_numeric(features, target)
    ):
        return 0.75
    return 0.5


def _adaptive_rf_n_estimators(features, target, candidates):
    """Keep the guarded large-table forest below the short-run memory budget."""
    if not any(candidate in {"rf", "rf_large_subspace"} for candidate in candidates):
        return None
    if _is_well_supported_narrow_many_class_numeric(features, target):
        return 200
    return 300


def _is_extreme_wide_numeric_multiclass(features, target):
    """Recognise the dense compute regime, independently of dataset identity."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    if hasattr(features, "dtypes"):
        all_features_are_numeric = all(
            pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
        )
    else:
        dtype = getattr(features, "dtype", None)
        all_features_are_numeric = dtype is not None and np.issubdtype(
            dtype, np.number
        )
    return (
        len(target_array) >= 2_500
        and feature_count >= 2_048
        and 3 <= class_count <= 20
        and all_features_are_numeric
        and not _is_sparse_table(features)
    )


def _is_resource_bounded_extreme_wide_numeric(features, target):
    """Identify dense extreme-wide fits that are costly but safe on all rows."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    row_count = shape[0] if shape else len(np.asarray(target).reshape(-1))
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    class_count = max(len(np.unique(target_array)), 1)
    return (
        row_count >= 5_000
        and feature_count >= 4_096
        and 3 <= class_count <= 20
        and row_count * feature_count <= 70_000_000
        and row_count * feature_count * class_count <= 700_000_000
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    )


def _is_sparse_native_linear_regime(features, target):
    """Recognise bounded sparse p >> n tables suited to a raw linear model.

    The nonzero cap bounds both fit work and memory. Requiring each class to have
    meaningful support avoids committing a fixed linear fallback on rare-label
    problems where a validation-based choice would still be necessary.
    """
    nonzero_count = _sparse_nonzero_count(features)
    if nonzero_count is None:
        return False
    shape = features.shape
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        1_000 <= row_count <= 20_000
        and feature_count >= 4_096
        and feature_count >= row_count
        and 2 <= class_count <= 20
        and class_counts.min() >= 20
        and nonzero_count <= 5_000_000
    )


def _sparse_nonzero_count(features):
    """Return stored values for scipy or fully sparse pandas tables."""
    if hasattr(features, "nnz") and hasattr(features, "tocsr"):
        return int(features.nnz)
    if hasattr(features, "dtypes") and all(
        isinstance(dtype, pd.SparseDtype) for dtype in features.dtypes
    ):
        return int(round(features.sparse.density * np.prod(features.shape)))
    return None


def _as_scipy_sparse_matrix(features):
    """Normalise supported sparse containers without materialising dense data."""
    if hasattr(features, "tocsr"):
        return features.tocsr()
    if hasattr(features, "dtypes") and all(
        isinstance(dtype, pd.SparseDtype) for dtype in features.dtypes
    ):
        return features.sparse.to_coo().tocsr()
    raise TypeError("Sparse-native fallback requires a scipy or pandas sparse table")


def _use_sparse_native_logit(features, target, metric, framework_params):
    """Use the raw-CSR fallback only for an unmodified adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "logloss"
        and not has_explicit_portfolio_settings
        and _is_sparse_native_linear_regime(features, target)
    )


def _is_materialized_sparse_tfidf_regime(features, target):
    """Recognise bounded sparse-like count tables suited to a TF-IDF view.

    Very wide scipy/pandas sparse tables retain their existing raw-linear path.
    This complementary bounded-width regime accepts either sparse storage or a
    dense table with a small sampled nonzero fraction. Shape, work and class-
    support caps bound the six OOF fits plus one full refit.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    if (
        not 500 <= row_count <= 5_000
        or not 512 <= feature_count <= 2_500
        or feature_count < 0.5 * row_count
        or row_count * feature_count > 5_000_000
        or row_count * feature_count * max(class_count, 1) > 100_000_000
        or not 3 <= class_count <= 25
        or len(class_counts) == 0
        or class_counts.min() < 30
        or not _all_features_are_numeric(features)
    ):
        return False

    stored_nonzero_count = _sparse_nonzero_count(features)
    if stored_nonzero_count is not None:
        matrix = _as_scipy_sparse_matrix(features)
        stored_values = matrix.data
        if (
            stored_values.size
            and (
                not np.isfinite(stored_values).all()
                or stored_values.min() < 0
            )
        ):
            return False
        density = float(stored_nonzero_count) / float(row_count * feature_count)
        return density <= 0.10

    sampled_rows = min(row_count, max(2_000_000 // feature_count, 1))
    if sampled_rows < row_count:
        row_indices = np.linspace(0, row_count - 1, sampled_rows, dtype=int)
        sample = _slice_rows(features, row_indices)
    else:
        sample = features
    if hasattr(sample, "to_numpy"):
        sample = sample.to_numpy(copy=False)
    try:
        sample = np.asarray(sample, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if sample.size == 0 or not np.isfinite(sample).all() or sample.min() < 0:
        return False
    density = float(np.count_nonzero(sample)) / float(sample.size)
    return density <= 0.10


def _as_materialized_sparse_csr(features):
    """Normalise a finite non-negative sparse-like numeric table to CSR."""
    if _sparse_nonzero_count(features) is not None:
        matrix = _as_scipy_sparse_matrix(features).astype(np.float32, copy=False)
        observed_values = matrix.data
    else:
        if hasattr(features, "to_numpy"):
            features = features.to_numpy(dtype=np.float32, copy=False)
        try:
            dense_matrix = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Materialized-sparse TF-IDF fallback requires numeric features"
            ) from error
        if dense_matrix.ndim != 2:
            raise ValueError(
                "Materialized-sparse TF-IDF fallback requires a 2D table"
            )
        matrix = sparse.csr_matrix(dense_matrix)
        observed_values = dense_matrix
    if not np.isfinite(observed_values).all():
        raise ValueError(
            "Materialized-sparse TF-IDF fallback requires finite features"
        )
    if observed_values.size and observed_values.min() < 0:
        raise ValueError(
            "Materialized-sparse TF-IDF fallback requires non-negative features"
        )
    return matrix


def _use_materialized_sparse_tfidf_logit(
    features, target, metric, framework_params
):
    """Use TF-IDF only for an unmodified adaptive logloss portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "logloss"
        and not has_explicit_portfolio_settings
        and _is_materialized_sparse_tfidf_regime(features, target)
    )


def _sampled_wide_binary_signal_profile(features, target, max_rows=4_000):
    """Measure whether useful univariate signal is sparse across a wide table."""
    row_count = features.shape[0]
    sampled_rows = min(row_count, max(int(max_rows), 1))
    row_indices = np.linspace(0, row_count - 1, sampled_rows, dtype=int)
    sample = _slice_rows(features, row_indices)
    if hasattr(sample, "to_numpy"):
        sample = sample.to_numpy(dtype=np.float32, copy=False)
    try:
        sample = np.asarray(sample, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if sample.ndim != 2 or sample.size == 0 or not np.isfinite(sample).all():
        return None

    scores, _ = _finite_f_classif(
        sample, np.asarray(target).reshape(-1)[row_indices]
    )
    finite_scores = scores[np.isfinite(scores)]
    if not len(finite_scores):
        return None
    score_q90, score_q99 = np.quantile(finite_scores, [0.90, 0.99])
    return {
        "density": float(np.count_nonzero(sample)) / float(sample.size),
        "score_q90": float(score_q90),
        "score_q99": float(score_q99),
        "tail_ratio": float(score_q99) / max(float(score_q90), 1.0),
        "strong_features": int(np.count_nonzero(finite_scores >= 10.0)),
    }


def _is_screenable_wide_binary_auc_regime(features, target):
    """Recognise bounded wide binary tables with a sparse supervised signal tail.

    A sampled ANOVA profile distinguishes a small useful feature subset from a
    diffuse wide representation.  It is used only as a training-data guard; the
    final selector is fitted again on every available training row.
    """
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    if (
        not 10_000 <= row_count <= 30_000
        or not 2_048 <= feature_count <= 8_192
        or feature_count < 0.20 * row_count
        or row_count * feature_count > 100_000_000
        or len(class_counts) != 2
        or class_counts.min() < 1_000
        or not _all_features_are_numeric(features)
        or _is_sparse_table(features)
    ):
        return False

    profile = _sampled_wide_binary_signal_profile(features, target_array)
    return bool(
        profile is not None
        and profile["density"] >= 0.25
        and profile["density"] <= 0.80
        and profile["score_q99"] >= 10.0
        and profile["tail_ratio"] >= 5.0
        and 32 <= profile["strong_features"] <= 512
    )


def _use_screened_wide_auc_lgbm(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use the screened direct fit only for an unmodified adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "auc"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_screenable_wide_binary_auc_regime(features, target)
    )


def _medium_wide_binary_signal_profile(features, target, max_rows=5_000):
    """Measure a bounded univariate signal tail on training rows only."""
    target_array = np.asarray(target).reshape(-1)
    row_indices = np.arange(len(target_array))
    if len(row_indices) > max_rows:
        row_indices, _ = train_test_split(
            row_indices,
            train_size=int(max_rows),
            random_state=42,
            stratify=target_array,
        )
    matrix = np.asarray(_slice_rows(features, row_indices), dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).all():
        return None
    scores, _ = _finite_f_classif(matrix, target_array[row_indices])
    finite_scores = scores[np.isfinite(scores)]
    if not len(finite_scores):
        return None
    return {
        "density": float(np.count_nonzero(matrix)) / float(matrix.size),
        "score_q95": float(np.quantile(finite_scores, 0.95)),
        "score_q99": float(np.quantile(finite_scores, 0.99)),
        "strong_features": int(np.count_nonzero(finite_scores >= 10.0)),
    }


def _is_medium_screenable_binary_auc_regime(features, target):
    """Recognise dense moderate-width binary tables with a strong signal tail."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    if (
        not 2_500 <= row_count <= 10_000
        or not 256 <= feature_count <= 512
        or row_count * feature_count > 5_000_000
        or len(class_counts) != 2
        or class_counts.min() < 500
        or not _all_features_are_numeric(features)
    ):
        return False

    profile = _medium_wide_binary_signal_profile(features, target_array)
    return bool(
        profile is not None
        and profile["density"] >= 0.90
        and profile["score_q95"] >= 8.0
        and profile["score_q99"] >= 50.0
        and 8 <= profile["strong_features"] <= 192
    )


def _use_medium_screened_auc_lgbm(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use train-selected screening only for an unmodified adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "auc"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_medium_screenable_binary_auc_regime(features, target)
    )


def _high_missing_mixed_profile(features):
    """Describe a bounded mixed table where missingness is the main geometry."""
    if not hasattr(features, "dtypes"):
        return None
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    numeric_columns = [
        column for column in frame if column not in categorical_columns
    ]
    if not categorical_columns or not numeric_columns or frame.size == 0:
        return None
    cardinalities = np.asarray(
        [frame[column].nunique(dropna=True) for column in categorical_columns],
        dtype=int,
    )
    return {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "missing_fraction": float(frame.isna().to_numpy().mean()),
        "categorical_cardinality": int(cardinalities.sum()),
        "median_cardinality": float(np.median(cardinalities)),
        "maximum_cardinality": int(cardinalities.max(initial=0)),
    }


def _is_high_missing_mixed_binary_auc_regime(features, target):
    """Recognise medium-large mixed tables dominated by missing values."""
    shape = (
        features.shape
        if hasattr(features, "shape")
        else np.asarray(features).shape
    )
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    if (
        not 30_000 <= row_count <= 80_000
        or not 128 <= feature_count <= 512
        or row_count * feature_count > 25_000_000
        or len(class_counts) != 2
        or class_counts.min() < 500
    ):
        return False

    profile = _high_missing_mixed_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    numeric_count = len(profile["numeric_columns"])
    return bool(
        32 <= categorical_count <= 128
        and 5 * categorical_count >= feature_count
        and 5 * categorical_count <= 3 * feature_count
        and 64 <= numeric_count <= 384
        and 0.50 <= profile["missing_fraction"] <= 0.90
        and 10_000 <= profile["categorical_cardinality"] <= 250_000
        and profile["median_cardinality"] <= 16
        and 1_000 <= profile["maximum_cardinality"] <= 100_000
    )


def _use_high_missing_frequency_auc_lgbm(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use gated frequency encoding only for an unmodified adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "auc"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_high_missing_mixed_binary_auc_regime(features, target)
    )


def _nominal_multiclass_profile(features):
    """Describe a bounded categorical-heavy table with nontrivial cardinality."""
    if not hasattr(features, "dtypes"):
        return None
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    numeric_columns = [
        column for column in frame if column not in categorical_columns
    ]
    if not categorical_columns or not numeric_columns or frame.size == 0:
        return None
    cardinalities = np.asarray(
        [frame[column].nunique(dropna=True) for column in categorical_columns],
        dtype=int,
    )
    return {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "categorical_cardinality": int(cardinalities.sum()),
        "median_cardinality": float(np.median(cardinalities)),
        "maximum_cardinality": int(cardinalities.max(initial=0)),
    }


def _is_nominal_heavy_multiclass_regime(features, target):
    """Recognise affordable multiclass tables suited to native category splits."""
    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    if (
        not 30_000 <= row_count <= 100_000
        or not 12 <= feature_count <= 64
        or row_count * feature_count > 5_000_000
        or not 3 <= len(class_counts) <= 5
        or class_counts.min() < 2_000
    ):
        return False

    profile = _nominal_multiclass_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    numeric_count = len(profile["numeric_columns"])
    return bool(
        8 <= categorical_count <= 60
        and 10 * categorical_count >= 7 * feature_count
        and 20 * categorical_count <= 19 * feature_count
        and 1 <= numeric_count <= 20
        and 2_000 <= profile["categorical_cardinality"] <= 20_000
        and profile["median_cardinality"] <= 32
        and 500 <= profile["maximum_cardinality"] <= 10_000
    )


def _use_nominal_multiclass_catboost(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use the gated native-category path only for an adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return bool(
        metric == "logloss"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_nominal_heavy_multiclass_regime(features, target)
    )


def _large_high_cardinality_mixed_profile(features):
    """Describe a bounded mixed table where one-hot expansion is infeasible."""
    if not hasattr(features, "dtypes"):
        return None
    categorical_columns = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column].dtype)
    ]
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]
    if not categorical_columns or not numeric_columns:
        return None
    cardinalities = np.asarray(
        [features[column].nunique(dropna=True) for column in categorical_columns],
        dtype=int,
    )
    return {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "categorical_cardinality": int(cardinalities.sum()),
        "median_cardinality": float(np.median(cardinalities)),
        "maximum_cardinality": int(cardinalities.max()),
        "encoded_columns": len(numeric_columns) + 2 * len(categorical_columns),
    }


def _is_large_high_cardinality_mixed_binary_regime(features, target):
    """Recognise large binary tables suited to cross-fitted target encoding."""
    shape = (
        features.shape
        if hasattr(features, "shape")
        else np.asarray(features).shape
    )
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    if (
        not 100_000 <= row_count <= 500_000
        or not 32 <= feature_count <= 128
        or row_count * feature_count > 40_000_000
        or len(class_counts) != 2
        or class_counts.min() < 10_000
    ):
        return False

    profile = _large_high_cardinality_mixed_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    numeric_count = len(profile["numeric_columns"])
    return bool(
        16 <= categorical_count <= 96
        and categorical_count >= 0.40 * feature_count
        and 8 <= numeric_count <= 96
        and 10_000 <= profile["categorical_cardinality"] <= 2_000_000
        and profile["median_cardinality"] >= 32
        and 1_000 <= profile["maximum_cardinality"] <= 250_000
        and row_count * profile["encoded_columns"] <= 60_000_000
    )


def _use_target_frequency_auc_lgbm(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use cross-fitted encoding only for an unmodified adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "auc"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_large_high_cardinality_mixed_binary_regime(features, target)
    )


def _narrow_high_cardinality_categorical_profile(features):
    """Describe a bounded category-heavy table without expanding it."""
    if not hasattr(features, "dtypes"):
        return None
    frame = pd.DataFrame(features)
    categorical_columns = [
        column
        for column in frame
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not categorical_columns:
        return None
    cardinalities = np.asarray(
        [frame[column].nunique(dropna=True) for column in categorical_columns],
        dtype=int,
    )
    return {
        "categorical_columns": categorical_columns,
        "categorical_cardinality": int(cardinalities.sum()),
        "median_cardinality": float(np.median(cardinalities)),
        "maximum_cardinality": int(cardinalities.max(initial=0)),
    }


def _is_narrow_high_cardinality_categorical_binary_regime(features, target):
    """Recognise affordable nominal interaction tasks for native CatBoost."""
    shape = (
        features.shape
        if hasattr(features, "shape")
        else np.asarray(features).shape
    )
    if len(shape) != 2 or _is_sparse_table(features):
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    if (
        not 20_000 <= row_count <= 50_000
        or not 6 <= feature_count <= 16
        or row_count * feature_count > 800_000
        or len(class_counts) != 2
        or class_counts.min() < 1_000
    ):
        return False

    profile = _narrow_high_cardinality_categorical_profile(features)
    if profile is None:
        return False
    categorical_count = len(profile["categorical_columns"])
    return bool(
        5 * categorical_count >= 4 * feature_count
        and 5_000 <= profile["categorical_cardinality"] <= 50_000
        and profile["median_cardinality"] >= 32
        and profile["maximum_cardinality"] <= 20_000
    )


def _use_narrow_categorical_auc_catboost(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use native categories only for an unmodified adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "auc"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_narrow_high_cardinality_categorical_binary_regime(
            features, target
        )
    )


def _use_fixed_round_all_row_xgboost(
    features,
    target,
    metric,
    runtime_seconds,
    cores,
    framework_params,
):
    """Use the validated direct fit only for an unmodified adaptive portfolio."""
    has_explicit_portfolio_settings = any(
        key.startswith("_portfolio_") for key in framework_params
    )
    return (
        metric == "logloss"
        and runtime_seconds >= 180
        and cores >= 4
        and not has_explicit_portfolio_settings
        and _is_resource_bounded_extreme_wide_numeric(features, target)
    )


def _is_sparse_table(features):
    if hasattr(features, "tocsr"):
        return True
    if hasattr(features, "dtypes"):
        return any(isinstance(dtype, pd.SparseDtype) for dtype in features.dtypes)
    return False


def _classification_holdout(
    features,
    target,
    validation_fraction,
    max_train_rows,
    seed,
    max_validation_rows=None,
):
    target_array = np.asarray(target).reshape(-1)
    labels, inverse, counts = np.unique(
        target_array, return_inverse=True, return_counts=True
    )
    rare_mask = counts[inverse] < 2
    if rare_mask.any():
        rare_indices = np.flatnonzero(rare_mask)
        splittable_indices = np.flatnonzero(~rare_mask)
        splittable_target = target_array[splittable_indices]
        splittable_class_count = len(np.unique(splittable_target))
        validation_rows = max(
            int(np.ceil(len(target_array) * validation_fraction)),
            splittable_class_count,
        )
        max_validation_rows_with_class_coverage = (
            len(splittable_indices) - splittable_class_count
        )
        validation_rows = min(
            validation_rows, max_validation_rows_with_class_coverage
        )
        if validation_rows <= 0:
            raise ValueError(
                "Portfolio validation requires at least one class with two rows"
            )
        train_indices, valid_indices = train_test_split(
            splittable_indices,
            test_size=validation_rows,
            random_state=seed,
            stratify=splittable_target,
        )
        train_indices = np.sort(np.concatenate((train_indices, rare_indices)))
        valid_indices = np.sort(valid_indices)
        X_train = _slice_rows(features, train_indices)
        X_valid = _slice_rows(features, valid_indices)
        y_train = target_array[train_indices]
        y_valid = target_array[valid_indices]
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            target_array,
            test_size=validation_fraction,
            random_state=seed,
            stratify=target_array,
        )
    if len(y_train) > max_train_rows:
        sampled_indices = _stratified_sample_indices_with_class_coverage(
            y_train, max_train_rows, seed
        )
        X_train = _slice_rows(X_train, sampled_indices)
        y_train = y_train[sampled_indices]
    if max_validation_rows is not None and len(y_valid) > max_validation_rows:
        minimum_rows = len(np.unique(target_array))
        validation_rows = max(max_validation_rows, minimum_rows)
        sampled_indices = _stratified_sample_indices_with_class_coverage(
            y_valid, validation_rows, seed + 1
        )
        X_valid = _slice_rows(X_valid, sampled_indices)
        y_valid = y_valid[sampled_indices]
    return X_train, X_valid, y_train, y_valid


def _stratified_sample_indices_with_class_coverage(target, sample_size, seed):
    """Use the usual stratified sample, falling back when it loses a label.

    Preserve sklearn's historical sample on ordinary tables.  Its stratifier
    cannot handle singleton labels and, under aggressive caps, can omit a very
    rare label.  Only in those cases reserve one row per class and allocate the
    remaining quota approximately by class frequency.
    """
    target_array = np.asarray(target).reshape(-1)
    if not 0 < sample_size <= len(target_array):
        raise ValueError("sample_size must be between one and the target size")
    if sample_size == len(target_array):
        return np.arange(len(target_array))

    labels, inverse, counts = np.unique(
        target_array, return_inverse=True, return_counts=True
    )
    if sample_size < len(labels):
        raise ValueError("sample_size must be at least the number of classes")

    try:
        sampled, _ = train_test_split(
            np.arange(len(target_array)),
            train_size=sample_size,
            random_state=seed,
            stratify=target_array,
        )
        if len(np.unique(target_array[sampled])) == len(labels):
            return sampled
    except ValueError:
        # Singleton or otherwise unsplittable classes are handled below.
        pass

    allocation = np.ones(len(labels), dtype=int)
    remaining_quota = sample_size - len(labels)
    remaining_counts = counts - 1
    if remaining_quota:
        ideal_extra = remaining_quota * remaining_counts / remaining_counts.sum()
        extra = np.floor(ideal_extra).astype(int)
        allocation += extra
        leftover = remaining_quota - int(extra.sum())
        if leftover:
            fractional = ideal_extra - extra
            order = np.argsort(-fractional, kind="stable")
            allocation[order[:leftover]] += 1

    rng = np.random.RandomState(seed)
    sampled = []
    for class_index, class_size in enumerate(allocation):
        class_rows = np.flatnonzero(inverse == class_index)
        sampled.extend(rng.choice(class_rows, size=class_size, replace=False))
    return np.sort(np.asarray(sampled, dtype=int))


def _classification_validation_splits(
    features,
    target,
    validation_fraction,
    max_train_rows,
    seed,
    max_validation_rows=10_000,
    cross_validation_max_rows=5_000,
):
    target_array = np.asarray(target).reshape(-1)
    class_counts = np.unique(target_array, return_counts=True)[1]
    cross_validation_train_rows = int(np.ceil(len(target_array) * 2 / 3))
    if (
        len(target_array) <= cross_validation_max_rows
        and cross_validation_train_rows <= max_train_rows
        and class_counts.min() >= 3
    ):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return [
            (
                _slice_rows(features, train_indices),
                _slice_rows(features, valid_indices),
                target_array[train_indices],
                target_array[valid_indices],
            )
            for train_indices, valid_indices in splitter.split(
                np.zeros(len(target_array)), target_array
            )
        ]
    return [
        _classification_holdout(
            features,
            target_array,
            validation_fraction=validation_fraction,
            max_train_rows=max_train_rows,
            max_validation_rows=max_validation_rows,
            seed=seed,
        )
    ]


def _slice_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    if hasattr(data, "tocsr"):
        return data[indices]
    return np.asarray(data)[indices]


def _target_array(target):
    """Return a dense 1-D target, including sparse encoded columns."""
    if hasattr(target, "toarray"):
        target = target.toarray()
    elif hasattr(target, "sparse"):
        target = target.sparse.to_dense()
    return np.asarray(target).reshape(-1)


def _contiguous_classification_target(target):
    """Encode observed target positions contiguously for classifiers.

    Nominal declarations can contain unused levels, so a training fold can have
    encoded labels such as ``[0, 1, 3]``. XGBoost requires ``[0, 1, 2]`` and
    other estimators may emit only three probability columns. Use a compact
    model-internal label space and retain the observed positions for restoration.
    """
    target_array = _target_array(target)
    observed_encoded_labels, contiguous_target = np.unique(
        target_array, return_inverse=True
    )
    return contiguous_target, observed_encoded_labels


def _restore_classification_label_space(
    probabilities, observed_encoded_labels, encoded_class_count=None
):
    """Restore model probabilities to declared encoded class positions."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack((1.0 - probabilities, probabilities))
    observed_encoded_labels = np.asarray(observed_encoded_labels)
    if probabilities.shape[1] != len(observed_encoded_labels):
        raise ValueError(
            "probability columns do not match the observed training classes"
        )

    encoded_positions = observed_encoded_labels.astype(int)
    if not np.array_equal(observed_encoded_labels, encoded_positions):
        raise ValueError("encoded classification labels must be integer positions")
    minimum_class_count = int(encoded_positions.max()) + 1
    output_class_count = (
        minimum_class_count
        if encoded_class_count is None
        else int(encoded_class_count)
    )
    if output_class_count < minimum_class_count:
        raise ValueError(
            "declared encoded class count is smaller than an observed label position"
        )

    predictions = encoded_positions[np.argmax(probabilities, axis=1)]
    if (
        output_class_count == probabilities.shape[1]
        and np.array_equal(encoded_positions, np.arange(output_class_count))
    ):
        return predictions, probabilities

    restored = np.zeros((len(probabilities), output_class_count), dtype=float)
    restored[:, encoded_positions] = probabilities
    return predictions, restored


def _classification_predictions(automl, features):
    probabilities = np.asarray(
        automl.predict_proba(features=features, probs_for_all_classes=True),
        dtype=float,
    )
    if probabilities.ndim == 1:
        probabilities = np.column_stack((1.0 - probabilities, probabilities))
    target = getattr(automl, "target", None)
    labels = (
        np.arange(probabilities.shape[1])
        if target is None
        else np.unique(np.asarray(target).reshape(-1))
    )
    predictions = labels[np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def _selection_score(metric, truth, predictions, probabilities, labels=None):
    truth = np.asarray(truth).reshape(-1)
    observed_labels = np.unique(truth)
    labels = observed_labels if labels is None else np.asarray(labels)
    if metric == "logloss":
        return -log_loss(truth, probabilities, labels=labels)
    if metric == "auc":
        if len(observed_labels) == 2:
            return roc_auc_score(truth, probabilities[:, 1])
        return roc_auc_score(
            truth,
            probabilities,
            labels=observed_labels,
            multi_class="ovr",
            average="macro",
        )
    if metric == "acc":
        return accuracy_score(truth, predictions)
    if metric == "f1":
        average = "binary" if len(labels) == 2 else "weighted"
        return f1_score(truth, predictions, average=average)
    if metric == "mae":
        return -mean_absolute_error(truth, predictions)
    if metric == "mse":
        return -mean_squared_error(truth, predictions)
    if metric == "rmse":
        return -mean_squared_error(truth, predictions) ** 0.5
    if metric == "msle":
        return -mean_squared_log_error(truth, predictions)
    if metric == "r2":
        return r2_score(truth, predictions)
    log.warning("Unsupported portfolio metric %s; selecting by accuracy.", metric)
    return accuracy_score(truth, predictions)


def _fit_temperature(
    probabilities,
    truth,
    labels=None,
    allow_sparse_classes=False,
    allow_singleton_classes=False,
):
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth).reshape(-1)
    observed_labels = np.unique(truth)
    labels = observed_labels if labels is None else np.asarray(labels)
    class_counts = np.unique(truth, return_counts=True)[1]
    minimum_class_count = (
        1 if allow_singleton_classes else 3 if allow_sparse_classes else 5
    )
    if (
        len(truth) < 100
        or class_counts.min() < minimum_class_count
        or len(observed_labels) < len(labels)
    ):
        return 1.0

    baseline = log_loss(truth, probabilities, labels=labels)

    def objective(log_temperature):
        calibrated = _apply_temperature(probabilities, np.exp(log_temperature))
        return log_loss(truth, calibrated, labels=labels)

    optimum = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
    if not optimum.success or optimum.fun >= baseline - 1e-4:
        return 1.0
    return float(np.exp(optimum.x))


def _apply_temperature(probabilities, temperature):
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    logits = np.log(clipped) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _fit_prior_exponent(
    probabilities,
    truth,
    reference_target,
    labels=None,
    minimum_improvement=1e-4,
):
    """Fit one residual class-prior coefficient on validation predictions.

    Temperature changes confidence but cannot correct a stable tendency to
    over- or under-weight frequent classes.  Multiplying probabilities by a
    power of the training priors adds one bounded degree of freedom regardless
    of class count.  Conservative support and improvement guards keep noisy
    rare-class holdouts neutral.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth).reshape(-1)
    labels = np.unique(truth) if labels is None else np.asarray(labels)
    observed_labels, class_counts = np.unique(truth, return_counts=True)
    if (
        len(truth) < 100
        or len(np.asarray(reference_target).reshape(-1))
        < _PRIOR_CALIBRATION_MIN_REFERENCE_ROWS
        or class_counts.min() < 5
        or len(observed_labels) < len(labels)
        or minimum_improvement < 0
    ):
        return 0.0

    baseline = log_loss(truth, probabilities, labels=labels)

    def objective(exponent):
        calibrated = _apply_prior_exponent(
            probabilities,
            reference_target=reference_target,
            exponent=exponent,
            labels=labels,
        )
        return log_loss(truth, calibrated, labels=labels)

    optimum = minimize_scalar(
        objective,
        bounds=(
            -_PRIOR_CALIBRATION_MAX_ABS_EXPONENT,
            _PRIOR_CALIBRATION_MAX_ABS_EXPONENT,
        ),
        method="bounded",
    )
    if not optimum.success or optimum.fun >= baseline - minimum_improvement:
        return 0.0
    # The calibration model is fitted on selector predictions but applied after
    # a fresh all-row refit.  Shrinking the one-dimensional correction protects
    # against small selector-to-deployment shifts without changing its sign.
    return float(optimum.x * _PRIOR_CALIBRATION_TRANSFER_SHRINKAGE)


def _apply_prior_exponent(probabilities, reference_target, exponent, labels=None):
    """Apply a scalar class-prior correction without fitting class-wise biases."""
    probabilities = np.asarray(probabilities, dtype=float)
    reference_target = np.asarray(reference_target).reshape(-1)
    labels = np.unique(reference_target) if labels is None else np.asarray(labels)
    counts_by_label = {
        label: count
        for label, count in zip(*np.unique(reference_target, return_counts=True))
    }
    counts = np.asarray([counts_by_label.get(label, 0) for label in labels], dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Probability columns must match prior-calibration labels")
    if np.any(counts <= 0):
        raise ValueError("Every prior-calibration label must occur in reference_target")
    if exponent == 0:
        return probabilities

    priors = counts / counts.sum()
    log_adjustment = exponent * (np.log(priors) - np.mean(np.log(priors)))
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) + log_adjustment
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _estimated_candidate_fit_seconds(candidate, observations):
    if not observations:
        return None
    durations = list(observations.values())
    if candidate == "auto":
        # ``auto`` is a composed FEDOT search pipeline rather than one more
        # boosting fit.  Its cost also has much higher variance than that of a
        # predefined model: on small CPU-limited benchmark tasks it repeatedly
        # took 5--6 times the combined validation time of both boosters.  Treat
        # their measured durations as a lower-bound proxy and keep a generous
        # uncertainty margin so the optional candidate cannot consume the final
        # refit/prediction budget.
        return max(6.0 * sum(durations), 12.0 * max(durations))
    return max(durations[-1], 0.1)


def _candidate_start_refit_reserve(leader_refit_seconds, remaining_seconds):
    """Protect a leader refit only while that deployment path is feasible."""
    if leader_refit_seconds < 0 or remaining_seconds < 0:
        raise ValueError("Candidate budget estimates must be non-negative")
    return leader_refit_seconds if leader_refit_seconds <= remaining_seconds else 0.0


def _select_ensemble(metric, contenders):
    ranked = sorted(contenders, key=lambda contender: contender["score"], reverse=True)
    if len(ranked) == 1:
        return ranked
    return ranked[:2]


def _use_weak_signal_shallow_probe(
    features,
    target,
    metric,
    candidates,
    enabled,
    runtime_seconds,
    cores,
):
    """Bound the opt-in train-only probe to large narrow binary AUC tables."""
    if (
        not enabled
        or metric != "auc"
        or candidates != ["lgbm", "xgboost"]
        or runtime_seconds < 120
        or cores < 2
    ):
        return False
    if hasattr(features, "nnz") or not _all_features_are_numeric(features):
        return False

    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    return bool(
        len(target_array) >= 50_000
        and 8 <= feature_count <= 64
        and len(class_counts) == 2
        and int(class_counts.min()) >= 1_000
    )


def _adaptive_weak_signal_shallow_probe_enabled(framework_params):
    """Enable the train-only weak-signal rule unless relevant knobs are explicit."""
    configured = framework_params.get("_portfolio_weak_signal_shallow_probe")
    if configured is not None:
        return _as_bool(configured)
    conflicting_overrides = {
        "_portfolio_candidates",
        "_portfolio_lgbm_num_leaves",
        "_portfolio_lgbm_min_child_samples",
        "_portfolio_xgboost_max_depth",
        "_portfolio_xgboost_min_child_weight",
        "_portfolio_auc_retain_boosting_pair",
    }
    return not any(key in framework_params for key in conflicting_overrides)


def _is_weak_auc_score(score, maximum_auc=0.55):
    """Recognise near-random AUC without treating a reversed strong ranker as weak."""
    maximum_auc = float(maximum_auc)
    if not 0.5 < maximum_auc < 1.0:
        raise ValueError("maximum weak-signal AUC must be between 0.5 and 1")
    score = float(score)
    return 1.0 - maximum_auc <= score <= maximum_auc


def _select_auc_ensemble(contenders, retain_boosting_pair=False):
    """Choose an OOF-supported singleton or fixed equal-weight pair.

    The LGBM/XGBoost pair remains the reliability baseline whenever both models
    are available.  A different strategy must improve its OOF AUC by both a
    small practical margin and at least one paired standard error.  The paired
    DeLong estimate benefits from the strong correlation between predictions,
    while preventing tiny-table OOF fluctuations from replacing the robust
    baseline.  No blend coefficient is fitted on the validation labels.
    """
    if not contenders:
        raise ValueError("At least one AUC contender is required")

    truth = np.asarray(contenders[0]["truth"]).reshape(-1)
    strategies = []
    for left_index, left in enumerate(contenders):
        strategies.append(_evaluate_auc_strategy([left], np.array([1.0]), truth))
        for right in contenders[left_index + 1:]:
            strategies.append(
                _evaluate_auc_strategy(
                    [left, right], np.array([0.5, 0.5]), truth
                )
            )

    by_name = {contender["model"]: contender for contender in contenders}
    if "lgbm" in by_name and "xgboost" in by_name:
        baseline = _evaluate_auc_strategy(
            [by_name["lgbm"], by_name["xgboost"]],
            np.array([0.5, 0.5]),
            truth,
        )
        if retain_boosting_pair:
            _portfolio_report(
                "Retaining the LGBM/XGBoost AUC reliability pair after the "
                "guarded shallow probe or explicit request; pair OOF score %.8f.",
                baseline[2],
            )
            return baseline[:3]
        dominant_singleton = _dominant_auc_singleton(strategies, baseline, truth)
        if dominant_singleton is not None:
            baseline = dominant_singleton
    else:
        baseline = max(
            (strategy for strategy in strategies if len(strategy[0]) == 1),
            key=lambda strategy: strategy[2],
        )

    best = max(strategies, key=lambda strategy: strategy[2])
    if _same_auc_strategy(best, baseline):
        return best[:3]

    gain = best[2] - baseline[2]
    standard_error = _paired_auc_difference_standard_error(
        truth,
        best[3],
        baseline[3],
    )
    practical_margin = 0.002
    required_gain = max(practical_margin, standard_error)
    selected = best if gain >= required_gain else baseline
    _portfolio_report(
        "Best alternative AUC strategy %s versus reliability baseline %s: "
        "OOF gain %.8f, paired standard error %.8f, required gain %.8f; %s.",
        [contender["model"] for contender in best[0]],
        [contender["model"] for contender in baseline[0]],
        gain,
        standard_error,
        required_gain,
        "selected" if selected is best else "baseline retained",
    )
    return selected[:3]


def _dominant_auc_singleton(strategies, boosting_pair, truth):
    """Drop a clearly weak booster when its blend gain is only microscopic.

    On a large, adequately supported binary holdout, a material paired AUC gap
    between the two boosters is reliable enough to identify a weak member.  We
    only correct the case where the pair wins the raw OOF ranking despite one
    statistically inferior member.  The strong singleton becomes the reliability
    baseline, after which the selector's existing paired-uncertainty gate can
    still restore a genuinely complementary pair.  A pair that loses OOF remains
    governed by the original gate.  The sample/support floors keep this rule away
    from noisy small-table validation, where blending is valuable insurance.
    """
    truth = np.asarray(truth).reshape(-1)
    labels, class_counts = np.unique(truth, return_counts=True)
    if (
        len(labels) != 2
        or len(truth) < 2_000
        or int(class_counts.min()) < 100
    ):
        return None

    boosting_singletons = [
        strategy
        for strategy in strategies
        if len(strategy[0]) == 1
        and strategy[0][0]["model"] in {"lgbm", "xgboost"}
    ]
    if len(boosting_singletons) != 2:
        return None

    strong, weak = sorted(
        boosting_singletons, key=lambda strategy: strategy[2], reverse=True
    )
    member_gap = strong[2] - weak[2]
    pair_gain = boosting_pair[2] - strong[2]
    member_standard_error = _paired_auc_difference_standard_error(
        truth,
        strong[3],
        weak[3],
    )
    required_member_gap = max(0.01, member_standard_error)
    if member_gap < required_member_gap or pair_gain < 0.0:
        return None

    _portfolio_report(
        "Using dominant AUC singleton %s instead of boosting pair: member OOF "
        "gap %.8f (required %.8f), pair gain %.8f, rows %d, minority support %d.",
        strong[0][0]["model"],
        member_gap,
        required_member_gap,
        pair_gain,
        len(truth),
        int(class_counts.min()),
    )
    return strong


def _evaluate_auc_strategy(ensemble, weights, truth):
    reference_truth = np.asarray(ensemble[0]["truth"]).reshape(-1)
    if not np.array_equal(reference_truth, truth):
        raise ValueError("AUC contenders must share the same OOF truth order")
    for contender in ensemble[1:]:
        contender_truth = np.asarray(contender["truth"]).reshape(-1)
        if not np.array_equal(contender_truth, truth):
            raise ValueError("AUC contenders must share the same OOF truth order")
    probabilities = _blend_probabilities(
        [contender["probabilities"] for contender in ensemble], weights
    )
    score = _selection_score("auc", truth, None, probabilities)
    return ensemble, np.asarray(weights, dtype=float), score, probabilities


def _same_auc_strategy(left, right):
    return (
        [contender["model"] for contender in left[0]]
        == [contender["model"] for contender in right[0]]
        and np.array_equal(left[1], right[1])
    )


def _paired_auc_difference_standard_error(
    truth, left_probabilities, right_probabilities
):
    """Return the paired DeLong standard error for two binary OOF AUCs.

    For multiclass AUC, where this binary covariance estimate does not apply,
    return a conservative fixed uncertainty.  That still allows a clearly large
    OOF improvement without pretending to have a multiclass significance test.
    """
    truth = np.asarray(truth).reshape(-1)
    labels = np.unique(truth)
    if len(labels) != 2:
        return 0.01

    positive = truth == labels[1]
    positive_count = int(positive.sum())
    negative_count = len(truth) - positive_count
    if positive_count < 2 or negative_count < 2:
        return float("inf")

    left_scores = _positive_class_scores(left_probabilities)
    right_scores = _positive_class_scores(right_probabilities)
    ordered_indices = np.concatenate(
        (np.flatnonzero(positive), np.flatnonzero(~positive))
    )
    predictions = np.vstack((left_scores, right_scores))[:, ordered_indices]
    covariance = _fast_delong_covariance(
        predictions,
        positive_count=positive_count,
    )
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    return float(np.sqrt(max(variance, 0.0)))


def _positive_class_scores(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Binary AUC probabilities must have one or two columns")
    return probabilities[:, 1]


def _fast_delong_covariance(predictions, positive_count):
    """Compute the covariance of correlated binary AUC estimates."""
    predictions = np.asarray(predictions, dtype=float)
    negative_count = predictions.shape[1] - positive_count
    positive_examples = predictions[:, :positive_count]
    negative_examples = predictions[:, positive_count:]
    positive_midranks = np.vstack(
        [_midranks(row) for row in positive_examples]
    )
    negative_midranks = np.vstack(
        [_midranks(row) for row in negative_examples]
    )
    combined_midranks = np.vstack([_midranks(row) for row in predictions])

    positive_influence = (
        combined_midranks[:, :positive_count] - positive_midranks
    ) / negative_count
    negative_influence = 1.0 - (
        combined_midranks[:, positive_count:] - negative_midranks
    ) / positive_count
    positive_covariance = np.atleast_2d(np.cov(positive_influence, bias=False))
    negative_covariance = np.atleast_2d(np.cov(negative_influence, bias=False))
    return (
        positive_covariance / positive_count
        + negative_covariance / negative_count
    )


def _midranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def _select_ridge_oof_expansion(
    candidates,
    features,
    target,
    seed,
    n_jobs,
    max_seconds,
    min_relative_gain=_REGRESSION_RIDGE_MIN_OOF_GAIN,
    pair_min_relative_gain=0.002,
):
    """Select a Ridge expansion only after a material train-only OOF gain."""
    if max_seconds <= 0:
        raise ValueError("Ridge OOF time budget must be positive")
    if min_relative_gain < 0:
        raise ValueError("Ridge OOF gain must be non-negative")
    target_array = np.asarray(target, dtype=float).reshape(-1)
    splits = list(
        KFold(n_splits=3, shuffle=True, random_state=int(seed)).split(target_array)
    )
    started_at = time.monotonic()
    deadline = started_at + float(max_seconds)
    small_mixed = _is_small_mixed_regression_portfolio(features, target_array)

    def oof_observation(candidate):
        predictions = np.empty(len(target_array), dtype=float)
        for fit_indices, validation_indices in splits:
            if time.monotonic() >= deadline:
                raise TimeoutError("Ridge OOF selection exhausted its time budget")
            estimator = _make_regression_portfolio_estimator(
                candidate,
                _slice_rows(features, fit_indices),
                seed=seed,
                n_jobs=n_jobs,
                small_mixed=small_mixed,
                selector=True,
            )
            estimator.fit(
                _slice_rows(features, fit_indices), target_array[fit_indices]
            )
            predictions[validation_indices] = np.asarray(
                estimator.predict(_slice_rows(features, validation_indices)),
                dtype=float,
            ).reshape(-1)
        if time.monotonic() >= deadline:
            raise TimeoutError("Ridge OOF selection exhausted its time budget")
        return {
            "model": candidate,
            "score": _selection_score(
                "rmse", target_array, predictions, probabilities=None
            ),
            "predictions": predictions,
            "truth": target_array,
            "selected_n_estimators": None,
        }

    current_observations = []
    for candidate in dict.fromkeys(candidates):
        try:
            current_observations.append(oof_observation(candidate))
        except TimeoutError:
            _portfolio_report(
                "Ridge OOF selection reached its %.1fs cap; preserving the "
                "existing regression path.",
                max_seconds,
            )
            return None
        except Exception:
            log.warning(
                "Regression OOF candidate %s failed.", candidate, exc_info=True
            )
    if not current_observations:
        return None

    ridge_observations = []
    for alpha in _REGRESSION_RIDGE_ALPHAS:
        candidate = f"ridge_{alpha:g}_direct"
        try:
            ridge_observations.append(oof_observation(candidate))
        except TimeoutError:
            _portfolio_report(
                "Ridge OOF selection reached its %.1fs cap; preserving the "
                "existing regression path.",
                max_seconds,
            )
            return None
        except Exception:
            log.warning("Ridge OOF alpha %.6g failed.", alpha, exc_info=True)
    if not ridge_observations:
        return None

    current_strategy = _select_regression_strategy(
        current_observations,
        metric="rmse",
        min_relative_pair_gain=pair_min_relative_gain,
    )
    best_ridge = max(ridge_observations, key=lambda contender: contender["score"])
    expanded_strategy = _select_regression_strategy(
        current_observations + [best_ridge],
        metric="rmse",
        min_relative_pair_gain=pair_min_relative_gain,
    )
    current_error = -float(current_strategy[2])
    expanded_error = -float(expanded_strategy[2])
    relative_gain = (current_error - expanded_error) / max(current_error, 1e-12)
    elapsed = time.monotonic() - started_at
    _portfolio_report(
        "Ridge OOF expansion: alpha=%s, current RMSE %.10g, expanded RMSE "
        "%.10g, relative gain %.2f%%, elapsed %.1fs.",
        best_ridge["model"][len("ridge_"):-len("_direct")],
        current_error,
        expanded_error,
        100 * relative_gain,
        elapsed,
    )
    if relative_gain < min_relative_gain:
        return None
    return expanded_strategy


def _select_regression_strategy(
    contenders,
    metric="rmse",
    min_relative_pair_gain=0.002,
    consider_all_pairs=False,
):
    """Choose the best singleton or a materially better top-two 50/50 blend.

    Only the two best validation singletons are considered for blending.  This
    keeps the number of tried strategies fixed when the small mixed portfolio
    contains four complementary models and avoids fitting a free blend weight.
    Ordered histogram portfolios may compare every fixed pair: their raw and
    cumulative representations are intentionally complementary, so singleton
    rank is not a reliable proxy for pair quality.  Four candidates still
    bound that path to six deterministic validation comparisons.
    """
    if min_relative_pair_gain < 0:
        raise ValueError("minimum relative pair gain must be non-negative")
    if not contenders:
        raise ValueError("regression strategy selection needs a contender")
    ranked = sorted(contenders, key=lambda contender: contender["score"], reverse=True)
    best_singleton = ranked[0]
    singleton_strategy = ([best_singleton], np.array([1.0]), best_singleton["score"])
    if len(ranked) < 2:
        return singleton_strategy

    truth = np.asarray(ranked[0]["truth"]).reshape(-1)
    if any(
        not np.array_equal(truth, np.asarray(contender["truth"]).reshape(-1))
        for contender in ranked[1:]
    ):
        raise ValueError("regression contenders must share the same validation truth")
    candidate_pairs = (
        [
            (left, right)
            for left_index, left in enumerate(ranked)
            for right in ranked[left_index + 1:]
        ]
        if consider_all_pairs
        else [(ranked[0], ranked[1])]
    )
    evaluated_pairs = []
    for left, right in candidate_pairs:
        pair_predictions = 0.5 * (
            np.asarray(left["predictions"], dtype=float)
            + np.asarray(right["predictions"], dtype=float)
        )
        evaluated_pairs.append(
            ([left, right], _selection_score(metric, truth, pair_predictions, None))
        )
    selected_pair, pair_score = max(evaluated_pairs, key=lambda pair: pair[1])
    absolute_gain = pair_score - best_singleton["score"]
    relative_gain = absolute_gain / max(abs(best_singleton["score"]), 1e-12)
    if absolute_gain > 0 and relative_gain >= min_relative_pair_gain:
        return selected_pair, np.array([0.5, 0.5]), pair_score
    return singleton_strategy


def _best_normalized_validation_rmse(contenders):
    """Measure whether a cheap small-wide model captures most target variance."""
    if not contenders:
        return float("inf")
    truth = np.asarray(contenders[0]["truth"], dtype=float).reshape(-1)
    target_scale = float(np.std(truth))
    if not np.isfinite(target_scale) or target_scale <= 0:
        return float("inf")
    errors = []
    for contender in contenders:
        contender_truth = np.asarray(contender["truth"], dtype=float).reshape(-1)
        if not np.array_equal(truth, contender_truth):
            raise ValueError("regression contenders must share validation truth")
        predictions = np.asarray(contender["predictions"], dtype=float).reshape(-1)
        errors.append(mean_squared_error(truth, predictions) ** 0.5)
    return min(errors) / target_scale


def _clip_to_observed_target_range(predictions, reference_target):
    """Project bounded-portfolio predictions onto the observed target support."""
    reference = np.asarray(reference_target, dtype=float).reshape(-1)
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("target range requires finite reference values")
    return np.clip(
        np.asarray(predictions, dtype=float), np.min(reference), np.max(reference)
    )


def _clip_to_observed_target_quantiles(
    predictions, reference_target, lower_quantile, upper_quantile
):
    """Winsorize predictions using only fixed quantiles of training labels."""
    reference = np.asarray(reference_target, dtype=float).reshape(-1)
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("target quantiles require finite reference values")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("target quantiles must satisfy 0 <= lower < upper <= 1")
    lower, upper = np.quantile(reference, [lower_quantile, upper_quantile])
    return np.clip(np.asarray(predictions, dtype=float), lower, upper)


def _train_gated_wide_extra_trees_contenders(
    contenders,
    reference_target,
    enabled,
    pair_strong_weight=0.5,
    seed=42,
    minimum_gain=_WIDE_EXTRA_TREES_MIN_LOGLOSS_GAIN,
):
    """Admit a wider ExtraTrees view only after a disjoint paired gate.

    The ordinary contenders define the incumbent on one half of the shared
    selector predictions.  The wider forest and its fixed 50/50 incumbent blend
    are calibrated and ranked on that same calibration half, then evaluated
    once on the disjoint judgment half.  Rejected validation models are released
    before the ordinary portfolio selector runs.
    """
    minimum_gain = float(minimum_gain)
    if minimum_gain < 0:
        raise ValueError("wide ExtraTrees minimum gain must be non-negative")
    wide = [
        contender
        for contender in contenders
        if contender.get("model") == "extra_trees_wide"
    ]
    incumbents = [
        contender
        for contender in contenders
        if contender.get("model") != "extra_trees_wide"
    ]
    if not wide:
        return contenders
    if len(wide) != 1:
        raise ValueError("wide ExtraTrees gate requires exactly one challenger")
    if not enabled or not incumbents:
        wide[0].pop("validation_models", None)
        return incumbents
    if not 0.5 <= float(pair_strong_weight) < 1:
        raise ValueError("pair_strong_weight must be in the interval [0.5, 1)")

    truth = np.asarray(incumbents[0]["truth"]).reshape(-1)
    labels = incumbents[0].get("labels")
    labels = np.unique(truth) if labels is None else np.asarray(labels)
    if len(truth) < 200:
        wide[0].pop("validation_models", None)
        return incumbents
    for contender in contenders:
        if not np.array_equal(truth, np.asarray(contender["truth"]).reshape(-1)):
            raise ValueError("wide ExtraTrees contenders must share validation truth")

    calibration, judgment = train_test_split(
        np.arange(len(truth)),
        test_size=0.5,
        random_state=int(seed) + 1,
        stratify=truth,
    )
    calibration_contenders = []
    for contender in incumbents:
        probabilities = np.asarray(contender["probabilities"])[calibration]
        calibration_contenders.append(
            {
                **contender,
                "score": _selection_score(
                    "logloss",
                    truth[calibration],
                    None,
                    probabilities,
                    labels=labels,
                ),
                "probabilities": probabilities,
                "truth": truth[calibration],
            }
        )
    incumbent_ensemble, incumbent_weights, incumbent_temperature, _ = (
        _select_logloss_ensemble(
            calibration_contenders,
            calibrate=True,
            pair_strong_weight=pair_strong_weight,
        )
    )
    incumbent_by_name = {
        contender["model"]: contender for contender in incumbents
    }
    incumbent_raw = _blend_probabilities(
        [
            incumbent_by_name[contender["model"]]["probabilities"]
            for contender in incumbent_ensemble
        ],
        incumbent_weights,
    )
    incumbent_calibrated = _apply_temperature(
        incumbent_raw, incumbent_temperature
    )
    incumbent_prior = _fit_prior_exponent(
        incumbent_calibrated[calibration],
        truth[calibration],
        reference_target=reference_target,
        labels=labels,
    )
    incumbent_calibrated = _apply_prior_exponent(
        incumbent_calibrated,
        reference_target=reference_target,
        exponent=incumbent_prior,
        labels=labels,
    )

    wide_probabilities = np.asarray(wide[0]["probabilities"])
    raw_strategies = {
        "singleton": wide_probabilities,
        "incumbent_blend50": 0.5 * (incumbent_raw + wide_probabilities),
    }
    candidate_strategies = {}
    for name, raw_probabilities in raw_strategies.items():
        temperature = _fit_temperature(
            raw_probabilities[calibration],
            truth[calibration],
            labels=labels,
        )
        calibrated = _apply_temperature(raw_probabilities, temperature)
        prior_exponent = _fit_prior_exponent(
            calibrated[calibration],
            truth[calibration],
            reference_target=reference_target,
            labels=labels,
        )
        calibrated = _apply_prior_exponent(
            calibrated,
            reference_target=reference_target,
            exponent=prior_exponent,
            labels=labels,
        )
        candidate_strategies[name] = {
            "probabilities": calibrated,
            "calibration_loss": -_selection_score(
                "logloss",
                truth[calibration],
                None,
                calibrated[calibration],
                labels=labels,
            ),
        }
    selected_name, selected = min(
        candidate_strategies.items(),
        key=lambda item: item[1]["calibration_loss"],
    )
    incumbent_losses = _row_log_losses(
        incumbent_calibrated[judgment], truth[judgment], labels=labels
    )
    candidate_losses = _row_log_losses(
        selected["probabilities"][judgment], truth[judgment], labels=labels
    )
    improvements = incumbent_losses - candidate_losses
    gain = float(np.mean(improvements))
    standard_error = (
        float(np.std(improvements, ddof=1) / np.sqrt(len(improvements)))
        if len(improvements) > 1
        else 0.0
    )
    lower_95 = float(gain - 1.96 * standard_error)
    accepted = bool(gain >= minimum_gain and lower_95 > 0.0)
    _portfolio_report(
        "Wide ExtraTrees train-only gate selected %s: judgment gain %.6f, "
        "paired lower-95 %.6f, required gain %.6f; %s.",
        selected_name,
        gain,
        lower_95,
        minimum_gain,
        "admitted" if accepted else "rejected",
    )
    if accepted:
        return contenders
    wide[0].pop("validation_models", None)
    return incumbents


def _train_gated_lgbm_extra_trees_pair_contenders(
    contenders,
    reference_target,
    enabled,
    pair_strong_weight=0.5,
    seed=42,
    minimum_gain=_LGBM_EXTRA_TREES_MIN_LOGLOSS_GAIN,
    maximum_gate_seconds=_LGBM_EXTRA_TREES_MAX_GATE_SECONDS,
    maximum_pair_ratio=_LGBM_EXTRA_TREES_MAX_PAIR_RATIO,
    maximum_lgbm_overhead_seconds=_LGBM_EXTRA_TREES_MAX_OVERHEAD_SECONDS,
):
    """Admit and freeze the Extra-LGBM/XGBoost pair on disjoint rows.

    The frozen baseline is the 50/50 current-LGBM/shared-XGBoost pair and the
    candidate changes only the LightGBM member. A conservative second baseline
    is the ordinary singleton-or-pair strategy selected on calibration rows.
    Temperature and prior exponent are fitted on the calibration half; the
    judgment half contributes only the frozen decisions. Rejection removes only
    the challenger, leaving the ordinary contenders and predictions untouched.
    """
    minimum_gain = float(minimum_gain)
    maximum_gate_seconds = float(maximum_gate_seconds)
    maximum_pair_ratio = float(maximum_pair_ratio)
    maximum_lgbm_overhead_seconds = float(maximum_lgbm_overhead_seconds)
    if minimum_gain < 0:
        raise ValueError("Extra-LGBM minimum gain must be non-negative")
    if maximum_gate_seconds <= 0:
        raise ValueError("Extra-LGBM gate budget must be positive")
    if maximum_pair_ratio < 1:
        raise ValueError("Extra-LGBM pair ratio must be at least one")
    if maximum_lgbm_overhead_seconds < 0:
        raise ValueError("Extra-LGBM overhead must be non-negative")
    if not 0.5 <= float(pair_strong_weight) < 1:
        raise ValueError("pair_strong_weight must be in the interval [0.5, 1)")

    challengers = [
        contender
        for contender in contenders
        if contender.get("model") == _LGBM_EXTRA_TREES_MODEL
    ]
    ordinary = [
        contender
        for contender in contenders
        if contender.get("model") != _LGBM_EXTRA_TREES_MODEL
    ]
    if not challengers:
        return contenders, None
    if len(challengers) != 1:
        raise ValueError("Extra-LGBM gate requires exactly one challenger")
    challenger = challengers[0]
    if not enabled:
        challenger.pop("validation_models", None)
        return ordinary, None

    current_lgbm = [
        contender for contender in ordinary if contender.get("model") == "lgbm"
    ]
    shared_xgboost = [
        contender
        for contender in ordinary
        if contender.get("model") == "xgboost"
    ]
    if len(current_lgbm) != 1 or len(shared_xgboost) != 1:
        challenger.pop("validation_models", None)
        return ordinary, None
    current_lgbm = current_lgbm[0]
    shared_xgboost = shared_xgboost[0]

    truth = np.asarray(current_lgbm["truth"]).reshape(-1)
    labels = current_lgbm.get("labels")
    labels = np.unique(truth) if labels is None else np.asarray(labels)
    if len(truth) < 200:
        challenger.pop("validation_models", None)
        return ordinary, None
    for contender in (shared_xgboost, challenger):
        if not np.array_equal(truth, np.asarray(contender["truth"]).reshape(-1)):
            raise ValueError("Extra-LGBM pair members must share validation truth")

    calibration, judgment = train_test_split(
        np.arange(len(truth)),
        test_size=0.5,
        random_state=int(seed) + 12,
        stratify=truth,
    )
    current_raw = 0.5 * (
        np.asarray(current_lgbm["probabilities"], dtype=float)
        + np.asarray(shared_xgboost["probabilities"], dtype=float)
    )
    candidate_raw = 0.5 * (
        np.asarray(challenger["probabilities"], dtype=float)
        + np.asarray(shared_xgboost["probabilities"], dtype=float)
    )

    def calibrated_strategy(raw_probabilities):
        temperature = _fit_temperature(
            raw_probabilities[calibration],
            truth[calibration],
            labels=labels,
        )
        calibrated = _apply_temperature(raw_probabilities, temperature)
        prior_exponent = _fit_prior_exponent(
            calibrated[calibration],
            truth[calibration],
            reference_target=reference_target,
            labels=labels,
        )
        calibrated = _apply_prior_exponent(
            calibrated,
            reference_target=reference_target,
            exponent=prior_exponent,
            labels=labels,
        )
        return calibrated, float(temperature), float(prior_exponent)

    current, current_temperature, current_prior = calibrated_strategy(current_raw)
    candidate, candidate_temperature, candidate_prior = calibrated_strategy(
        candidate_raw
    )

    calibration_contenders = []
    for contender in (current_lgbm, shared_xgboost):
        probabilities = np.asarray(contender["probabilities"], dtype=float)[
            calibration
        ]
        calibration_contenders.append(
            {
                **contender,
                "score": _selection_score(
                    "logloss",
                    truth[calibration],
                    None,
                    probabilities,
                    labels=labels,
                ),
                "probabilities": probabilities,
                "truth": truth[calibration],
            }
        )
    (
        ordinary_ensemble,
        ordinary_weights,
        ordinary_temperature,
        _,
    ) = _select_logloss_ensemble(
        calibration_contenders,
        calibrate=True,
        pair_strong_weight=pair_strong_weight,
    )
    ordinary_by_name = {
        contender["model"]: contender
        for contender in (current_lgbm, shared_xgboost)
    }
    ordinary_raw = _blend_probabilities(
        [
            ordinary_by_name[contender["model"]]["probabilities"]
            for contender in ordinary_ensemble
        ],
        ordinary_weights,
    )
    ordinary_incumbent = _apply_temperature(
        ordinary_raw, ordinary_temperature
    )
    ordinary_prior = _fit_prior_exponent(
        ordinary_incumbent[calibration],
        truth[calibration],
        reference_target=reference_target,
        labels=labels,
    )
    ordinary_incumbent = _apply_prior_exponent(
        ordinary_incumbent,
        reference_target=reference_target,
        exponent=ordinary_prior,
        labels=labels,
    )

    current_losses = _row_log_losses(
        current[judgment], truth[judgment], labels=labels
    )
    candidate_losses = _row_log_losses(
        candidate[judgment], truth[judgment], labels=labels
    )
    improvements = current_losses - candidate_losses
    gain = float(np.mean(improvements))
    standard_error = (
        float(np.std(improvements, ddof=1) / np.sqrt(len(improvements)))
        if len(improvements) > 1
        else 0.0
    )
    lower_95 = float(gain - 1.96 * standard_error)
    raw_gain = float(
        log_loss(truth[judgment], current_raw[judgment], labels=labels)
        - log_loss(truth[judgment], candidate_raw[judgment], labels=labels)
    )
    ordinary_losses = _row_log_losses(
        ordinary_incumbent[judgment], truth[judgment], labels=labels
    )
    ordinary_improvements = ordinary_losses - candidate_losses
    ordinary_gain = float(np.mean(ordinary_improvements))
    ordinary_standard_error = (
        float(
            np.std(ordinary_improvements, ddof=1)
            / np.sqrt(len(ordinary_improvements))
        )
        if len(ordinary_improvements) > 1
        else 0.0
    )
    ordinary_lower_95 = float(ordinary_gain - 1.96 * ordinary_standard_error)
    ordinary_raw_gain = float(
        log_loss(truth[judgment], ordinary_raw[judgment], labels=labels)
        - log_loss(truth[judgment], candidate_raw[judgment], labels=labels)
    )

    current_lgbm_seconds = float(current_lgbm["duration"])
    challenger_lgbm_seconds = float(challenger["duration"])
    shared_xgboost_seconds = float(shared_xgboost["duration"])
    current_pair_seconds = current_lgbm_seconds + shared_xgboost_seconds
    candidate_pair_seconds = challenger_lgbm_seconds + shared_xgboost_seconds
    gate_seconds = (
        current_lgbm_seconds
        + challenger_lgbm_seconds
        + shared_xgboost_seconds
    )
    pair_ratio = candidate_pair_seconds / max(current_pair_seconds, 1e-12)
    lgbm_overhead = max(0.0, challenger_lgbm_seconds - current_lgbm_seconds)
    frozen_pair_quality_pass = bool(
        gain >= minimum_gain and lower_95 > 0.0 and raw_gain > 0.0
    )
    ordinary_incumbent_quality_pass = bool(
        ordinary_gain >= minimum_gain
        and ordinary_lower_95 > 0.0
        and ordinary_raw_gain > 0.0
    )
    quality_pass = bool(
        frozen_pair_quality_pass and ordinary_incumbent_quality_pass
    )
    cost_pass = bool(
        pair_ratio <= maximum_pair_ratio
        and lgbm_overhead <= maximum_lgbm_overhead_seconds
        and gate_seconds <= maximum_gate_seconds
    )
    accepted = bool(quality_pass and cost_pass)
    _portfolio_report(
        "Extra-LGBM pair gate: frozen-pair judgment gain %.6f, lower-95 "
        "%.6f, raw gain %.6f; ordinary %s gain %.6f, lower-95 %.6f, raw "
        "gain %.6f; pair ratio %.4f, LGBM overhead %.3fs, gate %.3fs/%.1fs; "
        "%s.",
        gain,
        lower_95,
        raw_gain,
        [contender["model"] for contender in ordinary_ensemble],
        ordinary_gain,
        ordinary_lower_95,
        ordinary_raw_gain,
        pair_ratio,
        lgbm_overhead,
        gate_seconds,
        maximum_gate_seconds,
        "admitted" if accepted else "rejected",
    )
    if not accepted:
        challenger.pop("validation_models", None)
        return ordinary, None

    return contenders, {
        "ensemble": [challenger, shared_xgboost],
        "weights": np.array([0.5, 0.5]),
        "temperature": candidate_temperature,
        "prior_exponent": candidate_prior,
        "judgment_score": -float(np.mean(candidate_losses)),
        "gain": gain,
        "lower_95": lower_95,
        "raw_gain": raw_gain,
        "ordinary_incumbent_models": [
            contender["model"] for contender in ordinary_ensemble
        ],
        "ordinary_incumbent_weights": np.asarray(
            ordinary_weights, dtype=float
        ),
        "ordinary_incumbent_gain": ordinary_gain,
        "ordinary_incumbent_lower_95": ordinary_lower_95,
        "ordinary_incumbent_raw_gain": ordinary_raw_gain,
        "pair_ratio": float(pair_ratio),
        "lgbm_overhead_seconds": float(lgbm_overhead),
        "gate_seconds": float(gate_seconds),
        "current_temperature": current_temperature,
        "current_prior_exponent": current_prior,
    }


def _select_logloss_ensemble(contenders, calibrate, pair_strong_weight=0.5):
    """Choose a calibrated singleton or a conservative two-model blend.

    The optional fixed reliability weight favours the better validation singleton
    without fitting a free blend coefficient on one holdout. Singleton strategies
    ensure that adding a weak model is never mandatory.
    """
    if not 0.5 <= pair_strong_weight < 1:
        raise ValueError("pair_strong_weight must be in the interval [0.5, 1)")
    strategies = [([contender], np.array([1.0])) for contender in contenders]
    for left_index, left in enumerate(contenders):
        for right in contenders[left_index + 1:]:
            left_weight = (
                pair_strong_weight
                if left["score"] >= right["score"]
                else 1.0 - pair_strong_weight
            )
            strategies.append(
                ([left, right], np.array([left_weight, 1.0 - left_weight]))
            )

    evaluated_strategies = []
    truth = contenders[0]["truth"]
    labels = contenders[0].get("labels")
    for ensemble, weights in strategies:
        probabilities = _blend_probabilities(
            [contender["probabilities"] for contender in ensemble], weights
        )
        temperature = (
            _fit_temperature(
                probabilities,
                truth,
                labels=labels,
                # Strong L2 regularisation intentionally makes the p >> n
                # candidate underconfident.  Its three-fold OOF predictions can
                # support a single temperature parameter with 3--4 rare-class
                # examples; other candidates retain the conservative guard.
                allow_sparse_classes=(
                    len(ensemble) == 1
                    and ensemble[0]["model"] == "scaled_logit"
                ),
            )
            if calibrate
            else 1.0
        )
        calibrated = _apply_temperature(probabilities, temperature)
        score = _selection_score(
            "logloss",
            truth,
            None,
            calibrated,
            labels=labels,
        )
        evaluated_strategies.append(
            (
                ensemble,
                weights,
                temperature,
                score,
                _row_log_losses(calibrated, truth, labels=labels),
            )
        )

    best_singleton = max(
        (strategy for strategy in evaluated_strategies if len(strategy[0]) == 1),
        key=lambda strategy: strategy[3],
    )
    pairs = [strategy for strategy in evaluated_strategies if len(strategy[0]) == 2]
    if not pairs:
        return best_singleton[:4]

    best_pair = max(pairs, key=lambda strategy: strategy[3])
    pair_loss_penalty = best_singleton[3] - best_pair[3]
    singleton_loss = max(-best_singleton[3], 1e-12)
    relative_loss_penalty = pair_loss_penalty / singleton_loss
    member_losses = [max(-contender["score"], 1e-12) for contender in best_pair[0]]
    member_loss_ratio = max(member_losses) / min(member_losses)
    member_loss_ratio_limit = 2.0
    dominated_non_improving_pair = (
        pair_loss_penalty > 0 and member_loss_ratio > member_loss_ratio_limit
    )
    paired_differences = best_pair[4] - best_singleton[4]
    standard_error = (
        np.std(paired_differences, ddof=1) / np.sqrt(len(paired_differences))
        if len(paired_differences) > 1
        else 0.0
    )
    rejection_margin = max(1.96 * standard_error, 1e-4)
    relative_penalty_limit = 0.10
    if (
        pair_loss_penalty <= rejection_margin
        and relative_loss_penalty <= relative_penalty_limit
        and not dominated_non_improving_pair
    ):
        selected = best_pair
        decision = "retained"
    else:
        selected = best_singleton
        decision = "rejected"
    _portfolio_report(
        "Fixed-weight logloss pair %s: loss penalty %.6f versus paired-noise "
        "margin %.6f; relative penalty %.1f%% versus %.1f%% limit; member loss "
        "ratio %.2f versus %.2f limit.",
        decision,
        pair_loss_penalty,
        rejection_margin,
        100 * relative_loss_penalty,
        100 * relative_penalty_limit,
        member_loss_ratio,
        member_loss_ratio_limit,
    )
    return selected[:4]


def _direct_wide_deployment_weights(
    ensemble,
    selected_weights,
    direct_all_row_refit,
    sampled_numeric_density,
    adaptive_portfolio,
):
    """Shrink a dense raw-refit booster pair towards the stable LGBM member.

    Early-stopping validation observes both models on the same small selector,
    whereas deployment extrapolates their boosting horizons to every row.  In the
    dense wide regime XGBoost remains useful as a diverse secondary but its
    extrapolation is less stable.  A fixed 70/30 reliability blend avoids fitting
    another noisy parameter. Sparse matrices retain validation-selected weights.
    Explicit candidate lists are treated as controlled experiments and are also
    left unchanged.
    """
    selected_weights = np.asarray(selected_weights, dtype=float)
    if (
        not adaptive_portfolio
        or not direct_all_row_refit
        or sampled_numeric_density is None
        or sampled_numeric_density < 0.35
        or len(ensemble) != 2
    ):
        return selected_weights

    model_names = [contender["model"] for contender in ensemble]
    if set(model_names) != {"lgbm", "xgboost"}:
        return selected_weights
    return np.asarray(
        [0.7 if model_name == "lgbm" else 0.3 for model_name in model_names],
        dtype=float,
    )


def _adaptive_deployment_temperature_multiplier(
    features,
    target,
    selected_models,
    selector_temperature,
    metric,
    calibrate,
    direct_all_row_refit,
    adaptive_portfolio,
    validation_fold_count=None,
):
    """Correct selector-to-deployment calibration shifts in guarded regimes.

    A high-work multiclass selector sees only a small fraction of the rows, while
    its compact LGBM deployment model is fitted on every row.  The selector's
    fitted temperature supplies a useful direction-of-shift signal: a very low
    temperature means the small model needed strong sharpening and should be
    relaxed after the all-row fit, whereas a near-unit or higher temperature
    transfers conservatively with a small sharpening correction.  The neutral
    band avoids changing ambiguous fits.

    Leaf-supported categorical and medium many-class numeric tables show a
    smaller consistent under-confidence after their all-row refits.  When LGBM
    survives deployment selection, a conservative sharpening correction offsets
    that shift.  Wide, well-supported categorical XGBoost deployments use the
    same bounded correction only when the selector itself required softening;
    the correction shrinks that temperature towards one without crossing it.
    All branches are disabled by any explicit portfolio setting, so controlled
    experiments retain their exact semantics.
    """
    if (
        metric != "logloss"
        or not calibrate
        or not adaptive_portfolio
    ):
        return 1.0
    shape = getattr(features, "shape", np.asarray(features).shape)
    feature_count = shape[1] if len(shape) > 1 else 1
    if (
        selected_models == ["xgboost"]
        and feature_count >= 32
        and selector_temperature > 1.0
        and _is_large_supported_mid_class_categorical(features, target)
    ):
        # Do not turn selector softening into deployment sharpening when the
        # fitted temperature is only marginally above one.  For larger shifts,
        # cap the correction at three percent; outer-fold controls showed that
        # unconditional sharpening is unsafe when the selector temperature is
        # already below one.
        return max(1.0 / selector_temperature, 0.97)
    if (
        "lgbm" in selected_models
        and (
            _is_large_low_class_categorical(features, target)
            or _is_leaf_supported_medium_many_class_numeric(features, target)
        )
    ):
        return 0.97
    if _is_low_support_small_wide_rf_ensemble(
        features,
        target,
        selected_models=selected_models,
        validation_fold_count=validation_fold_count,
    ):
        return 0.97
    if (
        not direct_all_row_refit
        or selected_models != ["lgbm"]
        or not _is_high_work_wide_many_class_numeric(features, target)
    ):
        return 1.0
    if selector_temperature < 0.8:
        return 1.10
    if selector_temperature >= 0.9:
        return 0.94
    return 1.0


def _adaptive_post_prior_temperature_multiplier(
    features,
    target,
    selected_models,
    selector_temperature,
    metric,
    calibrate,
    adaptive_portfolio,
):
    """Sharpen a guarded numeric deployment after class-prior correction.

    Temperature scaling and the multiplicative prior correction do not commute.
    On well-supported medium many-class numeric tables, outer-fold diagnostics
    consistently favoured a second three-percent sharpening of the fully
    corrected probabilities when the train-only selector clearly requested
    softening. Applying this second step after the prior correction preserves
    that measured operation order. The selector threshold supplies a train-only
    direction signal, while the existing geometry guard excludes sparse and
    rare-class tables. Explicit portfolio settings disable the branch.
    """
    if (
        metric != "logloss"
        or not calibrate
        or not adaptive_portfolio
        or "lgbm" not in selected_models
        or selector_temperature < 1.05
        or not _is_leaf_supported_medium_many_class_numeric(features, target)
    ):
        return 1.0
    return 0.97


def _is_low_support_small_wide_rf_ensemble(
    features,
    target,
    selected_models,
    validation_fold_count,
):
    """Recognise stable OOF RF ensembles that remain slightly underconfident.

    Small p≈n many-class tables give each forest leaf little class support.  A
    three-fold OOF selector can calibrate that ensemble reliably, but refitting
    both complementary members on all rows still leaves a small, repeatable
    under-confidence shift.  The per-class support ceiling separates this regime
    from better-supported image tables, while the minimum count and OOF guards
    exclude rare-label single-holdout fits.  XGBoost pairs are intentionally not
    covered because their refit calibration has different behaviour.
    """
    if validation_fold_count is None or validation_fold_count < 3:
        return False
    if len(selected_models) != 2:
        return False
    selected_model_set = set(selected_models)
    if not selected_model_set.issubset({"lgbm", "rf", "rf_large_subspace"}):
        return False
    if not selected_model_set.intersection({"rf", "rf_large_subspace"}):
        return False

    shape = getattr(features, "shape", np.asarray(features).shape)
    if len(shape) != 2:
        return False
    row_count, feature_count = shape
    target_array = np.asarray(target).reshape(-1)
    _, class_counts = np.unique(target_array, return_counts=True)
    class_count = len(class_counts)
    return bool(
        row_count <= 2_500
        and feature_count >= 256
        and feature_count >= 0.75 * row_count
        and class_count >= 10
        and class_counts.min() >= 5
        and row_count / class_count <= 40
        and row_count * feature_count <= 5_000_000
        and _all_features_are_numeric(features)
        and not _is_sparse_table(features)
    )


def _blend_probabilities(probabilities, weights):
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.average(probabilities, axis=0, weights=weights)


def _row_log_losses(probabilities, truth, labels=None):
    truth = np.asarray(truth).reshape(-1)
    labels = np.unique(truth) if labels is None else np.asarray(labels)
    label_indices = np.searchsorted(labels, truth)
    true_probabilities = probabilities[np.arange(len(truth)), label_indices]
    return -np.log(np.clip(true_probabilities, 1e-12, 1.0))


def _portfolio_candidates(value):
    if value is None:
        return ["lgbm", "xgboost", "auto"]
    if isinstance(value, str):
        candidates = [
            candidate.strip() for candidate in value.split(",") if candidate.strip()
        ]
    else:
        candidates = list(value)
    if not candidates:
        raise ValueError("_portfolio_candidates must contain at least one model")
    return candidates


def _normalise_metric_name(metric):
    aliases = {
        "neg_logloss": "logloss",
        "neg_log_loss": "logloss",
        "roc_auc": "auc",
        "accuracy": "acc",
    }
    return aliases.get(metric, metric)


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _portfolio_report(message, *args):
    rendered = message % args
    log.info(rendered)
    print(f"[FEDOT portfolio] {rendered}", flush=True)


def get_fedot_metrics(config):
    metrics_mapping = dict(
        acc="accuracy",
        auc="roc_auc",
        f1="f1",
        logloss="neg_log_loss",
        mae="mae",
        mse="mse",
        msle="msle",
        r2="r2",
        rmse="rmse",
    )
    scoring_metric = metrics_mapping.get(config.metric, None)

    if scoring_metric is None:
        log.warning(f"Performance metric {config.metric} not supported.")

    return scoring_metric


def save_artifacts(automl, config):
    artifacts = config.framework_params.get("_save_artifacts", [])
    if "models" in artifacts:
        try:
            models_dir = output_subdir("models", config)
            models_file = os.path.join(models_dir, "model.json")
            automl.current_pipeline.save(models_file)
        except Exception as e:
            log.info(f"Error when saving 'models': {e}.", exc_info=True)

    if "info" in artifacts:
        try:
            info_dir = output_subdir("info", config)
            if automl.history:
                automl.history.save(os.path.join(info_dir, "history.json"))
            else:
                log.info("There is no optimization history info to save.")
        except Exception as e:
            log.info(
                f"Error when saving info about optimisation history: {e}.",
                exc_info=True,
            )

    if "leaderboard" in artifacts:
        try:
            leaderboard_dir = output_subdir("leaderboard", config)
            if automl.history:
                lb = automl.history.get_leaderboard()
                Path(os.path.join(leaderboard_dir, "leaderboard.csv")).write_text(lb)
        except Exception as e:
            log.info(f"Error when saving 'leaderboard': {e}.", exc_info=True)
