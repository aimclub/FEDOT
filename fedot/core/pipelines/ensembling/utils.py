from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from fedot.core.composer.metrics import Accuracy, F1, Logloss, MSE, R2, RMSE, ROCAUC
from fedot.core.data.input_data.data import InputData
from fedot.core.data.input_data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter


@dataclass(frozen=True)
class ChunkedEnsembleValidationData:
    """Train/validation split and auxiliary data needed to fit chunked ensembles."""

    train_data: InputData
    validation_data: InputData
    class_representatives: Optional[dict]


def calculate_validation_metrics(
    y_true: Any,
    y_labels: Any,
    task_type: TaskTypesEnum,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true_arr = np.ravel(np.asarray(y_true))
    y_pred_arr = np.ravel(np.asarray(y_labels))
    reference = _metric_reference_data(y_true_arr, task_type)
    labels_output = _metric_output_data(reference, y_pred_arr)

    if task_type == TaskTypesEnum.classification:
        metrics: Dict[str, float] = {
            "accuracy": float(Accuracy.metric(reference, labels_output)),
            "f1": float(F1.metric(reference, labels_output)),
            "roc_auc": float("nan"),
            "roc_auc_pen": float("nan"),
            "neg_log_loss": float("nan"),
        }
        if y_proba is not None:
            prepared_proba = _prepare_proba_for_log_loss(y_proba)
            proba_output = _metric_output_data(reference, _prepare_proba_for_roc_auc(y_true_arr, prepared_proba))
            metrics["roc_auc"] = float(ROCAUC.metric(reference, proba_output))
            metrics["roc_auc_pen"] = metrics["roc_auc"]
            metrics["neg_log_loss"] = float(Logloss.metric(reference, _metric_output_data(reference, prepared_proba)))
        return metrics

    return {
        "mse": float(MSE.metric(reference, labels_output)),
        "rmse": float(RMSE.metric(reference, labels_output)),
        "r2": float(R2.metric(reference, labels_output)),
    }


def prepare_chunked_ensemble_validation(train_data: InputData, plan: Any) -> ChunkedEnsembleValidationData:
    data_producer = DataSourceSplitter(
        cv_folds=None,
        split_ratio=plan.train_split_ratio,
        shuffle=True,
        stratify=True,
        random_seed=plan.validation_split_seed,
    ).build(train_data)
    ensemble_train_data, validation_data = next(data_producer())
    class_representatives = (
        select_one_sample_per_class(ensemble_train_data)
        if plan.should_select_class_representatives
        else None
    )
    return ChunkedEnsembleValidationData(
        train_data=ensemble_train_data,
        validation_data=validation_data,
        class_representatives=class_representatives,
    )


def select_one_sample_per_class(train_data: InputData, random_state: int = 42) -> Dict[Any, Dict[str, Any]]:
    """Pick one representative per class for chunk repair when sampling drops rare classes."""
    target = np.ravel(np.asarray(train_data.target))
    rng = np.random.default_rng(random_state)
    representatives: Dict[Any, Dict[str, Any]] = {}

    for cls in np.unique(target):
        class_positions = np.where(target == cls)[0]
        picked = int(rng.choice(class_positions))
        representatives[cls] = {
            "feature": _take_single_row(train_data.features, picked),
            "target": target[picked],
            "categorical_feature": None
            if train_data.categorical_features is None
            else _take_single_row(train_data.categorical_features, picked),
        }
    return representatives


def ensure_all_classes_in_chunk(chunk_data: InputData,
                                class_representatives: Optional[Dict[Any, Dict[str, Any]]]) -> InputData:
    if not class_representatives:
        return chunk_data

    target = np.ravel(np.asarray(chunk_data.target))
    present_classes = set(np.unique(target))
    all_classes = set(class_representatives.keys())
    missing_classes = [cls for cls in all_classes if cls not in present_classes]
    if not missing_classes:
        return chunk_data

    extra_features = []
    extra_target = []
    extra_categorical = [] if chunk_data.categorical_features is not None else None

    for cls in missing_classes:
        rep = class_representatives[cls]
        extra_features.append(rep["feature"])
        extra_target.append(rep["target"])
        if extra_categorical is not None:
            extra_categorical.append(rep["categorical_feature"])

    features_extended = _append_rows(chunk_data.features, np.asarray(extra_features))
    target_extended = np.concatenate([target, np.asarray(extra_target)])
    idx_extended = _append_idx(chunk_data.idx, len(extra_target))

    categorical_features_extended = None
    if chunk_data.categorical_features is not None:
        categorical_features_extended = _append_rows(
            chunk_data.categorical_features,
            np.asarray(extra_categorical),
        )

    return InputData(
        idx=idx_extended,
        features=features_extended,
        target=target_extended,
        task=deepcopy(chunk_data.task),
        data_type=chunk_data.data_type,
        supplementary_data=chunk_data.supplementary_data,
        categorical_features=categorical_features_extended,
        categorical_idx=chunk_data.categorical_idx,
        numerical_idx=chunk_data.numerical_idx,
        encoded_idx=chunk_data.encoded_idx,
        features_names=chunk_data.features_names,
    )


def _prepare_proba_for_log_loss(y_proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(y_proba)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    elif proba.ndim == 2 and proba.shape[1] == 1:
        col = proba[:, 0]
        proba = np.column_stack([1.0 - col, col])
    return np.clip(proba, 1e-15, 1 - 1e-15)


def _prepare_proba_for_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    if y_proba.ndim == 2 and y_proba.shape[1] == 2 and len(np.unique(y_true)) == 2:
        return y_proba[:, 1]
    return y_proba


def _metric_reference_data(y_true: np.ndarray, task_type: TaskTypesEnum) -> InputData:
    return InputData(
        idx=np.arange(len(y_true)),
        features=np.empty((len(y_true), 0)),
        target=y_true,
        task=Task(task_type),
        data_type=DataTypesEnum.table,
    )


def _metric_output_data(reference: InputData, prediction: np.ndarray) -> OutputData:
    return OutputData(
        idx=reference.idx,
        features=reference.features,
        target=reference.target,
        task=reference.task,
        data_type=reference.data_type,
        predict=prediction,
    )


def _take_single_row(values: Any, position: int) -> Any:
    if values is None:
        return None
    if isinstance(values, pd.DataFrame):
        return values.iloc[position].to_numpy()
    if isinstance(values, pd.Series):
        return values.iloc[position]
    return np.asarray(values)[position]


def _append_rows(base: Any, extra_rows: np.ndarray) -> Any:
    if isinstance(base, pd.DataFrame):
        extra_df = pd.DataFrame(extra_rows, columns=base.columns)
        for column, dtype in base.dtypes.items():
            try:
                extra_df[column] = extra_df[column].astype(dtype)
            except Exception:
                continue
        return pd.concat([base, extra_df], ignore_index=True)

    base_arr = np.asarray(base)
    if base_arr.ndim == 1:
        return np.concatenate([base_arr, np.asarray(extra_rows).reshape(-1)])
    return np.vstack([base_arr, np.asarray(extra_rows)])


def _append_idx(base_idx: Any, extra_count: int) -> np.ndarray:
    idx = np.asarray(base_idx)
    if idx.size == 0:
        return np.arange(extra_count, dtype=int)
    if np.issubdtype(idx.dtype, np.number):
        start = int(np.max(idx)) + 1
        extra = np.arange(start, start + extra_count, dtype=idx.dtype)
    else:
        extra = np.asarray([f"added_{i}" for i in range(extra_count)], dtype=object)
    return np.concatenate([idx, extra])
