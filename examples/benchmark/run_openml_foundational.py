from __future__ import annotations

import argparse
import gc
import math
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODULE_DIR = Path(__file__).resolve().parent

DEFAULT_CLASSIFICATION_SUITE = 271
DEFAULT_REGRESSION_SUITE = 269
ROW_THRESHOLD_NO_CHUNKING = 20_000
TARGET_CHUNKS = 10
CHUNK_SIZE = 20_000

RESULTS_DIR = MODULE_DIR / "all_frameworks_res"
TASK_RESULTS_DIR = MODULE_DIR / "results"
CACHE_DIR = MODULE_DIR / "cache"
BINARY_CSV = RESULTS_DIR / "binary_classification.csv"
MULTICLASS_CSV = RESULTS_DIR / "multiclass_classification.csv"
REGRESSION_CSV = RESULTS_DIR / "regression.csv"


def _require_openml() -> ModuleType:
    try:
        import openml  # type: ignore
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "openml is required for this benchmark script. "
            "Install it in FEDOT environment, e.g. `pip install openml`."
        ) from ex
    return openml


def _require_fedot():
    try:
        from fedot.api.main import Fedot
    except Exception as ex:
        raise RuntimeError(
            "FEDOT import failed. Check FEDOT environment/dependencies "
            "(for example, CUDA-enabled packages like cudf in non-GPU environment)."
        ) from ex
    return Fedot


def _load_suite_tasks(suite_id: int, task_names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    openml = _require_openml()
    suite = openml.study.get_suite(suite_id)
    task_ids = set(suite.tasks)

    tasks_df = openml.tasks.list_tasks(output_format="dataframe")
    suite_tasks_df = tasks_df[tasks_df["tid"].isin(task_ids)][["tid", "name"]].copy()

    if task_names is not None:
        requested = set(task_names)
        suite_tasks_df = suite_tasks_df[suite_tasks_df["name"].isin(requested)]

    suite_tasks_df = suite_tasks_df.sort_values("name").reset_index(drop=True)
    return suite_tasks_df


def _read_task_names(csv_path: Path) -> set[str]:
    frame = pd.read_csv(csv_path)
    return set(frame["Task"].astype(str).tolist())


def _format_metric(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000 or (abs_value > 0 and abs_value < 1e-3):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _upsert_result(csv_path: Path, task_name: str, metric_value: float) -> None:
    frame = pd.read_csv(csv_path)
    if "foundational" not in frame.columns:
        frame["foundational"] = "-"

    task_col = frame["Task"].astype(str)
    exact_matches = frame.index[task_col == task_name].tolist()
    if not exact_matches:
        lowered = task_col.str.lower()
        ci_matches = frame.index[lowered == task_name.lower()].tolist()
        exact_matches = ci_matches

    if not exact_matches:
        print(f"[skip] '{task_name}' is not present in {csv_path.name}")
        return

    frame.loc[exact_matches[0], "foundational"] = _format_metric(metric_value)
    frame.to_csv(csv_path, index=False)


def _collect_metrics(
    automl,
    problem: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_test)
    y_pred = np.asarray(prediction).reshape(-1)

    if problem == "classification":
        proba = np.asarray(automl.predict_proba(features=X_test, probs_for_all_classes=True))
        n_classes = pd.Series(y_true).nunique(dropna=False)
        metrics = {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float("nan"),
            "roc_auc": float("nan"),
        }
        try:
            metrics["log_loss"] = float(log_loss(y_true=y_true, y_pred=proba))
        except ValueError:
            pass
        try:
            if n_classes <= 2:
                y_score = proba[:, 1] if proba.ndim == 2 else proba
                metrics["roc_auc"] = float(roc_auc_score(y_true=y_true, y_score=y_score))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true=y_true, y_score=proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass
        return metrics

    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _save_task_metrics(task_name: str,
                       problem: str,
                       metrics: dict[str,
                                     float],
                       main_metric_name: str,
                       main_metric_value: float) -> None:
    TASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_path = TASK_RESULTS_DIR / f"{task_name}.csv"
    payload = {
        "task": task_name,
        "problem": problem,
        "main_metric_name": main_metric_name,
        "main_metric_value": main_metric_value}
    payload.update(metrics)
    frame = pd.DataFrame([payload])
    frame.to_csv(file_path, index=False)


def get_split_indices_cached(task, repeat: int = 0, fold: int = 0, sample: int = 0):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    task_id = getattr(task, "task_id", None)
    if task_id is None:
        task_id = getattr(task, "id", None)
    if task_id is None:
        task_id = str(getattr(task, "name", "unknown_task")).replace("/", "_")

    cache_file = CACHE_DIR / f"task_{task_id}_r{repeat}_f{fold}_s{sample}.pkl"
    if cache_file.exists():
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
        return np.asarray(payload["train_idx"]), np.asarray(payload["test_idx"])

    train_idx, test_idx = task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=sample)
    with cache_file.open("wb") as f:
        pickle.dump(
            {"train_idx": np.asarray(train_idx), "test_idx": np.asarray(test_idx)},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return np.asarray(train_idx), np.asarray(test_idx)


def _to_xy_split(task) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    X, y = task.get_X_and_y(dataset_format="dataframe")
    train_idx, test_idx = get_split_indices_cached(task, repeat=0, fold=0, sample=0)

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
    else:
        X_np = np.asarray(X)
        X_train = pd.DataFrame(X_np[train_idx])
        X_test = pd.DataFrame(X_np[test_idx])

    y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
    y_train = y_series.iloc[train_idx].to_numpy()
    y_test = y_series.iloc[test_idx].to_numpy()
    return X_train, y_train, X_test, y_test


def _make_sampling_config(train_rows: int, seed: int) -> dict:
    n_partitions = max(1, int(math.ceil(train_rows / CHUNK_SIZE)))
    chunks_percent = min(100.0, (TARGET_CHUNKS / n_partitions) * 100.0)
    return {
        "strategy_kind": "chunking",
        "provider": "sampling_zoo",
        "strategy": "random",
        "strategy_params": {
            "n_partitions": n_partitions,
            "chunks_percent": chunks_percent,
        },
        "random_state": int(seed),
    }


def _run_single_task(
    task_id: int,
    task_name: str,
    problem: str,
    metric_name: str,
    operation_name: str,
    timeout: float,
    seed: int,
    n_jobs: int,
    logging_level: int,
    cv_folds: int,
) -> tuple[float, dict[str, float]]:
    openml = _require_openml()
    Fedot = _require_fedot()
    task = openml.tasks.get_task(task_id)
    X_train, y_train, X_test, y_test = _to_xy_split(task)

    use_sampling = len(X_train) > ROW_THRESHOLD_NO_CHUNKING
    sampling_config = _make_sampling_config(len(X_train), seed) if use_sampling else None

    print(
        f"[run] {task_name} | rows_train={len(X_train)} | "
        f"sampling={'on' if use_sampling else 'off'} | model={operation_name}"
    )

    automl = Fedot(
        problem=problem,
        timeout=timeout,
        seed=seed,
        n_jobs=n_jobs,
        logging_level=logging_level,
        with_tuning=False,
        cv_folds=cv_folds,
        available_operations=[operation_name],
        sampling_config=sampling_config,
    )

    predefined_model = operation_name
    automl.fit(features=X_train, target=y_train, predefined_model=predefined_model)
    prediction = automl.predict(features=X_test)
    metrics = _collect_metrics(
        automl=automl,
        problem=problem,
        X_test=X_test,
        y_test=y_test,
        prediction=prediction,
    )
    score = float(metrics[metric_name])

    del automl, X_train, X_test, y_train, y_test, task
    gc.collect()
    return score, metrics


def run_openml_foundational(
    classification_suite: int = DEFAULT_CLASSIFICATION_SUITE,
    regression_suite: int = DEFAULT_REGRESSION_SUITE,
    classification_tasks: Optional[Sequence[str]] = None,
    regression_tasks: Optional[Sequence[str]] = None,
    timeout: float = 30.0,
    seed: int = 42,
    n_jobs: int = -1,
    logging_level: int = 30,
    cv_folds: int = 3,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    binary_tasks_from_table = _read_task_names(BINARY_CSV)
    multiclass_tasks_from_table = _read_task_names(MULTICLASS_CSV)
    regression_tasks_from_table = _read_task_names(REGRESSION_CSV)

    if classification_tasks is None:
        classification_filter = binary_tasks_from_table | multiclass_tasks_from_table
    else:
        classification_filter = set(classification_tasks)

    if regression_tasks is None:
        regression_filter = regression_tasks_from_table
    else:
        regression_filter = set(regression_tasks)

    cls_tasks_df = _load_suite_tasks(classification_suite, classification_filter)
    reg_tasks_df = _load_suite_tasks(regression_suite, regression_filter)

    print(
        f"[info] classification tasks to run: {len(cls_tasks_df)} | "
        f"regression tasks to run: {len(reg_tasks_df)}"
    )

    for row in cls_tasks_df.itertuples(index=False):
        task_name = str(row.name)
        # try:
        openml = _require_openml()
        task = openml.tasks.get_task(int(row.tid))
        _, y = task.get_X_and_y(dataset_format="dataframe")
        n_classes = int(pd.Series(y).nunique(dropna=False))
        del task, y
        gc.collect()

        if n_classes <= 2:
            metric_name = "roc_auc"
            target_csv = BINARY_CSV
        else:
            metric_name = "log_loss"
            target_csv = MULTICLASS_CSV

        score, metrics = _run_single_task(
            task_id=int(row.tid),
            task_name=task_name,
            problem="classification",
            metric_name=metric_name,
            operation_name="tabicl",
            timeout=timeout,
            seed=seed,
            n_jobs=n_jobs,
            logging_level=logging_level,
            cv_folds=cv_folds,
        )
        _upsert_result(target_csv, task_name, score)
        _save_task_metrics(
            task_name=task_name,
            problem="classification",
            metrics=metrics,
            main_metric_name=metric_name,
            main_metric_value=score,
        )
        print(f"[ok] {task_name}: {metric_name}={score}")
        # except Exception as ex:
        #     print(f"[fail] {task_name}: {ex}")

    for row in reg_tasks_df.itertuples(index=False):
        task_name = str(row.name)
        try:
            score, metrics = _run_single_task(
                task_id=int(row.tid),
                task_name=task_name,
                problem="regression",
                metric_name="rmse",
                operation_name="tabiclreg",
                timeout=timeout,
                seed=seed,
                n_jobs=n_jobs,
                logging_level=logging_level,
                cv_folds=cv_folds,
            )
            _upsert_result(REGRESSION_CSV, task_name, score)
            _save_task_metrics(
                task_name=task_name,
                problem="regression",
                metrics=metrics,
                main_metric_name="rmse",
                main_metric_value=score,
            )
            print(f"[ok] {task_name}: rmse={score}")
        except Exception as ex:
            print(f"[fail] {task_name}: {ex}")


def _parse_tasks(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenML benchmark with FEDOT TabICL and update foundational column.")
    parser.add_argument("--classification-suite", type=int, default=DEFAULT_CLASSIFICATION_SUITE)
    parser.add_argument("--regression-suite", type=int, default=DEFAULT_REGRESSION_SUITE)
    parser.add_argument("--classification-tasks", type=str, default=None, help="Comma-separated task names.")
    parser.add_argument("--regression-tasks", type=str, default=None, help="Comma-separated task names.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--logging-level", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()

    run_openml_foundational(
        classification_suite=args.classification_suite,
        regression_suite=args.regression_suite,
        classification_tasks=["ada"],
        regression_tasks=[],
        timeout=args.timeout,
        seed=args.seed,
        n_jobs=args.n_jobs,
        logging_level=args.logging_level,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
