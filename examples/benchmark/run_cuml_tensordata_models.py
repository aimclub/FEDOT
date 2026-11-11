from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.benchmark.run_experimental_tabular_models import (  # noqa: E402
    _collect_classification_metrics,
    _collect_regression_metrics,
    _find_task,
    _parse_list,
    _to_raw_tensordata_split,
)
from examples.benchmark.run_openml_foundational import (  # noqa: E402
    DEFAULT_CLASSIFICATION_SUITE,
    DEFAULT_REGRESSION_SUITE,
    _require_openml,
    _to_xy_split,
)
from fedot.core.data.bridges.tensor_to_input import tensordata_to_input_data  # noqa: E402
from fedot.core.pipelines.node import PipelineNode  # noqa: E402
from fedot.core.pipelines.pipeline import Pipeline  # noqa: E402
from fedot.core.repository.operation_types_repository import OperationTypesRepository  # noqa: E402
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService  # noqa: E402
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum  # noqa: E402


DEFAULT_RESULT_PATH = ROOT_DIR / 'docs' / 'dev' / 'cuml_tensordata_benchmark_2026_09_08.csv'
CLASSIFICATION_OPERATIONS = {'logit', 'rf', 'svc', 'knn', 'bernb'}
REGRESSION_OPERATIONS = {'linear', 'ridge', 'lasso', 'rfr', 'knnreg'}
MODEL_PARAMETERS = {
    'logit': {'C': 1.0, 'max_iter': 10000, 'tol': 1e-7},
    'rf': {
        'n_estimators': 300,
        'max_depth': 16,
        'max_features': 'sqrt',
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'criterion': 'gini',
        'random_state': 42,
        'n_jobs': 1,
    },
    'svc': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True, 'max_iter': 100000},
    'knn': {'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
    'bernb': {'alpha': 1.0, 'binarize': 0.0},
    'linear': {},
    'ridge': {'alpha': 1.0},
    'lasso': {'alpha': 0.001, 'max_iter': 2000},
    'rfr': {
        'n_estimators': 300,
        'max_depth': 16,
        'max_features': 1.0,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': 1,
    },
    'knnreg': {'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
}


def _synchronize(engine: str):
    if engine == 'cuml':
        torch.cuda.synchronize()


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_pipeline(operation: str) -> Pipeline:
    node = PipelineNode(operation)
    node.parameters = MODEL_PARAMETERS[operation]
    return Pipeline(node)


def _run_pipeline(engine: str, operation: str, problem: str, train_data, test_data):
    repository = 'gpu_models_repository.json' if engine == 'cuml' else 'model_repository.json'
    OperationTypesRepository.assign_repo('model', repository)
    try:
        pipeline = _build_pipeline(operation)
        _synchronize(engine)
        started = time.perf_counter()
        pipeline.fit(train_data)
        if problem == 'classification':
            labels = pipeline.predict(test_data, output_mode='labels').predict
            probabilities = pipeline.predict(test_data, output_mode='full_probs').predict
            result = (_to_numpy(labels), _to_numpy(probabilities))
        else:
            prediction = pipeline.predict(test_data, output_mode='default').predict
            result = _to_numpy(prediction)
        _synchronize(engine)
        elapsed = time.perf_counter() - started
    finally:
        OperationTypesRepository.assign_repo('model', 'model_repository.json')
    return result, elapsed


def _prepare_engine_data(engine: str, train_tensor_data, test_tensor_data):
    if engine == 'cpu':
        return tensordata_to_input_data(train_tensor_data), tensordata_to_input_data(test_tensor_data), 0.0

    started = time.perf_counter()
    train_data = copy.deepcopy(train_tensor_data).to('cuda')
    test_data = copy.deepcopy(test_tensor_data).to('cuda')
    torch.cuda.synchronize()
    return train_data, test_data, time.perf_counter() - started


def _apply_shared_scaling(train_tensor_data, test_tensor_data):
    service = OptionalTabularService(use_cache=False).fit(
        train_tensor_data,
        {PreprocessingStepEnum.scaling: None},
    )
    return service.predict(train_tensor_data), service.predict(test_tensor_data)


def run_tasks(
    task_names: Sequence[str],
    operations: Sequence[str],
    engines: Sequence[str],
    repeats: int,
    classification_suite: int,
    regression_suite: int,
    result_path: Path,
):
    openml = _require_openml()
    rows = []
    for task_spec in task_names:
        problem, task_name = task_spec.split(':', 1)
        suite_id = classification_suite if problem == 'classification' else regression_suite
        task_id, resolved_name = _find_task(suite_id, task_name)
        task = openml.tasks.get_task(task_id)
        X_train, y_train, X_test, y_test = _to_xy_split(task)
        train_td, test_td, _, y_test_encoded = _to_raw_tensordata_split(
            problem, X_train, y_train, X_test, y_test
        )
        train_td, test_td = _apply_shared_scaling(train_td, test_td)

        suitable_operations = CLASSIFICATION_OPERATIONS if problem == 'classification' else REGRESSION_OPERATIONS
        for operation in operations:
            if operation not in suitable_operations:
                continue
            for engine in engines:
                train_data, test_data, transfer_seconds = _prepare_engine_data(engine, train_td, test_td)
                for repeat in range(1, repeats + 1):
                    status = 'ok'
                    error = ''
                    metrics = {}
                    try:
                        prediction, model_seconds = _run_pipeline(
                            engine, operation, problem, train_data, test_data
                        )
                        if problem == 'classification':
                            labels, probabilities = prediction
                            metrics = _collect_classification_metrics(y_test_encoded, labels, probabilities)
                        else:
                            metrics = _collect_regression_metrics(y_test_encoded, prediction)
                    except Exception as ex:
                        status = 'fail'
                        error = f'{type(ex).__name__}: {ex}'
                        model_seconds = float('nan')

                    row = {
                        'task': resolved_name,
                        'problem': problem,
                        'operation': operation,
                        'engine': engine,
                        'repeat': repeat,
                        'status': status,
                        'error': error,
                        'train_rows': len(X_train),
                        'test_rows': len(X_test),
                        'features': X_train.shape[1],
                        'transfer_seconds': transfer_seconds,
                        'model_seconds': model_seconds,
                    }
                    row.update(metrics)
                    rows.append(row)
                    print(row, flush=True)
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(result_path, index=False)
                    gc.collect()

    print(f'[saved] {result_path}')


def main():
    parser = argparse.ArgumentParser(description='Compare legacy CPU models with TensorData-native cuML models.')
    parser.add_argument(
        '--tasks',
        default='classification:bank-marketing,regression:diamonds',
        help='Comma-separated problem:OpenML-task-name specs.',
    )
    parser.add_argument(
        '--operations',
        default='logit,rf,svc,knn,bernb,linear,ridge,lasso,rfr,knnreg',
        help='Comma-separated operation names.',
    )
    parser.add_argument('--engines', default='cpu,cuml', help='Comma-separated cpu,cuml engines.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--classification-suite', type=int, default=DEFAULT_CLASSIFICATION_SUITE)
    parser.add_argument('--regression-suite', type=int, default=DEFAULT_REGRESSION_SUITE)
    parser.add_argument('--result-path', type=Path, default=DEFAULT_RESULT_PATH)
    args = parser.parse_args()

    run_tasks(
        task_names=_parse_list(args.tasks),
        operations=_parse_list(args.operations),
        engines=_parse_list(args.engines),
        repeats=args.repeats,
        classification_suite=args.classification_suite,
        regression_suite=args.regression_suite,
        result_path=args.result_path,
    )


if __name__ == '__main__':
    main()
