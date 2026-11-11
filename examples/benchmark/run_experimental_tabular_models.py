from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.benchmark.run_openml_foundational import (  # noqa: E402
    DEFAULT_CLASSIFICATION_SUITE,
    DEFAULT_REGRESSION_SUITE,
    BINARY_CSV,
    MULTICLASS_CSV,
    REGRESSION_CSV,
    _load_suite_tasks,
    _require_openml,
    _to_xy_split,
)
from fedot.api.create_data import create_data  # noqa: E402
from fedot.api.main import Fedot  # noqa: E402
from fedot.core.pipelines.node import PipelineNode  # noqa: E402
from fedot.core.pipelines.pipeline import Pipeline  # noqa: E402
from fedot.core.repository.dataset_types import DataTypesEnum  # noqa: E402
from fedot.core.repository.tasks import Task, TaskTypesEnum  # noqa: E402


DEFAULT_RESULT_PATH = ROOT_DIR / 'examples' / 'benchmark' / 'all_frameworks_res' / 'experimental_tabular_models.csv'
TENSOR_CLASSIFICATION_OPS = {'tabm', 'ft_transformer', 'tab_resnet', 'realmlp', 'torch_linear', 'torch_mlp'}
TENSOR_REGRESSION_OPS = {
    'tabmreg', 'ft_transformerreg', 'tab_resnetreg', 'realmlpreg', 'torch_linear_reg', 'torch_mlp_reg'
}
OPERATION_ALIASES = {
    'tabm_package': 'tabm',
    'tabmreg_package': 'tabmreg',
}
LEGACY_CLASSIFICATION_OPS = {'extra_trees', 'hist_gb', 'ebm'}
LEGACY_REGRESSION_OPS = {'hist_gbreg', 'ebmreg', 'mlpreg'}


def _baseline_row(problem: str, task_name: str, y_train) -> dict:
    if problem == 'regression':
        table_path = REGRESSION_CSV
        metric_name = 'rmse'
    else:
        metric_name = 'roc_auc' if pd.Series(y_train).nunique(dropna=False) <= 2 else 'log_loss'
        table_path = BINARY_CSV if metric_name == 'roc_auc' else MULTICLASS_CSV

    table = pd.read_csv(table_path)
    task_col = table['Task'].astype(str)
    matches = table[task_col == task_name]
    if matches.empty:
        matches = table[task_col.str.lower() == task_name.lower()]
    if matches.empty:
        return {'dataset_size': '', 'fedot_baseline': '', 'main_metric': metric_name}
    row = matches.iloc[0]
    return {
        'dataset_size': row.get('dataset_size', ''),
        'fedot_baseline': row.get('FEDOT', ''),
        'main_metric': metric_name,
    }


def _parse_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(',') if item.strip()]


def _find_task(suite_id: int, task_name: str):
    tasks = _load_suite_tasks(suite_id, [task_name])
    if tasks.empty:
        raise ValueError(f'Task {task_name!r} was not found in OpenML suite {suite_id}')
    row = tasks.iloc[0]
    return int(row['tid']), str(row['name'])


def _to_numeric_frame(X_train: pd.DataFrame, X_test: pd.DataFrame):
    merged = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    categorical_idx = []
    for col_idx, col in enumerate(merged.columns):
        if not pd.api.types.is_numeric_dtype(merged[col]):
            categorical_idx.append(col_idx)
            merged[col] = pd.factorize(merged[col].astype('string'), sort=True)[0]
    merged = merged.replace([np.inf, -np.inf], np.nan)
    for col in merged.columns:
        if merged[col].isna().any():
            fill_value = merged[col].median() if pd.api.types.is_numeric_dtype(merged[col]) else 0
            merged[col] = merged[col].fillna(fill_value)
    train_numeric = merged.iloc[:len(X_train)].to_numpy(dtype=np.float32)
    test_numeric = merged.iloc[len(X_train):].to_numpy(dtype=np.float32)
    return train_numeric, test_numeric, np.array(categorical_idx, dtype=int)


def _tensor_pipeline(operation: str, fast_epochs: int, device: str) -> Pipeline:
    node = PipelineNode(operation)
    if operation.startswith('torch_'):
        node.parameters = {
            'device': device,
            'epochs': fast_epochs,
            'learning_rate': 0.01 if 'linear' in operation else 0.001,
            'hidden_layer_sizes': [] if 'linear' in operation else [256, 128],
            'batch_size': 256,
            'validation_fraction': 0.1,
            'patience': max(10, fast_epochs // 5),
            'random_state': 42,
        }
    elif operation == 'tabm':
        node.parameters = {
            'device': device,
            'n_epochs': fast_epochs,
            'patience': max(20, fast_epochs // 2),
            'tabm_k': 8,
            'd_block': 256,
            'dropout': 0.0,
            'arch_type': 'tabm-mini',
            'share_training_batches': False,
            'num_emb_n_bins': 0,
        }
    elif operation == 'tabmreg':
        node.parameters = {
            'device': device,
            'n_epochs': max(fast_epochs, 300),
            'patience': max(fast_epochs, 300),
            'tabm_k': 8,
            'd_block': 256,
            'arch_type': 'tabm-mini',
            'share_training_batches': False,
            'num_emb_n_bins': 16,
        }
    elif operation.startswith('ft_transformer'):
        node.parameters = {
            'device': device,
            'n_epochs': fast_epochs,
            'patience': max(5, fast_epochs // 2),
            'd_block': 64,
            'n_blocks': 3,
            'attention_n_heads': 4,
            'attention_dropout': 0.1,
            'ffn_dropout': 0.1,
            'num_emb_n_bins': 0,
        }
    elif operation.startswith('tab_resnet'):
        node.parameters = {
            'device': device,
            'n_epochs': fast_epochs,
            'patience': max(5, fast_epochs // 2),
            'd_block': 128,
            'n_blocks': 3,
            'dropout1': 0.1,
            'dropout2': 0.1,
            'num_emb_n_bins': 0,
        }
    else:
        node.parameters = {'device': device, 'n_epochs': fast_epochs, 'n_hidden_layers': 2, 'hidden_width': 256}
    return Pipeline(node)


def _to_tensordata_split(problem: str, X_train, y_train, X_test, y_test):
    X_train_num, X_test_num, categorical_idx = _to_numeric_frame(X_train, X_test)
    if problem == 'classification':
        label_encoder = LabelEncoder()
        y_train_run = label_encoder.fit_transform(y_train)
        y_test_run = label_encoder.transform(y_test)
        task = Task(TaskTypesEnum.classification)
    else:
        y_train_run = np.asarray(y_train, dtype=np.float32)
        y_test_run = np.asarray(y_test, dtype=np.float32)
        task = Task(TaskTypesEnum.regression)

    train_td = create_data(
        X_train_num,
        backend='cpu',
        target=y_train_run,
        task=task,
        data_type=DataTypesEnum.tabular,
        categorical_idx=categorical_idx,
        use_cache=False,
    )
    test_td = create_data(
        X_test_num,
        backend='cpu',
        from_data=train_td,
        use_cache=False,
    )
    return train_td, test_td, y_train_run, y_test_run


def _to_raw_tensordata_split(problem: str, X_train, y_train, X_test, y_test):
    if problem == 'classification':
        label_encoder = LabelEncoder()
        y_train_run = label_encoder.fit_transform(y_train)
        y_test_run = label_encoder.transform(y_test)
        task = Task(TaskTypesEnum.classification)
    else:
        y_train_run = np.asarray(y_train, dtype=np.float32)
        y_test_run = np.asarray(y_test, dtype=np.float32)
        task = Task(TaskTypesEnum.regression)

    train_td = create_data(
        X_train,
        backend='cpu',
        target=y_train_run,
        task=task,
        data_type=DataTypesEnum.tabular,
        use_cache=False,
    )
    test_td = create_data(
        X_test,
        backend='cpu',
        from_data=train_td,
        use_cache=False,
    )
    return train_td, test_td, y_train_run, y_test_run


def _run_tensor_operation(operation: str, problem: str, X_train, y_train, X_test, y_test,
                          fast_epochs: int, device: str):
    if operation.startswith('torch_'):
        train_td, test_td, _, y_test_run = _to_raw_tensordata_split(
            problem, X_train, y_train, X_test, y_test
        )
    else:
        train_td, test_td, _, y_test_run = _to_tensordata_split(problem, X_train, y_train, X_test, y_test)
    automl = Fedot(problem=problem, logging_level=50, with_tuning=False, use_optional_preprocessing=False)
    automl.fit(train_td, predefined_model=_tensor_pipeline(operation, fast_epochs, device))
    if problem == 'classification':
        prediction = automl.current_pipeline.predict(test_td, output_mode='labels').predict
        proba = automl.current_pipeline.predict(test_td, output_mode='full_probs').predict
        metrics = _collect_classification_metrics(y_test_run, prediction, proba)
    else:
        prediction = automl.current_pipeline.predict(test_td).predict
        metrics = _collect_regression_metrics(y_test_run, prediction)
    return metrics


def _run_legacy_operation(operation: str, problem: str, X_train, y_train, X_test, y_test):
    train_td, test_td, _, y_test_run = _to_raw_tensordata_split(problem, X_train, y_train, X_test, y_test)
    automl = Fedot(problem=problem, logging_level=50, with_tuning=False, available_operations=[operation],
                   use_optional_preprocessing=False)
    predefined_model = operation
    if operation in {'ebm', 'ebmreg'}:
        node = PipelineNode(operation)
        node.parameters = {
            'max_rounds': 200,
            'early_stopping_rounds': 20,
            'outer_bags': 2,
            'max_bins': 256,
            'max_interaction_bins': 32,
            'interactions': '3x' if operation == 'ebm' else '5x',
            'n_jobs': 1,
        }
        predefined_model = Pipeline(node)
    elif operation == 'mlpreg':
        node = PipelineNode(operation)
        node.parameters = {
            'hidden_layer_sizes': (100,),
            'max_iter': 300,
            'early_stopping': True,
            'random_state': 42,
        }
        predefined_model = Pipeline(node)
    automl.fit(train_td, predefined_model=predefined_model)
    if problem == 'classification':
        prediction = automl.current_pipeline.predict(test_td, output_mode='labels').predict
        proba = automl.current_pipeline.predict(test_td, output_mode='full_probs').predict
        return _collect_classification_metrics(y_test_run, prediction, proba)
    prediction = automl.current_pipeline.predict(test_td).predict
    return _collect_regression_metrics(y_test_run, prediction)


def _collect_classification_metrics(y_true, prediction, proba):
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(prediction).reshape(-1)
    if np.issubdtype(y_true.dtype, np.number) and not np.issubdtype(y_pred.dtype, np.number):
        y_pred = pd.to_numeric(pd.Series(y_pred), errors='raise').to_numpy()
    proba = np.asarray(proba)
    metrics = {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'log_loss': float('nan'),
        'roc_auc': float('nan'),
    }
    try:
        metrics['log_loss'] = float(log_loss(y_true, proba))
    except ValueError:
        pass
    try:
        if len(np.unique(y_true)) <= 2:
            y_score = proba[:, 1] if proba.ndim == 2 else proba
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_score))
        else:
            metrics['roc_auc'] = float(roc_auc_score(y_true, proba, multi_class='ovr', average='macro'))
    except ValueError:
        pass
    return metrics


def _collect_regression_metrics(y_true, prediction):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(prediction).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        'rmse': rmse,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def run_tasks(task_names: Sequence[str], operations: Sequence[str], classification_suite: int,
              regression_suite: int, fast_epochs: int, result_path: Path = DEFAULT_RESULT_PATH,
              compact_result: bool = False, device: str = 'auto'):
    rows = []
    openml = _require_openml()
    for task_spec in task_names:
        problem, task_name = task_spec.split(':', 1)
        suite_id = classification_suite if problem == 'classification' else regression_suite
        task_id, resolved_name = _find_task(suite_id, task_name)
        task = openml.tasks.get_task(task_id)
        X_train, y_train, X_test, y_test = _to_xy_split(task)

        for operation in operations:
            resolved_operation = OPERATION_ALIASES.get(operation, operation)
            is_tensor = resolved_operation in TENSOR_CLASSIFICATION_OPS or resolved_operation in TENSOR_REGRESSION_OPS
            if problem == 'classification' and resolved_operation in TENSOR_REGRESSION_OPS | LEGACY_REGRESSION_OPS:
                continue
            if problem == 'regression' and resolved_operation in TENSOR_CLASSIFICATION_OPS | LEGACY_CLASSIFICATION_OPS:
                continue
            start = time.time()
            try:
                if is_tensor:
                    metrics = _run_tensor_operation(
                        resolved_operation, problem, X_train, y_train, X_test, y_test, fast_epochs, device
                    )
                else:
                    metrics = _run_legacy_operation(resolved_operation, problem, X_train, y_train, X_test, y_test)
                status = 'ok'
                error = ''
            except Exception as ex:
                metrics = {}
                status = 'fail'
                error = f'{type(ex).__name__}: {ex}'

            row = {
                'task': resolved_name,
                'problem': problem,
                'operation': operation,
                'status': status,
                'error': error,
                'train_rows': len(X_train),
                'test_rows': len(X_test),
                'features': X_train.shape[1],
                'seconds': round(time.time() - start, 3),
            }
            row.update(_baseline_row(problem, resolved_name, y_train))
            row.update(metrics)
            row['main_metric_value'] = row.get(row['main_metric'], np.nan)
            rows.append(row)
            print(row)
            gc.collect()

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if compact_result:
        compact_columns = [
            'task',
            'problem',
            'operation',
            'train_rows',
            'test_rows',
            'features',
            'seconds',
            'dataset_size',
            'fedot_baseline',
            'main_metric',
            'main_metric_value',
        ]
        result = result[compact_columns]
    result.to_csv(result_path, index=False)
    print(f'[saved] {result_path}')


def main():
    parser = argparse.ArgumentParser(description='Run quick FEDOT checks for experimental tabular models.')
    parser.add_argument(
        '--tasks',
        default='classification:Australian,classification:car,regression:boston',
        help='Comma-separated problem:task_name specs.',
    )
    parser.add_argument(
        '--operations',
        default='extra_trees,hist_gb,hist_gbreg,ebm,ebmreg,mlpreg,torch_linear,torch_mlp,'
                'torch_linear_reg,torch_mlp_reg,tabm,ft_transformer,tab_resnet,realmlp,tabmreg,'
                'ft_transformerreg,tab_resnetreg,realmlpreg',
        help='Comma-separated operation names.',
    )
    parser.add_argument('--classification-suite', type=int, default=DEFAULT_CLASSIFICATION_SUITE)
    parser.add_argument('--regression-suite', type=int, default=DEFAULT_REGRESSION_SUITE)
    parser.add_argument('--fast-epochs', type=int, default=50)
    parser.add_argument('--device', default='auto', help='Device for TensorData operations: auto, cpu or cuda.')
    parser.add_argument('--result-path', type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument('--compact-result', action='store_true')
    args = parser.parse_args()

    run_tasks(
        task_names=_parse_list(args.tasks),
        operations=_parse_list(args.operations),
        classification_suite=args.classification_suite,
        regression_suite=args.regression_suite,
        fast_epochs=args.fast_epochs,
        result_path=args.result_path,
        compact_result=args.compact_result,
        device=args.device,
    )


if __name__ == '__main__':
    main()
