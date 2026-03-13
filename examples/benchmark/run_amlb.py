from __future__ import annotations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

import argparse
import inspect
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_TIMEOUT_MINUTES_PER_DATASET = 10.0
DEFAULT_RESULTS_ROOT = Path('examples/benchmark/results')
DEFAULT_CATEGORY_PROFILE = ('amlb_top20_mix',)


@dataclass(frozen=True)
class AMLBDatasetSpec:
    name: str
    openml_name: Optional[str] = None
    openml_id: Optional[int] = None
    task_type: str = 'classification'


@dataclass(frozen=True)
class LoadedDataset:
    spec: AMLBDatasetSpec
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class BenchmarkRunConfig:
    timeout_minutes_per_dataset: float = DEFAULT_TIMEOUT_MINUTES_PER_DATASET
    output_root: Path = DEFAULT_RESULTS_ROOT
    seed: int = 42
    n_jobs: int = -1
    preset: str = 'best_quality'
    with_tuning: bool = True
    dataset_names: Tuple[str, ...] = ()
    amlb_categories: Tuple[str, ...] = DEFAULT_CATEGORY_PROFILE
    test_size: float = 0.2
    max_rows_per_dataset: int = 25000
    include_baseline: bool = True
    include_sampling: bool = True
    sampling_config: Dict[str, Any] = field(default_factory=dict)


def _default_sampling_config(seed: int) -> Dict[str, Any]:
    return {
        'strategy_kind': 'subset',
        'provider': 'sampling_zoo',
        'strategy': 'random',
        'strategy_params': {},
        'candidate_ratios': [0.15, 0.2, 0.3, 0.5],
        'delta_metric_threshold': 0.03,
        'delta_type': 'relative',
        'validation_size': 0.2,
        'budget_policy': 'dynamic_cap',
        'cap_max_timeout_share': 0.35,
        'min_automl_time_minutes': 0.1,
        'infinite_timeout_cap_minutes': 5.0,
        'error_policy': 'fail_fast',
        'artifact_mode': 'minimal',
        'random_state': seed,
    }


AMLB_DATASETS: Dict[str, AMLBDatasetSpec] = {
    'amlb_adult': AMLBDatasetSpec(name='amlb_adult', openml_name='adult'),
    'amlb_covertype': AMLBDatasetSpec(name='amlb_covertype', openml_name='covertype'),
    'amlb_optdigits': AMLBDatasetSpec(name='amlb_optdigits', openml_name='optdigits'),
    'amlb_vehicle': AMLBDatasetSpec(name='amlb_vehicle', openml_name='vehicle'),
    'amlb_mfeat_factors': AMLBDatasetSpec(name='amlb_mfeat_factors', openml_name='mfeat-factors'),
    'amlb_segment': AMLBDatasetSpec(name='amlb_segment', openml_name='segment'),
    'amlb_credit_g': AMLBDatasetSpec(name='amlb_credit_g', openml_name='credit-g'),
    'amlb_kr_vs_kp': AMLBDatasetSpec(name='amlb_kr_vs_kp', openml_name='kr-vs-kp'),
    'amlb_sick': AMLBDatasetSpec(name='amlb_sick', openml_name='sick'),
    'amlb_spambase': AMLBDatasetSpec(name='amlb_spambase', openml_name='spambase'),
    'amlb_letter': AMLBDatasetSpec(name='amlb_letter', openml_name='letter'),
    'amlb_satimage': AMLBDatasetSpec(name='amlb_satimage', openml_name='satimage'),
    'amlb_waveform': AMLBDatasetSpec(name='amlb_waveform', openml_name='waveform-5000'),
    'amlb_phoneme': AMLBDatasetSpec(name='amlb_phoneme', openml_name='phoneme'),
    'amlb_page_blocks': AMLBDatasetSpec(name='amlb_page_blocks', openml_name='page-blocks'),
    'amlb_ionosphere': AMLBDatasetSpec(name='amlb_ionosphere', openml_name='ionosphere'),
    'amlb_banknote_authentication': AMLBDatasetSpec(name='amlb_banknote_authentication',
                                                    openml_name='banknote-authentication'),
    'amlb_wine_quality_red': AMLBDatasetSpec(name='amlb_wine_quality_red', openml_name='wine-quality-red'),
    'amlb_wine_quality_white': AMLBDatasetSpec(name='amlb_wine_quality_white', openml_name='wine-quality-white'),
    'amlb_magic_telescope': AMLBDatasetSpec(name='amlb_magic_telescope', openml_name='magic-telescope'),
}

AMLB_CATEGORY_PROFILES: Dict[str, Tuple[str, ...]] = {
    'small_samples_many_classes': (
        'amlb_optdigits',
        'amlb_vehicle',
        'amlb_mfeat_factors',
        'amlb_segment',
        'amlb_satimage',
        'amlb_letter',
    ),
    'large_samples_binary': (
        'amlb_adult',
        'amlb_covertype',
        'amlb_magic_telescope',
        'amlb_spambase',
        'amlb_banknote_authentication',
        'amlb_ionosphere',
    ),
    'tabular_mixed_classification': (
        'amlb_credit_g',
        'amlb_kr_vs_kp',
        'amlb_sick',
        'amlb_waveform',
        'amlb_phoneme',
        'amlb_page_blocks',
        'amlb_wine_quality_red',
        'amlb_wine_quality_white',
    ),
    'amlb_top20_mix': (
        'amlb_adult',
        'amlb_covertype',
        'amlb_optdigits',
        'amlb_vehicle',
        'amlb_mfeat_factors',
        'amlb_segment',
        'amlb_credit_g',
        'amlb_kr_vs_kp',
        'amlb_sick',
        'amlb_spambase',
        'amlb_letter',
        'amlb_satimage',
        'amlb_waveform',
        'amlb_phoneme',
        'amlb_page_blocks',
        'amlb_ionosphere',
        'amlb_banknote_authentication',
        'amlb_wine_quality_red',
        'amlb_wine_quality_white',
        'amlb_magic_telescope',
    ),
}


def parse_ratio_list(ratios: str) -> Tuple[float, ...]:
    parsed = []
    for raw in ratios.split(','):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if value <= 0 or value > 1:
            raise ValueError(f'Candidate ratio must be in (0, 1], got {value}.')
        parsed.append(value)

    if not parsed:
        raise ValueError('At least one candidate ratio must be provided.')

    unique_sorted = tuple(sorted(set(parsed)))
    return unique_sorted


def _safe_name(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(payload)), ensure_ascii=False, indent=2), encoding='utf-8')


def _resolve_dataset_specs(dataset_names: Sequence[str], amlb_categories: Sequence[str]) -> List[AMLBDatasetSpec]:
    requested_names: List[str] = []

    if dataset_names:
        requested_names.extend(dataset_names)
    else:
        categories = amlb_categories or DEFAULT_CATEGORY_PROFILE
        for category in categories:
            if category not in AMLB_CATEGORY_PROFILES:
                raise ValueError(
                    f'Unknown AMLB category: {category}. Available: {sorted(AMLB_CATEGORY_PROFILES.keys())}'
                )
            requested_names.extend(AMLB_CATEGORY_PROFILES[category])

    unique_names = []
    seen = set()
    for name in requested_names:
        normalized = name.strip().lower()
        if not normalized:
            continue
        if normalized not in AMLB_DATASETS:
            raise ValueError(f'Unknown AMLB dataset profile: {name}.')
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_names.append(normalized)

    return [AMLB_DATASETS[name] for name in unique_names]


def _extract_target_name(dataset: Any, frame: pd.DataFrame) -> str:
    if getattr(dataset, 'target', None) is not None and getattr(dataset.target, 'name', None):
        return str(dataset.target.name)

    target_names = list(getattr(dataset, 'target_names', []) or [])
    if target_names:
        return str(target_names[0])

    fallback = frame.columns[-1]
    return str(fallback)


def _sanitize_features_for_fedot(features: pd.DataFrame,
                                 numeric_columns: Sequence[str],
                                 categorical_columns: Sequence[str]) -> pd.DataFrame:
    sanitized = features.copy()

    for column in numeric_columns:
        numeric_column = pd.to_numeric(sanitized[column], errors='coerce')
        fill_value = numeric_column.median(skipna=True)
        if pd.isna(fill_value):
            fill_value = 0.0
        sanitized[column] = numeric_column.fillna(fill_value)

    for column in categorical_columns:
        category_column = sanitized[column].astype('object')
        category_column = category_column.where(~pd.isna(category_column), '__missing__')
        # Keep AMLB categories consistent and numeric for FEDOT assumptions stage.
        category_column = category_column.astype(str)
        codes, _ = pd.factorize(category_column, sort=True)
        sanitized[column] = codes.astype(np.int64)

    extra_columns = [column for column in sanitized.columns if
                     column not in set(numeric_columns) | set(categorical_columns)]
    for column in extra_columns:
        fallback_column = sanitized[column].astype('object')
        fallback_column = fallback_column.where(~pd.isna(fallback_column), '__missing__')
        fallback_column = fallback_column.astype(str)
        codes, _ = pd.factorize(fallback_column, sort=True)
        sanitized[column] = codes.astype(np.int64)

    return sanitized


def _load_amlb_dataset(spec: AMLBDatasetSpec,
                       seed: int,
                       max_rows: int,
                       test_size: float) -> LoadedDataset:
    if spec.openml_id is not None:
        dataset = fetch_openml(data_id=spec.openml_id, as_frame=True, parser='auto')
    elif spec.openml_name is not None:
        dataset = fetch_openml(name=spec.openml_name, as_frame=True, parser='auto')
    else:
        raise ValueError(f'Dataset spec {spec.name} has neither openml_name nor openml_id.')

    frame = dataset.frame.copy()
    target_name = _extract_target_name(dataset, frame)

    y_raw = frame[target_name]
    x = frame.drop(columns=[target_name])

    valid = y_raw.notna()
    x = x.loc[valid].reset_index(drop=True)
    y_raw = y_raw.loc[valid].reset_index(drop=True)

    if spec.task_type == 'classification':
        y = pd.Series(y_raw.astype('category').cat.codes, name='target')
        valid_classes = y >= 0
        x = x.loc[valid_classes].reset_index(drop=True)
        y = y.loc[valid_classes].reset_index(drop=True)
    elif spec.task_type == 'regression':
        y = pd.to_numeric(y_raw, errors='coerce')
        valid_numeric = y.notna()
        x = x.loc[valid_numeric].reset_index(drop=True)
        y = y.loc[valid_numeric].reset_index(drop=True)
        y = pd.Series(y, name='target')
    else:
        raise ValueError(f'Unsupported task type: {spec.task_type}')

    if max_rows > 0 and len(x) > max_rows:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(np.arange(len(x)), size=max_rows, replace=False))
        x = x.iloc[selected].reset_index(drop=True)
        y = y.iloc[selected].reset_index(drop=True)

    numeric_columns = x.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]
    x = _sanitize_features_for_fedot(
        features=x,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    stratify = y if spec.task_type == 'classification' and y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    metadata = {
        'n_rows': int(len(x)),
        'n_features': int(x.shape[1]),
        'n_train': int(len(x_train)),
        'n_test': int(len(x_test)),
        'n_numeric_features': int(len(numeric_columns)),
        'n_categorical_features': int(len(categorical_columns)),
        'task_type': spec.task_type,
        'target_name': target_name,
    }

    y_train_frame = pd.DataFrame({'target': np.asarray(y_train).reshape(-1)})
    y_test_frame = pd.DataFrame({'target': np.asarray(y_test).reshape(-1)})

    return LoadedDataset(
        spec=spec,
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train_frame.reset_index(drop=True),
        y_test=y_test_frame.reset_index(drop=True),
        metadata=metadata,
    )


def _evaluate_metrics(task_type: str,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray]) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if task_type == 'classification':
        metrics: Dict[str, Any] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        }

        if y_proba is not None:
            try:
                classes_count = len(np.unique(y_true))
                proba = np.asarray(y_proba)
                if classes_count <= 2:
                    if proba.ndim == 2:
                        positive_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                    else:
                        positive_proba = proba.reshape(-1)
                    metrics['roc_auc'] = float(roc_auc_score(y_true, positive_proba))
                else:
                    if proba.ndim == 1:
                        raise ValueError('Multiclass ROC-AUC requires 2D probability array.')
                    metrics['roc_auc_ovr_macro'] = float(
                        roc_auc_score(y_true, proba, average='macro', multi_class='ovr')
                    )
            except Exception as ex:
                metrics['roc_auc_error'] = str(ex)

        return metrics

    if task_type == 'regression':
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            'r2': float(r2_score(y_true, y_pred)),
            'rmse': rmse,
            'mae': float(mean_absolute_error(y_true, y_pred)),
        }

    raise ValueError(f'Unsupported task type: {task_type}')


def _invoke_with_supported_kwargs(obj: Any,
                                  method_name: str,
                                  kwargs: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    method = getattr(obj, method_name, None)
    if method is None:
        return False, 'method_not_available'

    signature = inspect.signature(method)
    call_kwargs = {}
    for key, value in kwargs.items():
        if key in signature.parameters:
            call_kwargs[key] = str(value) if isinstance(value, Path) else value

    try:
        method(**call_kwargs)
        return True, None
    except Exception as ex:
        return False, str(ex)


def _save_history_visualizations(history: Any, output_dir: Path) -> Dict[str, Any]:
    from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer

    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = PipelineHistoryVisualizer(history)

    tasks = [
        ('fitness_line', {'save_path': output_dir / 'fitness_line.png'}),
        ('fitness_line_interactive', {'save_path': output_dir / 'fitness_line_interactive.html'}),
        ('fitness_box', {'save_path': output_dir / 'fitness_box.png', 'best_fraction': 1.0}),
        ('operations_kde', {'save_path': output_dir / 'operations_kde.png'}),
        ('operations_animated_bar', {'save_path': output_dir / 'operations_animated_bar.gif', 'show_fitness': True}),
        ('diversity_population', {'save_path': output_dir / 'diversity_population.gif', 'fps': 1}),
    ]

    report: Dict[str, Any] = {}
    for method_name, method_kwargs in tasks:
        ok, error = _invoke_with_supported_kwargs(visualizer, method_name, method_kwargs)
        method_report = {'status': 'saved' if ok else 'skipped', 'error': error}
        if ok and 'save_path' in method_kwargs:
            method_report['artifact'] = str(method_kwargs['save_path'])
        report[method_name] = method_report

    return report


def _save_pipeline_visualization(model: 'Fedot', output_path: Path) -> Dict[str, Any]:
    try:
        model.current_pipeline.show(save_path=output_path)
        return {'status': 'saved', 'artifact': str(output_path)}
    except Exception as ex:
        return {'status': 'skipped', 'error': str(ex)}


def _save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _run_fedot_mode(dataset: LoadedDataset,
                    config: BenchmarkRunConfig,
                    mode_name: str,
                    sampling_config: Optional[Mapping[str, Any]],
                    mode_dir: Path) -> Dict[str, Any]:
    from fedot import Fedot

    mode_dir.mkdir(parents=True, exist_ok=True)
    history_dir = mode_dir / 'composer_history'
    history_dir.mkdir(parents=True, exist_ok=True)

    fedot_params: Dict[str, Any] = {
        'problem': dataset.spec.task_type,
        'timeout': config.timeout_minutes_per_dataset,
        'seed': config.seed,
        'n_jobs': 1,  # config.n_jobs
        'preset': config.preset,
        'use_input_preprocessing': False,
        'logging_level': 10,
        'with_tuning': config.with_tuning,
        'history_dir': str(history_dir),
        'keep_history': True,
        'sampling_config': dict(sampling_config) if sampling_config is not None else None,
    }

    run_started = perf_counter()
    model = Fedot(**fedot_params)

    fit_started = perf_counter()
    model.fit(features=dataset.x_train, target=dataset.y_train)
    fit_seconds = perf_counter() - fit_started

    predict_started = perf_counter()
    raw_prediction = model.predict(features=dataset.x_test)
    predict_seconds = perf_counter() - predict_started

    prediction = np.asarray(raw_prediction).reshape(-1)

    probabilities: Optional[np.ndarray] = None
    if dataset.spec.task_type == 'classification':
        try:
            raw_probabilities = model.predict_proba(features=dataset.x_test, probs_for_all_classes=True)
            probabilities = np.asarray(raw_probabilities)
        except Exception:
            probabilities = None

    metrics = _evaluate_metrics(
        task_type=dataset.spec.task_type,
        y_true=np.asarray(dataset.y_test),
        y_pred=prediction,
        y_proba=probabilities,
    )

    predictions_frame = pd.DataFrame({
        'y_true': np.asarray(dataset.y_test).reshape(-1),
        'y_pred': prediction,
    })
    _save_dataframe(mode_dir / 'predictions.csv', predictions_frame)

    if probabilities is not None:
        np.save(mode_dir / 'prediction_proba.npy', probabilities)

    artifacts: Dict[str, Any] = {}

    if model.history is not None:
        history_path = mode_dir / 'opt_history.json'
        model.history.save(str(history_path))
        artifacts['opt_history'] = str(history_path)

        try:
            leaderboard = model.history.get_leaderboard()
            if isinstance(leaderboard, pd.DataFrame):
                _save_dataframe(mode_dir / 'history_leaderboard.csv', leaderboard)
                artifacts['history_leaderboard'] = str(mode_dir / 'history_leaderboard.csv')
        except Exception:
            pass

        artifacts['history_visualizations'] = _save_history_visualizations(
            history=model.history,
            output_dir=mode_dir / 'history_visualizations',
        )

    artifacts['pipeline_visualization'] = _save_pipeline_visualization(
        model=model,
        output_path=mode_dir / 'pipeline.png',
    )

    try:
        pipeline_save_dir = mode_dir / 'pipeline_saved'
        pipeline_save_dir.mkdir(parents=True, exist_ok=True)
        model.current_pipeline.save(path=str(pipeline_save_dir), create_subdir=False, is_datetime_in_path=False)
        artifacts['pipeline_serialized'] = str(pipeline_save_dir)
    except Exception as ex:
        artifacts['pipeline_serialized'] = {'status': 'skipped', 'error': str(ex)}

    try:
        report = model.return_report()
        report.to_csv(mode_dir / 'fedot_time_report.csv')
        artifacts['fedot_time_report'] = str(mode_dir / 'fedot_time_report.csv')
    except Exception as ex:
        artifacts['fedot_time_report'] = {'status': 'skipped', 'error': str(ex)}

    sampling_metadata = model.sampling_stage_metadata
    if sampling_metadata is not None:
        _save_json(mode_dir / 'sampling_stage_metadata.json', sampling_metadata)
        artifacts['sampling_stage_metadata'] = str(mode_dir / 'sampling_stage_metadata.json')

    total_seconds = perf_counter() - run_started

    result = {
        'dataset': dataset.spec.name,
        'mode': mode_name,
        'task_type': dataset.spec.task_type,
        'status': 'success',
        'metrics': metrics,
        'timings_seconds': {
            'fit': float(fit_seconds),
            'predict': float(predict_seconds),
            'total': float(total_seconds),
        },
        'sampling_enabled': sampling_config is not None,
        'sampling_strategy': sampling_config.get('strategy') if sampling_config else None,
        'rows_train': int(len(dataset.y_train)),
        'rows_test': int(len(dataset.y_test)),
        'artifacts': artifacts,
    }

    _save_json(mode_dir / 'run_result.json', result)
    return result


def _build_markdown_report(records: Sequence[Mapping[str, Any]], report_path: Path) -> None:
    lines = [
        '# FEDOT AMLB Sampling Benchmark Report',
        '',
        '| Dataset | Mode | Status | Main metrics | Fit (s) | Total (s) |',
        '|---|---|---|---|---:|---:|',
    ]

    for record in records:
        metrics = record.get('metrics', {}) or {}
        if 'f1_macro' in metrics:
            main_metrics = f"f1_macro={metrics.get('f1_macro', float('nan')):.4f}"
        elif 'rmse' in metrics:
            main_metrics = f"rmse={metrics.get('rmse', float('nan')):.4f}"
        else:
            main_metrics = '-'

        timings = record.get('timings_seconds', {}) or {}
        lines.append(
            f"| {record.get('dataset', '-')} | {record.get('mode', '-')} | {record.get('status', '-')} | "
            f"{main_metrics} | {float(timings.get('fit', float('nan'))):.3f} | "
            f"{float(timings.get('total', float('nan'))):.3f} |"
        )

    report_path.write_text('\n'.join(lines), encoding='utf-8')


def run_benchmark(config: BenchmarkRunConfig) -> Dict[str, Any]:
    if not config.include_baseline and not config.include_sampling:
        raise ValueError('At least one mode must be enabled: baseline or sampling.')

    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_dir = config.output_root / f'run_amlb_fedot_sampling_{run_id}'
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_specs = _resolve_dataset_specs(config.dataset_names, config.amlb_categories)

    run_meta = {
        'run_id': run_id,
        'started_utc': datetime.utcnow().isoformat(timespec='seconds'),
        'timeout_minutes_per_dataset': config.timeout_minutes_per_dataset,
        'dataset_count': len(dataset_specs),
        'datasets': [spec.name for spec in dataset_specs],
        'config': {
            **asdict(config),
            'output_root': str(config.output_root),
        },
    }
    _save_json(run_dir / 'run_meta.json', run_meta)

    records: List[Dict[str, Any]] = []

    for dataset_spec in dataset_specs:
        print(f'\\n=== Dataset: {dataset_spec.name} ===')
        dataset_dir = run_dir / _safe_name(dataset_spec.name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            loaded = _load_amlb_dataset(
                spec=dataset_spec,
                seed=config.seed,
                max_rows=config.max_rows_per_dataset,
                test_size=config.test_size,
            )
            _save_json(dataset_dir / 'dataset_metadata.json', loaded.metadata)
        except Exception as ex:
            error_record = {
                'dataset': dataset_spec.name,
                'mode': 'dataset_loading',
                'task_type': dataset_spec.task_type,
                'status': 'failed',
                'error': str(ex),
                'traceback': traceback.format_exc(),
            }
            _save_json(dataset_dir / 'dataset_loading_error.json', error_record)
            records.append(error_record)
            print(f'Failed to load dataset {dataset_spec.name}: {ex}')
            continue

        mode_specs: List[Tuple[str, Optional[Mapping[str, Any]]]] = []
        if config.include_baseline:
            mode_specs.append(('fedot_full_dataset', None))
        if config.include_sampling:
            mode_specs.append(('fedot_sampling_stage', dict(config.sampling_config)))

        for mode_name, sampling_config in mode_specs:
            print(f'  -> Mode: {mode_name}')
            mode_dir = dataset_dir / _safe_name(mode_name)

            try:
                mode_result = _run_fedot_mode(
                    dataset=loaded,
                    config=config,
                    mode_name=mode_name,
                    sampling_config=sampling_config,
                    mode_dir=mode_dir,
                )
                records.append(mode_result)
                print(f"     success: fit={mode_result['timings_seconds']['fit']:.2f}s")
            except Exception as ex:
                failed = {
                    'dataset': dataset_spec.name,
                    'mode': mode_name,
                    'task_type': dataset_spec.task_type,
                    'status': 'failed',
                    'sampling_enabled': sampling_config is not None,
                    'error': str(ex),
                    'traceback': traceback.format_exc(),
                }
                _save_json(mode_dir / 'run_result_error.json', failed)
                records.append(failed)
                print(f'     failed: {ex}')

    summary_frame = pd.DataFrame(records)
    summary_csv = run_dir / 'benchmark_runs.csv'
    summary_json = run_dir / 'benchmark_runs.json'
    summary_frame.to_csv(summary_csv, index=False)
    summary_json.write_text(summary_frame.to_json(orient='records', force_ascii=False, indent=2), encoding='utf-8')

    _build_markdown_report(records, run_dir / 'report.md')

    finished = {
        'run_id': run_id,
        'run_dir': str(run_dir),
        'records_count': len(records),
        'successful_runs': int(sum(1 for record in records if record.get('status') == 'success')),
        'failed_runs': int(sum(1 for record in records if record.get('status') != 'success')),
    }
    _save_json(run_dir / 'run_summary.json', finished)

    print('\\n=== Benchmark complete ===')
    print(f"Results dir: {run_dir}")
    print(f"Successful runs: {finished['successful_runs']}")
    print(f"Failed runs: {finished['failed_runs']}")

    return {
        'summary': finished,
        'records': records,
        'run_dir': run_dir,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run FEDOT AMLB benchmark with sampling stage and save optimization artifacts.'
    )
    parser.add_argument('--datasets', nargs='*', default=[],
                        help='Explicit AMLB dataset profile names. If omitted, categories are used.')
    parser.add_argument('--amlb-categories', nargs='*', default=list(DEFAULT_CATEGORY_PROFILE),
                        help='AMLB category profiles. Used only when --datasets is empty.')
    parser.add_argument('--timeout-minutes', type=float, default=DEFAULT_TIMEOUT_MINUTES_PER_DATASET,
                        help='Time budget per dataset in minutes. Default: 15.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs for FEDOT.')
    parser.add_argument('--preset', type=str, default='best_quality', help='FEDOT preset.')
    parser.add_argument('--disable-tuning', action='store_true', help='Disable post-composition tuning inside FEDOT.')
    parser.add_argument('--max-rows', type=int, default=25000,
                        help='Maximum rows per dataset to keep benchmark stable in runtime.')
    parser.add_argument('--output-root', type=str, default=str(DEFAULT_RESULTS_ROOT),
                        help='Root directory for benchmark artifacts.')

    parser.add_argument('--disable-baseline', action='store_true',
                        help='Disable baseline mode (full dataset, no sampling).')
    parser.add_argument('--disable-sampling', action='store_true',
                        help='Disable sampling mode.')

    parser.add_argument('--sampling-strategy', type=str, default='random',
                        help='Sampling strategy name for sampling_zoo provider.')
    parser.add_argument('--sampling-strategy-params-json', type=str, default='{}',
                        help='JSON object with strategy params for sampling stage.')
    parser.add_argument('--candidate-ratios', type=str, default='0.15,0.2,0.3,0.5',
                        help='Comma-separated candidate ratios for effective-size protocol.')
    parser.add_argument('--delta-threshold', type=float, default=0.03,
                        help='Allowed metric delta threshold for effective-size selection.')
    parser.add_argument('--cap-max-timeout-share', type=float, default=0.35,
                        help='Max timeout share for sampling stage in dynamic cap policy.')

    return parser.parse_args()


def _build_config_from_args(args: argparse.Namespace) -> BenchmarkRunConfig:
    candidate_ratios = parse_ratio_list(args.candidate_ratios)

    try:
        strategy_params = json.loads(args.sampling_strategy_params_json)
    except json.JSONDecodeError as ex:
        raise ValueError(f'Invalid --sampling-strategy-params-json: {ex}')

    if not isinstance(strategy_params, dict):
        raise ValueError('--sampling-strategy-params-json must decode to a JSON object.')

    sampling_config = _default_sampling_config(seed=args.seed)
    sampling_config['strategy'] = args.sampling_strategy
    sampling_config['strategy_params'] = strategy_params
    sampling_config['candidate_ratios'] = list(candidate_ratios)
    sampling_config['delta_metric_threshold'] = args.delta_threshold
    sampling_config['cap_max_timeout_share'] = args.cap_max_timeout_share

    return BenchmarkRunConfig(
        timeout_minutes_per_dataset=args.timeout_minutes,
        output_root=Path(args.output_root),
        seed=args.seed,
        n_jobs=args.n_jobs,
        preset=args.preset,
        with_tuning=not args.disable_tuning,
        dataset_names=tuple(args.datasets),
        amlb_categories=tuple(args.amlb_categories),
        max_rows_per_dataset=args.max_rows,
        include_baseline=not args.disable_baseline,
        include_sampling=not args.disable_sampling,
        sampling_config=sampling_config,
    )


def main() -> None:
    args = _parse_args()
    config = _build_config_from_args(args)
    run_benchmark(config)


if __name__ == '__main__':
    main()
