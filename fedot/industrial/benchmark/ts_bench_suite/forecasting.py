from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from fedot_ind.core.models.kernel.okhs_common import OKHSMethod, QPolicy, normalize_okhs_method
from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd_forecasting import DMDForecaster

try:  # pragma: no cover - optional heavyweight dependency tree in test envs
    from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_LENGTH, M4_PREFIX, M4_SEASONALITY
except Exception:  # pragma: no cover - lightweight fallback for benchmark-v2
    M4_FORECASTING_LENGTH = {'D': 14, 'W': 13, 'M': 18, 'Q': 8, 'Y': 6}
    M4_SEASONALITY = {'D': 1, 'W': 1, 'M': 12, 'Q': 4, 'Y': 1}
    M4_PREFIX = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}

from .core import (
    BenchmarkAggregateReport,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ForecastingBenchmarkResult,
    ForecastingModelAdapter,
    ForecastingSeriesRecord,
    MetricRecord,
    ModelSpec,
    PredictionRecord,
    RunStatus,
    TaskType,
    new_run_id,
)
from .progress import BenchmarkProgressMonitor

SUPPORTED_FORECASTING_METRICS = ('mase', 'smape', 'owa', 'rmse', 'mae')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_M4_DIR = PROJECT_ROOT / 'examples' / 'data' / 'm4' / 'datasets'
DEFAULT_LOCAL_MONASH_DIR = PROJECT_ROOT / 'examples' / 'data' / 'benchmark' / 'forecasting' / 'monash_benchmark'


class BenchmarkConfigurationError(ValueError):
    pass


class ModelExecutionError(RuntimeError):
    def __init__(self, status: RunStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def validate_forecasting_suite_config(config: BenchmarkSuiteConfig) -> None:
    if config.task_type is not TaskType.FORECASTING:
        raise BenchmarkConfigurationError('Forecasting suite expects task_type=forecasting.')
    if not config.datasets:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one dataset spec.')
    if not config.models:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one model spec.')
    unsupported = set(config.metrics) - set(SUPPORTED_FORECASTING_METRICS)
    if unsupported:
        raise BenchmarkConfigurationError(f'Unsupported forecasting metrics: {sorted(unsupported)}')


def _normalize_subset_name(subset: str) -> str:
    mapping = {
        'd': 'Daily',
        'daily': 'Daily',
        'w': 'Weekly',
        'weekly': 'Weekly',
        'm': 'Monthly',
        'monthly': 'Monthly',
        'q': 'Quarterly',
        'quarterly': 'Quarterly',
        'y': 'Yearly',
        'yearly': 'Yearly',
    }
    return mapping.get(subset.lower(), subset)


def _infer_m4_key(subset: str) -> str:
    normalized = _normalize_subset_name(subset)
    reverse = {value: key for key, value in M4_PREFIX.items()}
    if normalized not in reverse:
        raise BenchmarkConfigurationError(f'Unsupported M4 subset: {subset}')
    return reverse[normalized]


def _sample_records(
        records: list[ForecastingSeriesRecord],
        spec: DatasetSpec,
) -> tuple[ForecastingSeriesRecord, ...]:
    filtered = records
    if spec.series_ids:
        requested = set(spec.series_ids)
        filtered = [record for record in filtered if record.series_id in requested]
    if spec.sample_size is not None and len(filtered) > spec.sample_size:
        rng = np.random.default_rng(spec.random_seed)
        indices = rng.choice(len(filtered), size=spec.sample_size, replace=False)
        filtered = [filtered[index] for index in sorted(indices)]
    return tuple(filtered)


def _series_split_from_full_values(
        values: np.ndarray,
        forecast_horizon: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    normalized = np.asarray(values, dtype=float).reshape(-1)
    if len(normalized) <= forecast_horizon:
        raise BenchmarkConfigurationError(
            f'Series length {len(normalized)} must be greater than forecast horizon {forecast_horizon}.'
        )
    train = tuple(float(value) for value in normalized[:-forecast_horizon])
    test = tuple(float(value) for value in normalized[-forecast_horizon:])
    return train, test


def _parse_frame_records(
        frame: pd.DataFrame,
        *,
        benchmark: str,
        dataset_name: str,
        subset: str,
        default_frequency: str,
        default_horizon: int,
        default_seasonal_period: int,
) -> list[ForecastingSeriesRecord]:
    identifier_column = 'series_id'
    for candidate in ('series_id', 'unique_id', 'item_id'):
        if candidate in frame.columns:
            identifier_column = candidate
            break
    if identifier_column not in frame.columns:
        raise BenchmarkConfigurationError('Dataset frame must include series_id or unique_id column.')

    value_column = 'value'
    for candidate in ('value', 'y', 'target'):
        if candidate in frame.columns:
            value_column = candidate
            break
    if value_column not in frame.columns:
        raise BenchmarkConfigurationError('Dataset frame must include value/y/target column.')

    sort_columns = [candidate for candidate in ('timestamp', 'datetime', 'ds') if candidate in frame.columns]
    records: list[ForecastingSeriesRecord] = []
    for series_id, group in frame.groupby(identifier_column):
        ordered = group.sort_values(sort_columns) if sort_columns else group
        series_values = ordered[value_column].astype(float).to_numpy()
        horizon = int(ordered['horizon'].iloc[0]) if 'horizon' in ordered.columns else default_horizon
        seasonal_period = (
            int(ordered['seasonal_period'].iloc[0])
            if 'seasonal_period' in ordered.columns
            else default_seasonal_period
        )
        frequency = str(ordered['frequency'].iloc[0]) if 'frequency' in ordered.columns else default_frequency
        record_dataset_name = (
            str(ordered['dataset_name'].iloc[0]) if 'dataset_name' in ordered.columns else dataset_name
        )
        train_values, test_values = _series_split_from_full_values(series_values, horizon)
        records.append(
            ForecastingSeriesRecord(
                benchmark=benchmark,
                dataset_name=record_dataset_name,
                subset=subset,
                series_id=str(series_id),
                frequency=frequency,
                forecast_horizon=horizon,
                seasonal_period=seasonal_period,
                train_values=train_values,
                test_values=test_values,
                metadata={'split_provenance': 'tail_holdout_from_full_series'},
            )
        )
    return records


def _parse_sequence_records(
        payload: list[dict[str, Any]],
        *,
        benchmark: str,
        dataset_name: str,
        subset: str,
        default_frequency: str,
        default_horizon: int,
        default_seasonal_period: int,
) -> list[ForecastingSeriesRecord]:
    records: list[ForecastingSeriesRecord] = []
    for index, item in enumerate(payload):
        series_id = str(item.get('series_id', item.get('unique_id', f'{dataset_name}_{index}')))
        frequency = str(item.get('frequency', default_frequency))
        horizon = int(item.get('horizon', default_horizon))
        seasonal_period = int(item.get('seasonal_period', default_seasonal_period))
        values = item.get('values')
        train = item.get('train_values')
        test = item.get('test_values')
        if values is not None:
            train_values, test_values = _series_split_from_full_values(np.asarray(values, dtype=float), horizon)
        elif train is not None and test is not None:
            train_values = tuple(float(value) for value in np.asarray(train, dtype=float).reshape(-1))
            test_values = tuple(float(value) for value in np.asarray(test, dtype=float).reshape(-1))
        else:
            raise BenchmarkConfigurationError('Sequence payload must include values or train_values/test_values.')

        records.append(
            ForecastingSeriesRecord(
                benchmark=benchmark,
                dataset_name=str(item.get('dataset_name', dataset_name)),
                subset=subset,
                series_id=series_id,
                frequency=frequency,
                forecast_horizon=horizon,
                seasonal_period=seasonal_period,
                train_values=train_values,
                test_values=test_values,
                metadata={'split_provenance': item.get('split_provenance', 'adapter_provided')},
            )
        )
    return records


class M4Adapter:
    benchmark_name = 'm4'

    def __init__(self, loader: Callable[[DatasetSpec], Any] | None = None):
        self.loader = loader or self._default_loader

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        subset = _normalize_subset_name(spec.subset)
        key = _infer_m4_key(subset)
        local_csv_dir = spec.adapter_options.get('local_csv_dir')
        use_local_files = spec.adapter_options.get('use_local_files', False)
        if local_csv_dir or use_local_files:
            records = self._load_local_records(spec, subset, key)
        else:
            frame = self.loader(spec)
            records = _parse_frame_records(
                frame=frame,
                benchmark=self.benchmark_name,
                dataset_name=spec.dataset_name,
                subset=subset,
                default_frequency=subset,
                default_horizon=M4_FORECASTING_LENGTH[key],
                default_seasonal_period=M4_SEASONALITY[key],
            )
        return _sample_records(records, spec)

    def _load_local_records(
            self,
            spec: DatasetSpec,
            subset: str,
            key: str,
    ) -> list[ForecastingSeriesRecord]:
        base_dir = Path(spec.adapter_options.get('local_csv_dir', DEFAULT_LOCAL_M4_DIR))
        train_path = base_dir / f'{subset}-train.csv'
        test_path = base_dir / f'{subset}-test.csv'
        if not train_path.exists() or not test_path.exists():
            raise BenchmarkConfigurationError(
                f'Local M4 files were not found: {train_path} and {test_path}.'
            )

        train_frame = pd.read_csv(train_path)
        test_frame = pd.read_csv(test_path)
        if 'V1' not in train_frame.columns or 'V1' not in test_frame.columns:
            raise BenchmarkConfigurationError('Local M4 CSV files must include the V1 identifier column.')

        train_frame = train_frame.set_index('V1')
        test_frame = test_frame.set_index('V1')
        common_ids = [series_id for series_id in train_frame.index if series_id in test_frame.index]
        if not common_ids:
            raise BenchmarkConfigurationError('Local M4 train/test files do not share any series identifiers.')

        records: list[ForecastingSeriesRecord] = []
        for series_id in common_ids:
            train_values = train_frame.loc[series_id].dropna().astype(float).to_numpy()
            test_values = test_frame.loc[series_id].dropna().astype(float).to_numpy()
            if len(test_values) == 0 or len(train_values) == 0:
                continue
            records.append(
                ForecastingSeriesRecord(
                    benchmark=self.benchmark_name,
                    dataset_name=spec.dataset_name or subset,
                    subset=subset,
                    series_id=str(series_id),
                    frequency=subset,
                    forecast_horizon=int(len(test_values)),
                    seasonal_period=M4_SEASONALITY[key],
                    train_values=tuple(float(value) for value in train_values),
                    test_values=tuple(float(value) for value in test_values),
                    metadata={'split_provenance': 'local_m4_train_test_csv'},
                )
            )
        return records

    @staticmethod
    def _default_loader(spec: DatasetSpec) -> pd.DataFrame:
        try:
            from datasetsforecast.m4 import M4
        except Exception as exc:  # pragma: no cover
            raise BenchmarkConfigurationError(f'M4 adapter is unavailable: {exc}') from exc

        subset = _normalize_subset_name(spec.subset)
        data_directory = spec.adapter_options.get('data_dir')
        frame, _, _ = M4.load(directory=data_directory, group=subset)
        return frame


class MonashAdapter:
    benchmark_name = 'monash'

    def __init__(self, loader: Callable[[DatasetSpec], Any] | None = None):
        self.loader = loader or self._default_loader

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        local_csv_path = spec.adapter_options.get('local_csv_path')
        use_local_files = spec.adapter_options.get('use_local_files', False)
        if local_csv_path or use_local_files:
            records = self._load_local_records(spec)
        else:
            payload = self.loader(spec)
            if isinstance(payload, pd.DataFrame):
                records = _parse_frame_records(
                    frame=payload,
                    benchmark=self.benchmark_name,
                    dataset_name=spec.dataset_name,
                    subset=spec.subset,
                    default_frequency=str(spec.adapter_options.get('frequency', spec.subset)),
                    default_horizon=int(spec.adapter_options.get('forecast_horizon', 1)),
                    default_seasonal_period=int(spec.adapter_options.get('seasonal_period', 1)),
                )
            else:
                records = _parse_sequence_records(
                    payload=list(payload),
                    benchmark=self.benchmark_name,
                    dataset_name=spec.dataset_name,
                    subset=spec.subset,
                    default_frequency=str(spec.adapter_options.get('frequency', spec.subset)),
                    default_horizon=int(spec.adapter_options.get('forecast_horizon', 1)),
                    default_seasonal_period=int(spec.adapter_options.get('seasonal_period', 1)),
                )
        return _sample_records(records, spec)

    def _load_local_records(self, spec: DatasetSpec) -> list[ForecastingSeriesRecord]:
        local_path = spec.adapter_options.get('local_csv_path')
        if local_path is None:
            file_candidates = sorted(DEFAULT_LOCAL_MONASH_DIR.glob(f'*{spec.dataset_name}*.csv'))
            if not file_candidates:
                raise BenchmarkConfigurationError(
                    f'No local Monash file matching dataset_name={spec.dataset_name} was found in {DEFAULT_LOCAL_MONASH_DIR}.'
                )
            local_path = file_candidates[0]
        csv_path = Path(local_path)
        if not csv_path.exists():
            raise BenchmarkConfigurationError(f'Local Monash CSV file was not found: {csv_path}')

        frame = pd.read_csv(csv_path)
        if 'label' not in frame.columns or 'value' not in frame.columns:
            raise BenchmarkConfigurationError('Local Monash CSV must include `label` and `value` columns.')

        frequency = str(spec.adapter_options.get('frequency', spec.subset))
        horizon = int(spec.adapter_options.get('forecast_horizon', self._infer_horizon_from_filename(csv_path)))
        seasonal_period = int(spec.adapter_options.get('seasonal_period', self._infer_seasonal_period(frequency)))
        dataset_name = spec.dataset_name or csv_path.stem

        records: list[ForecastingSeriesRecord] = []
        sort_columns = [column for column in ('datetime', 'timestamp', 'ds') if column in frame.columns]
        for series_id, group in frame.groupby('label'):
            ordered = group.sort_values(sort_columns) if sort_columns else group
            values = ordered['value'].astype(float).to_numpy()
            if len(values) <= horizon:
                continue
            train_values, test_values = _series_split_from_full_values(values, horizon)
            records.append(
                ForecastingSeriesRecord(
                    benchmark=self.benchmark_name,
                    dataset_name=dataset_name,
                    subset=spec.subset,
                    series_id=str(series_id),
                    frequency=frequency,
                    forecast_horizon=horizon,
                    seasonal_period=seasonal_period,
                    train_values=train_values,
                    test_values=test_values,
                    metadata={
                        'split_provenance': 'local_monash_csv',
                        'source_file': csv_path.name,
                    },
                )
            )
        return records

    @staticmethod
    def _infer_horizon_from_filename(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.rsplit('_', maxsplit=1)[-1])
        except Exception:
            return 1

    @staticmethod
    def _infer_seasonal_period(frequency: str) -> int:
        normalized = frequency.lower()
        if 'month' in normalized:
            return 12
        if 'quarter' in normalized:
            return 4
        if 'week' in normalized:
            return 52
        if 'day' in normalized:
            return 7
        if 'hour' in normalized:
            return 24
        return 1

    @staticmethod
    def _default_loader(spec: DatasetSpec) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover
            raise BenchmarkConfigurationError(f'Monash adapter is unavailable: {exc}') from exc

        dataset_path = spec.adapter_options.get('dataset_path', spec.dataset_name)
        split_name = spec.adapter_options.get('split', 'train')
        dataset = load_dataset(dataset_path, split=split_name)
        payload: list[dict[str, Any]] = []
        for row in dataset:
            payload.append(
                {
                    'series_id': row.get('item_id', row.get('series_id', row.get('unique_id'))),
                    'values': row.get('target', row.get('values')),
                    'horizon': row.get('horizon', spec.adapter_options.get('forecast_horizon', 1)),
                    'frequency': row.get('freq', row.get('frequency', spec.subset)),
                    'seasonal_period': row.get(
                        'seasonal_period',
                        spec.adapter_options.get('seasonal_period', 1),
                    ),
                    'dataset_name': spec.dataset_name,
                    'split_provenance': 'monash_dataset_loader',
                }
            )
        return payload


class InMemoryForecastingAdapter:
    benchmark_name = 'in_memory'

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        payload = list(spec.adapter_options.get('records', ()))
        if not payload:
            raise BenchmarkConfigurationError('InMemory adapter requires adapter_options["records"].')
        records = _parse_sequence_records(
            payload=payload,
            benchmark=self.benchmark_name,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            default_frequency=str(spec.adapter_options.get('frequency', spec.subset)),
            default_horizon=int(spec.adapter_options.get('forecast_horizon', 1)),
            default_seasonal_period=int(spec.adapter_options.get('seasonal_period', 1)),
        )
        return _sample_records(records, spec)


def _safe_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


@dataclass
class NaiveLastValueModel(ForecastingModelAdapter):
    name: str = 'NaiveLastValue'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = np.asarray(series_record.train_values, dtype=float)
        return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'last_value'}


@dataclass
class NaiveMeanModel(ForecastingModelAdapter):
    name: str = 'NaiveMean'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = np.asarray(series_record.train_values, dtype=float)
        return np.full(series_record.forecast_horizon, np.mean(train), dtype=float), {'strategy': 'mean'}


@dataclass
class NaiveDriftModel(ForecastingModelAdapter):
    name: str = 'NaiveDrift'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = np.asarray(series_record.train_values, dtype=float)
        if len(train) <= 1:
            return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'fallback'}
        slope = (train[-1] - train[0]) / max(len(train) - 1, 1)
        horizon = np.arange(1, series_record.forecast_horizon + 1, dtype=float)
        return train[-1] + slope * horizon, {'strategy': 'drift'}


@dataclass
class MovingAverageModel(ForecastingModelAdapter):
    window_size: int = 3
    name: str = 'MovingAverage'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = np.asarray(series_record.train_values, dtype=float)
        window = min(max(self.window_size, 1), len(train))
        return np.full(series_record.forecast_horizon, np.mean(train[-window:]), dtype=float), {'window_size': window}


@dataclass
class LinearTrendModel(ForecastingModelAdapter):
    name: str = 'LinearTrend'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = np.asarray(series_record.train_values, dtype=float)
        if len(train) <= 1:
            return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'constant'}
        index = np.arange(len(train), dtype=float)
        slope, intercept = np.polyfit(index, train, deg=1)
        future_index = np.arange(len(train), len(train) + series_record.forecast_horizon, dtype=float)
        return intercept + slope * future_index, {'slope': float(slope), 'intercept': float(intercept)}


@dataclass
class ClassicalDMDModel(ForecastingModelAdapter):
    window_size: int = 12
    n_modes: int | None = None
    name: str = 'ClassicalDMD'
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'dmd')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        if not _safe_import('torch'):
            return RunStatus.NOT_AVAILABLE, 'torch is required for DMDForecaster.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = np.asarray(series_record.train_values, dtype=float)
        window_size = min(max(self.window_size, series_record.forecast_horizon + 1), len(train) - 1)
        if window_size <= series_record.forecast_horizon:
            raise ModelExecutionError(RunStatus.SKIPPED, 'Series is too short for ClassicalDMD windowing.')
        trajectories = np.array(
            [train[index:index + window_size] for index in range(len(train) - window_size + 1)],
            dtype=float,
        )
        model = DMDForecaster(
            forecast_horizon=series_record.forecast_horizon,
            n_modes=self.n_modes,
            use_koopman=False,
            epochs=1,
            device='cpu',
        )
        model.fit(trajectories, window_size=window_size)
        prediction = np.asarray(model.predict(train[-window_size:]), dtype=float).reshape(-1)
        return prediction[:series_record.forecast_horizon], {'window_size': window_size}


@dataclass
class OKHSModel(ForecastingModelAdapter):
    method: str | OKHSMethod = OKHSMethod.DMD
    q: float = 0.7
    q_policy: str | QPolicy = QPolicy.FIXED
    n_modes: int = 5
    window_size: int = 20
    window_policy: str = 'adaptive_cycle_aware'
    trajectory_representation_policy: str = 'projected'
    latent_trajectory_stride_policy: str = 'adaptive'
    latent_trajectory_stride: int | None = None
    mode_selection_policy: str = "energy"
    mode_energy_threshold: float = 0.95
    prediction_mode_selection_policy: str = "adaptive_tail_energy"
    max_prediction_modes: int | None = None
    min_prediction_modes: int = 4
    boundary_alignment_policy: str = "tapered_offset"
    boundary_alignment_decay: float = 4.0
    prediction_stability_threshold: float | None = 0.03
    name: str = 'OKHS'
    tags: tuple[str, ...] = ('okhs', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = np.asarray(series_record.train_values, dtype=float)
        window_size = min(max(self.window_size, 4), len(train) - 1)
        if window_size < 2:
            raise ModelExecutionError(RunStatus.SKIPPED, 'Series is too short for OKHS forecasting.')
        method = normalize_okhs_method(self.method)
        model = OKHSForecaster(
            q=self.q,
            forecast_horizon=series_record.forecast_horizon,
            n_modes=self.n_modes,
            method=method,
            q_policy=self.q_policy,
            window_policy=self.window_policy,
            trajectory_representation_policy=self.trajectory_representation_policy,
            latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
            latent_trajectory_stride=self.latent_trajectory_stride,
            mode_selection_policy=self.mode_selection_policy,
            mode_energy_threshold=self.mode_energy_threshold,
            prediction_mode_selection_policy=self.prediction_mode_selection_policy,
            max_prediction_modes=self.max_prediction_modes,
            min_prediction_modes=self.min_prediction_modes,
            boundary_alignment_policy=self.boundary_alignment_policy,
            boundary_alignment_decay=self.boundary_alignment_decay,
            prediction_stability_threshold=self.prediction_stability_threshold,
        )
        model.fit(train, window_size=window_size)
        prediction = np.asarray(model.predict(train), dtype=float).reshape(-1)
        forecast = prediction[:series_record.forecast_horizon]
        metadata = {
            **model.get_optimization_info(),
            'selected_q': float(getattr(model, 'resolved_q_', self.q)),
            'method': str(method),
            'window_size': window_size,
            'last_train_value': float(train[-1]),
            'first_prediction_value': float(forecast[0]) if len(forecast) else None,
            'first_actual_value': float(series_record.test_values[0]) if series_record.test_values else None,
        }
        if metadata['first_prediction_value'] is not None:
            metadata['first_step_delta'] = float(metadata['first_prediction_value'] - metadata['last_train_value'])
        if metadata['first_actual_value'] is not None:
            metadata['first_actual_delta'] = float(metadata['first_actual_value'] - metadata['last_train_value'])
        return forecast, metadata


@dataclass
class OptionalExternalModel(ForecastingModelAdapter):
    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'external')
    optional: bool = True
    scaffold_reason: str = 'Adapter scaffold is registered but backend training is not wired yet.'

    def availability(self) -> tuple[RunStatus, str]:
        if not _safe_import(self.dependency_name):
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'
        return RunStatus.SUCCESS, 'dependency is available'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        del series_record
        raise ModelExecutionError(RunStatus.SKIPPED, self.scaffold_reason)


def build_dataset_adapter(spec: DatasetSpec):
    benchmark = spec.benchmark.lower()
    custom_loader = spec.adapter_options.get('loader')
    if benchmark == 'm4':
        return M4Adapter(loader=custom_loader)
    if benchmark == 'monash':
        return MonashAdapter(loader=custom_loader)
    if benchmark == 'in_memory':
        return InMemoryForecastingAdapter()
    raise BenchmarkConfigurationError(f'Unsupported forecasting benchmark adapter: {spec.benchmark}')


def build_model_adapter(spec: ModelSpec) -> ForecastingModelAdapter:
    adapter_name = spec.adapter_name.lower()
    params = dict(spec.params)
    if adapter_name == 'okhs':
        return OKHSModel(name=spec.display_name, tags=spec.tags or ('okhs', 'forecasting'), **params)
    if adapter_name == 'naive_last_value':
        return NaiveLastValueModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'naive_mean':
        return NaiveMeanModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'naive_drift':
        return NaiveDriftModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'moving_average':
        return MovingAverageModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'), **params)
    if adapter_name == 'linear_trend':
        return LinearTrendModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'classical_dmd':
        return ClassicalDMDModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting', 'dmd'), **params)
    if adapter_name == 'autogluon':
        return OptionalExternalModel(
            dependency_name='autogluon',
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting', 'external', 'autogluon'),
        )
    if adapter_name == 'nbeats':
        return OptionalExternalModel(
            dependency_name='neuralforecast',
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting', 'external', 'nbeats'),
        )
    if adapter_name == 'tft':
        return OptionalExternalModel(
            dependency_name='pytorch_forecasting',
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting', 'external', 'tft'),
        )
    raise BenchmarkConfigurationError(f'Unsupported forecasting model adapter: {spec.adapter_name}')


def _seasonal_naive_forecast(train: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    base = train[-lag:]
    repeats = int(math.ceil(horizon / lag))
    return np.tile(base, repeats)[:horizon]


def _mase_scale(train: np.ndarray, seasonal_period: int) -> float:
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    if len(train) <= lag:
        return 1.0
    scale = np.mean(np.abs(train[lag:] - train[:-lag]))
    return float(scale if scale > 1e-8 else 1.0)


def compute_forecasting_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> float:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    train = np.asarray(y_train, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise BenchmarkConfigurationError('Metric inputs must have the same length.')
    if metric_name == 'mae':
        return float(np.mean(np.abs(actual - predicted)))
    if metric_name == 'rmse':
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    if metric_name == 'smape':
        denominator = np.abs(actual) + np.abs(predicted)
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return float(100.0 * np.mean(2.0 * np.abs(predicted - actual) / denominator))
    if metric_name == 'mase':
        return float(np.mean(np.abs(actual - predicted)) / _mase_scale(train, seasonal_period))
    if metric_name == 'owa':
        baseline = _seasonal_naive_forecast(train, len(actual), seasonal_period)
        smape = compute_forecasting_metric('smape', actual, predicted, train, seasonal_period)
        mase = compute_forecasting_metric('mase', actual, predicted, train, seasonal_period)
        baseline_smape = compute_forecasting_metric('smape', actual, baseline, train, seasonal_period)
        baseline_mase = compute_forecasting_metric('mase', actual, baseline, train, seasonal_period)
        baseline_smape = baseline_smape if baseline_smape > 1e-8 else 1.0
        baseline_mase = baseline_mase if baseline_mase > 1e-8 else 1.0
        return float(0.5 * ((smape / baseline_smape) + (mase / baseline_mase)))
    raise BenchmarkConfigurationError(f'Unsupported metric: {metric_name}')


def compute_pointwise_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> np.ndarray:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if metric_name == 'mae':
        return np.abs(actual - predicted)
    if metric_name == 'rmse':
        return np.sqrt((actual - predicted) ** 2)
    if metric_name == 'smape':
        denominator = np.abs(actual) + np.abs(predicted)
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return 100.0 * 2.0 * np.abs(predicted - actual) / denominator
    if metric_name == 'mase':
        return np.abs(actual - predicted) / _mase_scale(np.asarray(y_train, dtype=float), seasonal_period)
    if metric_name == 'owa':
        train = np.asarray(y_train, dtype=float)
        baseline = _seasonal_naive_forecast(train, len(actual), seasonal_period)
        pointwise_smape = compute_pointwise_metric('smape', actual, predicted, train, seasonal_period)
        pointwise_mase = compute_pointwise_metric('mase', actual, predicted, train, seasonal_period)
        baseline_smape = compute_pointwise_metric('smape', actual, baseline, train, seasonal_period)
        baseline_mase = compute_pointwise_metric('mase', actual, baseline, train, seasonal_period)
        baseline_smape = np.where(baseline_smape <= 1e-8, 1.0, baseline_smape)
        baseline_mase = np.where(baseline_mase <= 1e-8, 1.0, baseline_mase)
        return 0.5 * ((pointwise_smape / baseline_smape) + (pointwise_mase / baseline_mase))
    raise BenchmarkConfigurationError(f'Unsupported metric: {metric_name}')


def build_leaderboard(
        run_records: tuple[BenchmarkRunRecord, ...],
        primary_metric: str,
) -> BenchmarkAggregateReport:
    successful = [record for record in run_records if record.status is RunStatus.SUCCESS]
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for record in successful:
        metric_value = record.metrics_summary.get(primary_metric)
        if metric_value is None:
            continue
        grouped.setdefault((record.benchmark, record.dataset_name, record.model_name), []).append(metric_value)

    leaderboard_rows = []
    for (benchmark, dataset_name, model_name), values in grouped.items():
        leaderboard_rows.append(
            {
                'benchmark': benchmark,
                'dataset_name': dataset_name,
                'model_name': model_name,
                primary_metric: float(np.mean(values)),
                'n_series': len(values),
            }
        )

    leaderboard_rows = sorted(leaderboard_rows, key=lambda row: row[primary_metric])
    for rank, row in enumerate(leaderboard_rows, start=1):
        row['rank'] = rank

    status_counts: dict[str, int] = {}
    for record in run_records:
        status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1

    run_id = run_records[0].run_id if run_records else new_run_id('empty')
    return BenchmarkAggregateReport(
        run_id=run_id,
        task_type=TaskType.FORECASTING,
        primary_metric=primary_metric,
        leaderboard_rows=tuple(leaderboard_rows),
        status_counts=status_counts,
    )


def run_forecasting_suite(config: BenchmarkSuiteConfig) -> ForecastingBenchmarkResult:
    validate_forecasting_suite_config(config)
    run_id = new_run_id(config.run_spec.run_name)

    series_records: list[ForecastingSeriesRecord] = []
    run_records: list[BenchmarkRunRecord] = []
    prediction_records: list[PredictionRecord] = []
    metric_records: list[MetricRecord] = []
    progress = BenchmarkProgressMonitor(
        enabled=config.run_spec.show_progress,
        task_type=config.task_type.value,
        run_name=config.run_spec.run_name,
        leave=config.run_spec.progress_leave,
        log_errors=config.run_spec.progress_log_errors,
        log_summaries=config.run_spec.progress_log_summaries,
    )

    try:
        for dataset_spec in config.datasets:
            dataset_adapter = build_dataset_adapter(dataset_spec)
            dataset_series = dataset_adapter.load_series(dataset_spec)
            series_records.extend(dataset_series)
            progress.extend_total(len(dataset_series) * len(config.models))
            progress.dataset_loaded(dataset_spec.dataset_name, len(dataset_series))

            for model_spec in config.models:
                model = build_model_adapter(model_spec)
                progress.model_started(dataset_spec.dataset_name, model.name)
                availability_status, availability_message = model.availability()

                if availability_status is not RunStatus.SUCCESS:
                    for series_record in dataset_series:
                        progress.item_started(series_record.dataset_name, model.name, series_record.series_id)
                        run_records.append(
                            BenchmarkRunRecord(
                                run_id=run_id,
                                benchmark=series_record.benchmark,
                                dataset_name=series_record.dataset_name,
                                subset=series_record.subset,
                                series_id=series_record.series_id,
                                model_name=model.name,
                                status=availability_status,
                                tags=model.tags,
                                message=availability_message,
                                metadata={'optional': model.optional},
                            )
                        )
                        progress.advance(availability_status.value, availability_message)
                    progress.model_finished()
                    continue

                for series_record in dataset_series:
                    progress.item_started(series_record.dataset_name, model.name, series_record.series_id)
                    try:
                        prediction, metadata = model.forecast(series_record)
                        actual = np.asarray(series_record.test_values, dtype=float)
                        forecast = np.asarray(prediction, dtype=float).reshape(-1)[: len(actual)]
                        if len(forecast) != len(actual):
                            raise ModelExecutionError(
                                RunStatus.FAILED,
                                f'Model returned {len(forecast)} predictions for horizon {len(actual)}.',
                            )

                        metrics_summary: dict[str, float] = {}
                        train = np.asarray(series_record.train_values, dtype=float)
                        for metric_name in config.metrics:
                            metric_value = compute_forecasting_metric(
                                metric_name,
                                actual,
                                forecast,
                                train,
                                series_record.seasonal_period,
                            )
                            metrics_summary[metric_name] = metric_value
                            metric_records.append(
                                MetricRecord(
                                    run_id=run_id,
                                    benchmark=series_record.benchmark,
                                    dataset_name=series_record.dataset_name,
                                    subset=series_record.subset,
                                    series_id=series_record.series_id,
                                    model_name=model.name,
                                    metric_name=metric_name,
                                    metric_value=metric_value,
                                    status=RunStatus.SUCCESS,
                                )
                            )
                            pointwise = compute_pointwise_metric(
                                metric_name,
                                actual,
                                forecast,
                                train,
                                series_record.seasonal_period,
                            )
                            for horizon_index, pointwise_value in enumerate(pointwise, start=1):
                                metric_records.append(
                                    MetricRecord(
                                        run_id=run_id,
                                        benchmark=series_record.benchmark,
                                        dataset_name=series_record.dataset_name,
                                        subset=series_record.subset,
                                        series_id=series_record.series_id,
                                        model_name=model.name,
                                        metric_name=metric_name,
                                        metric_value=float(pointwise_value),
                                        status=RunStatus.SUCCESS,
                                        horizon_index=horizon_index,
                                    )
                                )

                        for horizon_index, (actual_value, forecast_value) in enumerate(zip(actual, forecast), start=1):
                            prediction_records.append(
                                PredictionRecord(
                                    run_id=run_id,
                                    benchmark=series_record.benchmark,
                                    dataset_name=series_record.dataset_name,
                                    subset=series_record.subset,
                                    series_id=series_record.series_id,
                                    model_name=model.name,
                                    horizon_index=horizon_index,
                                    y_true=float(actual_value),
                                    y_pred=float(forecast_value),
                                    status=RunStatus.SUCCESS,
                                )
                            )

                        run_records.append(
                            BenchmarkRunRecord(
                                run_id=run_id,
                                benchmark=series_record.benchmark,
                                dataset_name=series_record.dataset_name,
                                subset=series_record.subset,
                                series_id=series_record.series_id,
                                model_name=model.name,
                                status=RunStatus.SUCCESS,
                                tags=model.tags,
                                metrics_summary=metrics_summary,
                                metadata=metadata,
                            )
                        )
                        progress.advance(RunStatus.SUCCESS.value)
                    except ModelExecutionError as exc:
                        run_records.append(
                            BenchmarkRunRecord(
                                run_id=run_id,
                                benchmark=series_record.benchmark,
                                dataset_name=series_record.dataset_name,
                                subset=series_record.subset,
                                series_id=series_record.series_id,
                                model_name=model.name,
                                status=exc.status,
                                tags=model.tags,
                                message=exc.message,
                                metadata={'optional': model.optional},
                            )
                        )
                        progress.advance(exc.status.value, exc.message)
                    except Exception as exc:
                        run_records.append(
                            BenchmarkRunRecord(
                                run_id=run_id,
                                benchmark=series_record.benchmark,
                                dataset_name=series_record.dataset_name,
                                subset=series_record.subset,
                                series_id=series_record.series_id,
                                model_name=model.name,
                                status=RunStatus.FAILED,
                                tags=model.tags,
                                message=str(exc),
                                metadata={'optional': model.optional},
                            )
                        )
                        progress.advance(RunStatus.FAILED.value, str(exc))
                progress.model_finished()
            progress.dataset_finished()
    finally:
        progress.close()

    aggregate_report = build_leaderboard(tuple(run_records), primary_metric=config.run_spec.primary_metric)
    return ForecastingBenchmarkResult(
        run_id=run_id,
        config=config,
        series_records=tuple(series_records),
        run_records=tuple(run_records),
        prediction_records=tuple(prediction_records),
        metric_records=tuple(metric_records),
        aggregate_report=aggregate_report,
    )
