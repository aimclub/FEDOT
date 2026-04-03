"""Metrics tracking and versioning for model registry."""

import hashlib
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Any

import pandas as pd

logger = logging.getLogger(__name__)

_OBJECTIVE_ATTRS = ['objective', 'params.objective', '_objective']
_METRICS_ATTRS = ['metrics', '_metrics']
_MAX_METRICS_TO_TRY = 10
_HASH_LENGTH = 16


class MetricsTracker:
    """Handles model versioning and metrics tracking."""

    @staticmethod
    def generate_model_id(model=None, model_path: Optional[str] = None) -> str:
        if model is not None:
            return f"model_{id(model)}"
        if model_path is not None:
            return f"path_{hashlib.md5(model_path.encode()).hexdigest()[:_HASH_LENGTH]}"
        return str(uuid.uuid4())

    @staticmethod
    def generate_version() -> str:
        return datetime.utcnow().isoformat()

    @staticmethod
    def sanitize_timestamp(timestamp: str) -> str:
        return timestamp.replace(':', '-')

    @staticmethod
    def find_best_checkpoint(df: pd.DataFrame, metric_name: str, mode: str = "max") -> Optional[dict]:
        if df.empty:
            return None

        def extract_metric(row):
            m = row.get("metrics", {})
            return m.get(metric_name, None) if isinstance(m, dict) else None

        df_copy = df.copy()
        df_copy["_metric_val"] = df_copy.apply(extract_metric, axis=1)
        df_copy = df_copy[df_copy["_metric_val"].notnull()]

        if df_copy.empty:
            return None

        best = df_copy.sort_values(["_metric_val", "version"], ascending=[mode == "min", False]).iloc[0]
        return best.drop(labels=["_metric_val"]).to_dict()

    @staticmethod
    def _get_objective(solver=None, history=None) -> Optional[Any]:
        if solver is not None:
            for attr_path in _OBJECTIVE_ATTRS:
                if '.' in attr_path:
                    parts = attr_path.split('.')
                    obj = solver
                    for part in parts:
                        obj = getattr(obj, part, None) if obj else None
                        if obj is None:
                            break
                    if obj:
                        return obj
                else:
                    obj = getattr(solver, attr_path, None)
                    if obj:
                        return obj

        if history and hasattr(history, 'objective') and history.objective:
            return history.objective

        return None

    @staticmethod
    def _get_num_metrics(history) -> Optional[int]:
        if not (hasattr(history, 'generations') and history.generations):
            return None

        for generation in history.generations:
            if not generation:
                continue

            fitness = None
            if hasattr(generation, '__iter__'):
                individuals = list(generation)
                if individuals and hasattr(individuals[0], 'fitness') and individuals[0].fitness:
                    fitness = individuals[0].fitness
            elif hasattr(generation, 'fitness') and generation.fitness:
                fitness = generation.fitness

            if fitness:
                if hasattr(fitness, 'values') and fitness.values:
                    return len(fitness.values)
                if hasattr(fitness, 'value'):
                    return 1

        return None

    @staticmethod
    def _extract_fitness_values(fitness) -> List[float]:
        candidates = []
        for attr in ('values', 'value', 'values_tuple'):
            candidate = getattr(fitness, attr, None)
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)):
                candidates.extend(candidate)
            else:
                candidates.append(candidate)
        return [float(v) for v in candidates if v is not None]

    @staticmethod
    def _get_best_individual(individuals: List) -> Optional[Any]:
        if not individuals:
            return None

        def get_fitness_value(ind):
            fitness = getattr(ind, 'fitness', None)
            values = MetricsTracker._extract_fitness_values(fitness) if fitness else []
            return values[0] if values else float('inf')

        return min(individuals, key=get_fitness_value)

    @staticmethod
    def _process_generation(generation, gen_idx: int, metric_names: List[str]) -> Optional[dict]:
        if not generation:
            return None

        fitness_values = []

        if hasattr(generation, '__iter__'):
            individuals = list(generation)
            best_individual = MetricsTracker._get_best_individual(individuals)
            fitness_values = MetricsTracker._extract_fitness_values(
                getattr(best_individual, 'fitness', None)) if best_individual else []
        else:
            fitness_values = MetricsTracker._extract_fitness_values(getattr(generation, 'fitness', None))

        if not fitness_values:
            return None

        row = {'generation': gen_idx}
        for i, value in enumerate(fitness_values):
            metric_name = metric_names[i] if i < len(metric_names) else f'metric_{i}'
            row[metric_name] = float(value)

        return row if len(row) > 1 else None

    @staticmethod
    def collect_metrics_from_history(solver=None, history=None) -> pd.DataFrame:
        if history is None:
            if solver is None or not hasattr(solver, 'history') or solver.history is None:
                logger.warning("No solver or history provided for metrics collection")
                return pd.DataFrame()
            history = solver.history

        objective = MetricsTracker._get_objective(solver, history)
        num_metrics = MetricsTracker._get_num_metrics(history)
        metric_names = MetricsTracker._extract_metric_names(objective, num_metrics) if objective else []

        if not metric_names and hasattr(history, 'objective') and history.objective:
            metric_names = list(history.objective.metric_names) if hasattr(history.objective, 'metric_names') else []

        metrics_data = []
        if hasattr(history, 'generations') and history.generations:
            for gen_idx, generation in enumerate(history.generations):
                row = MetricsTracker._process_generation(generation, gen_idx, metric_names)
                if row:
                    metrics_data.append(row)

        if not metrics_data and hasattr(history, 'all_historical_quality'):
            num_metrics_to_try = num_metrics if num_metrics else (len(metric_names) if metric_names else 1)
            for metric_idx in range(min(num_metrics_to_try, _MAX_METRICS_TO_TRY)):
                historical_quality = history.all_historical_quality(metric_idx)
                if historical_quality and len(historical_quality) > 0:
                    metric_name = metric_names[metric_idx] if metric_idx < len(metric_names) else f'metric_{metric_idx}'
                    for gen_idx, value in enumerate(historical_quality):
                        if gen_idx < len(metrics_data):
                            metrics_data[gen_idx][metric_name] = float(value)
                        else:
                            metrics_data.append({'generation': gen_idx, metric_name: float(value)})

        if metrics_data:
            df = pd.DataFrame(metrics_data)
            logger.info(f"Collected {len(df)} rows of metrics with {len(df.columns) - 1} metrics from history")
            return df

        logger.warning("No metrics data collected from history. Available attributes: " +
                       str([attr for attr in dir(history) if not attr.startswith('_')]))
        return pd.DataFrame()

    @staticmethod
    def _extract_metric_name(metric) -> str:
        return str(getattr(metric, 'value', None) or getattr(metric, 'name', None) or metric)

    @staticmethod
    def _extract_metric_names(objective, num_metrics: int = None) -> list:
        metric_names = []

        for attr_name in _METRICS_ATTRS:
            metrics = getattr(objective, attr_name, None)
            if metrics and isinstance(metrics, (list, tuple)):
                metric_names.extend([MetricsTracker._extract_metric_name(m) for m in metrics])
                break

        if not metric_names and hasattr(objective, 'objective_func'):
            obj_func = objective.objective_func
            if hasattr(obj_func, 'metrics') and obj_func.metrics:
                metric_names.extend([MetricsTracker._extract_metric_name(m) for m in obj_func.metrics])

        if not metric_names and hasattr(objective, 'metric_names'):
            metric_names = list(objective.metric_names)

        if num_metrics is not None and len(metric_names) < num_metrics:
            metric_names.extend([f"metric_{i}" for i in range(len(metric_names), num_metrics)])

        return metric_names