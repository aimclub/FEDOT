import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from golem.core.log import LoggerAdapter, default_log
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split

from fedot.api.sampling_stage.config import SamplingConfig, validate_sampling_config
from fedot.api.sampling_stage.providers import SamplingProvider, SamplingZooProvider
from fedot.core.data.data import InputData, InputDataList, data_type_is_table
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass
class SamplingStageOutput:
    train_data: Union[InputData, InputDataList]
    metadata: Dict[str, Any]
    elapsed_seconds: float
    updated_timeout_minutes: Optional[float]


class SamplingStageExecutor:
    def __init__(self,
                 sampling_config: Dict[str, Any],
                 task_type: TaskTypesEnum,
                 total_timeout_minutes: Optional[float],
                 log: Optional[LoggerAdapter] = None,
                 provider: Optional[SamplingProvider] = None):
        self.config: SamplingConfig = validate_sampling_config(sampling_config)
        if self.config is None:
            raise ValueError('Sampling stage config must not be None when executor is created.')

        self.task_type = task_type
        self.total_timeout_minutes = total_timeout_minutes
        self.log = log or default_log(self)
        self.provider = provider or self._create_provider(self.config.provider)

    def execute(self, train_data: InputData) -> SamplingStageOutput:
        self._validate_task_compatibility(train_data)

        started_at = time.perf_counter()
        budget_seconds = self._compute_budget_seconds(self.config, self.total_timeout_minutes)

        available_methods = {
            'chunking': lambda: SamplingStageExecutor._execute_chunking(
                train_data=train_data,
                started_at=started_at,
                budget_seconds=budget_seconds,
                provider=self.provider,
                config=self.config,
                total_timeout_minutes=self.total_timeout_minutes
            ),
            'subset': lambda: SamplingStageExecutor._execute_subset(
                train_data=train_data,
                started_at=started_at,
                budget_seconds=budget_seconds,
                provider=self.provider,
                config=self.config,
                task_type=self.task_type,
                total_timeout_minutes=self.total_timeout_minutes
            ),
        }
        execute_method = available_methods.get(self.config.strategy_kind)
        if execute_method is None:
            raise ValueError(f'Unknown strategy_kind: {self.config.strategy_kind}')
        return execute_method()

    @staticmethod
    def _execute_subset(train_data: InputData,
                        started_at: float,
                        budget_seconds: float,
                        provider: SamplingProvider,
                        config: SamplingConfig,
                        task_type: TaskTypesEnum,
                        total_timeout_minutes: Optional[float]) -> SamplingStageOutput:
        effective_size_result = SamplingStageExecutor._select_effective_ratio(
            train_data=train_data,
            started_at=started_at,
            budget_seconds=budget_seconds,
            provider=provider,
            config=config,
            task_type=task_type,
        )

        SamplingStageExecutor._raise_if_budget_exceeded(started_at, budget_seconds)
        remaining_budget_value = SamplingStageExecutor._remaining_budget(started_at, budget_seconds)
        final_provider_result = provider.sample(
            features=np.asarray(train_data.features),
            target=SamplingStageExecutor._flatten_target(train_data.target),
            strategy=config.strategy,
            strategy_params=config.strategy_params,
            random_state=config.random_state,
            budget_seconds=remaining_budget_value,
            strategy_kind=config.strategy_kind,
            injectable_params={'ratio': effective_size_result['selected_ratio']},
        )

        selected_indices = SamplingStageExecutor._validate_indices(final_provider_result.sample_indices,
                                                                   upper_bound=len(train_data.idx),
                                                                   data_label='full train data')
        reduced_data = SamplingStageExecutor._subset_by_positions(train_data, selected_indices)

        elapsed_seconds = time.perf_counter() - started_at
        timeout_after_stage = SamplingStageExecutor._compute_updated_timeout(
            elapsed_seconds, total_timeout_minutes, config.min_automl_time_minutes
        )

        metadata = {
            'status': 'applied',
            'provider': config.provider,
            'strategy': config.strategy,
            'selected_ratio': effective_size_result['selected_ratio'],
            'selected_delta': effective_size_result['selected_delta'],
            'baseline_score': effective_size_result['baseline_score'],
            'selected_score': effective_size_result['selected_score'],
            'rows_before': int(len(train_data.idx)),
            'rows_after': int(len(reduced_data.idx)),
            'elapsed_seconds': elapsed_seconds,
            'budget_seconds': budget_seconds,
            'artifact_mode': config.artifact_mode,
            'protocol_trials': effective_size_result['trials'],
            'provider_meta': final_provider_result.meta,
        }

        return SamplingStageOutput(train_data=reduced_data,
                                   metadata=metadata,
                                   elapsed_seconds=elapsed_seconds,
                                   updated_timeout_minutes=timeout_after_stage)

    @staticmethod
    def _execute_chunking(train_data: InputData,
                          started_at: float,
                          budget_seconds: float,
                          provider: SamplingProvider,
                          config: SamplingConfig,
                          total_timeout_minutes: Optional[float]) -> SamplingStageOutput:
        SamplingStageExecutor._raise_if_budget_exceeded(started_at, budget_seconds)
        remaining_budget_value = SamplingStageExecutor._remaining_budget(started_at, budget_seconds)

        provider_result = provider.sample(
            features=np.asarray(train_data.features),
            target=SamplingStageExecutor._flatten_target(train_data.target),
            strategy=config.strategy,
            strategy_params=config.strategy_params,
            random_state=config.random_state,
            budget_seconds=remaining_budget_value,
            strategy_kind=config.strategy_kind,
            injectable_params=None,
        )

        partitions = provider_result.partitions

        chunked_data = SamplingStageExecutor._partitions_to_input_data_list(partitions, train_data)
        rows_after = sum(len(chunk.idx) for chunk in chunked_data)

        elapsed_seconds = time.perf_counter() - started_at
        timeout_after_stage = SamplingStageExecutor._compute_updated_timeout(
            elapsed_seconds, total_timeout_minutes, config.min_automl_time_minutes
        )

        metadata = {
            'status': 'applied',
            'provider': config.provider,
            'strategy': config.strategy,
            'rows_before': len(train_data.idx),
            'rows_after': rows_after,
            'elapsed_seconds': elapsed_seconds,
            'budget_seconds': budget_seconds,
            'artifact_mode': config.artifact_mode,
            'n_partitions': len(partitions),
            'provider_meta': provider_result.meta,
        }

        return SamplingStageOutput(train_data=chunked_data,
                                   metadata=metadata,
                                   elapsed_seconds=elapsed_seconds,
                                   updated_timeout_minutes=timeout_after_stage)

    def _validate_task_compatibility(self, train_data: InputData) -> None:
        if self.task_type not in (TaskTypesEnum.classification, TaskTypesEnum.regression):
            raise ValueError('Sampling stage supports only classification/regression tasks in V1.')

        if not isinstance(train_data, InputData):
            raise ValueError('Sampling stage supports only InputData in V1.')

        if not data_type_is_table(train_data):
            raise ValueError('Sampling stage supports only tabular InputData in V1.')

        if train_data.target is None:
            raise ValueError('Sampling stage requires non-empty target in train data.')

        if len(train_data.idx) < 5:
            raise ValueError('Sampling stage requires at least 5 rows in train data.')

    @staticmethod
    def _select_effective_ratio(train_data: InputData,
                                started_at: float,
                                budget_seconds: float,
                                provider: SamplingProvider,
                                config: SamplingConfig,
                                task_type: TaskTypesEnum) -> Dict[str, Any]:
        train_split, valid_split = SamplingStageExecutor._split_for_protocol(train_data, config, task_type)

        baseline_score = SamplingStageExecutor._score_light_model(
            train_split, valid_split, task_type, config.random_state
        )
        sorted_ratios = sorted(config.candidate_ratios)
        selected_ratio = None
        selected_delta = None
        selected_score = None
        trials: List[Dict[str, Any]] = []

        for ratio in sorted_ratios:
            SamplingStageExecutor._raise_if_budget_exceeded(started_at, budget_seconds)
            provider_result = provider.sample(
                features=np.asarray(train_split.features),
                target=SamplingStageExecutor._flatten_target(train_split.target),
                strategy=config.strategy,
                strategy_params=config.strategy_params,
                random_state=config.random_state,
                budget_seconds=SamplingStageExecutor._remaining_budget(started_at, budget_seconds),
                strategy_kind=config.strategy_kind,
                injectable_params={'ratio': ratio},
            )
            candidate_indices = SamplingStageExecutor._validate_indices(provider_result.sample_indices,
                                                                        upper_bound=len(train_split.idx),
                                                                        data_label='train split')
            candidate_split = SamplingStageExecutor._subset_by_positions(train_split, candidate_indices)
            candidate_score = SamplingStageExecutor._score_light_model(
                candidate_split, valid_split, task_type, config.random_state
            )
            delta = SamplingStageExecutor._calculate_delta(baseline_score, candidate_score, config.delta_type)

            trials.append({
                'ratio': float(ratio),
                'score': float(candidate_score),
                'delta': float(delta),
                'sample_size': int(len(candidate_indices)),
            })

            if delta <= config.delta_metric_threshold:
                selected_ratio = float(ratio)
                selected_delta = float(delta)
                selected_score = float(candidate_score)
                break

        if selected_ratio is None:
            raise ValueError(
                'No candidate ratio satisfied "delta_metric_threshold". '
                f'Checked ratios: {sorted_ratios}, threshold={config.delta_metric_threshold}.'
            )

        return {
            'selected_ratio': selected_ratio,
            'selected_delta': selected_delta,
            'selected_score': selected_score,
            'baseline_score': float(baseline_score),
            'trials': trials,
        }

    @staticmethod
    def _split_for_protocol(train_data: InputData,
                            config: SamplingConfig,
                            task_type: TaskTypesEnum) -> Sequence[InputData]:
        indices = np.arange(len(train_data.idx))
        target = SamplingStageExecutor._flatten_target(train_data.target)

        stratify_target = target if task_type == TaskTypesEnum.classification else None
        try:
            train_ids, valid_ids = train_test_split(indices,
                                                    test_size=config.validation_size,
                                                    random_state=config.random_state,
                                                    stratify=stratify_target)
        except ValueError:
            train_ids, valid_ids = train_test_split(indices,
                                                    test_size=config.validation_size,
                                                    random_state=config.random_state,
                                                    stratify=None)

        train_split = SamplingStageExecutor._subset_by_positions(train_data, train_ids)
        valid_split = SamplingStageExecutor._subset_by_positions(train_data, valid_ids)
        return train_split, valid_split

    @staticmethod
    def _score_light_model(train_data: InputData,
                           valid_data: InputData,
                           task_type: TaskTypesEnum,
                           random_state: Optional[int]) -> float:
        x_train_df, x_valid_df = SamplingStageExecutor._prepare_feature_matrices(
            train_data.features, valid_data.features
        )
        y_train = SamplingStageExecutor._flatten_target(train_data.target)
        y_valid = SamplingStageExecutor._flatten_target(valid_data.target)

        if task_type == TaskTypesEnum.classification:
            model = RandomForestClassifier(n_estimators=100,
                                           random_state=random_state,
                                           n_jobs=-1)
            model.fit(x_train_df, y_train)
            prediction = model.predict(x_valid_df)
            return float(f1_score(y_valid, prediction, average='macro'))

        model = RandomForestRegressor(n_estimators=100,
                                      random_state=random_state,
                                      n_jobs=1)
        model.fit(x_train_df, y_train)
        prediction = model.predict(x_valid_df)
        return float(r2_score(y_valid, prediction))

    @staticmethod
    def _prepare_feature_matrices(train_features: Any, valid_features: Any) -> Sequence[pd.DataFrame]:
        x_train_df = pd.get_dummies(pd.DataFrame(train_features), dummy_na=True)
        x_valid_df = pd.get_dummies(pd.DataFrame(valid_features), dummy_na=True)
        x_valid_df = x_valid_df.reindex(columns=x_train_df.columns, fill_value=0)
        return x_train_df, x_valid_df

    @staticmethod
    def _compute_budget_seconds(config: SamplingConfig,
                                total_timeout_minutes: Optional[float]) -> float:
        if config.budget_policy != 'dynamic_cap':
            raise ValueError(f'Unsupported budget_policy={config.budget_policy}')

        if total_timeout_minutes is None:
            return float(config.infinite_timeout_cap_minutes * 60)

        total_seconds = float(total_timeout_minutes * 60)
        max_share_seconds = total_seconds * config.cap_max_timeout_share
        guaranteed_remaining_seconds = config.min_automl_time_minutes * 60
        max_by_remaining = max(0.0, total_seconds - guaranteed_remaining_seconds)
        budget_seconds = min(max_share_seconds, max_by_remaining)

        if budget_seconds <= 0:
            raise ValueError(
                'Sampling stage has zero budget due to timeout constraints. '
                f'Increase timeout or reduce min_automl_time_minutes ({config.min_automl_time_minutes}).'
            )

        return float(budget_seconds)

    @staticmethod
    def _compute_updated_timeout(elapsed_seconds: float,
                                 total_timeout_minutes: Optional[float],
                                 min_automl_time_minutes: float) -> Optional[float]:
        if total_timeout_minutes is None:
            return None

        remaining = float(total_timeout_minutes) - elapsed_seconds / 60.0
        return float(max(min_automl_time_minutes, remaining))

    def _create_provider(self, provider_name: str) -> SamplingProvider:
        if provider_name == 'sampling_zoo':
            return SamplingZooProvider()
        raise ValueError(f'Unknown sampling provider: {provider_name}')

    @staticmethod
    def _flatten_target(target: Any) -> np.ndarray:
        values = np.asarray(target)
        if values.ndim > 1 and values.shape[1] == 1:
            values = values.reshape(-1)
        return values

    @staticmethod
    def _calculate_delta(baseline_score: float, sampled_score: float, delta_type: str) -> float:
        score_drop = max(0.0, baseline_score - sampled_score)
        if delta_type == 'absolute':
            return float(score_drop)

        denominator = max(abs(baseline_score), 1e-12)
        return float(score_drop / denominator)

    @staticmethod
    def _validate_indices(indices: np.ndarray, upper_bound: int, data_label: str) -> np.ndarray:
        values = np.asarray(indices)
        if values.ndim != 1:
            raise ValueError(f'Sampled indices for {data_label} must be a 1D array.')
        if len(values) == 0:
            raise ValueError(f'Sampled indices for {data_label} must not be empty.')

        try:
            values = values.astype(int)
        except Exception as ex:
            raise ValueError(f'Sampled indices for {data_label} must be integer-like. Details: {ex}')

        if len(np.unique(values)) != len(values):
            raise ValueError(f'Sampled indices for {data_label} must be unique.')

        if values.min() < 0 or values.max() >= upper_bound:
            raise ValueError(
                f'Sampled indices for {data_label} are out of bounds. '
                f'Allowed range: [0, {upper_bound - 1}].'
            )

        return values

    @staticmethod
    def _subset_by_positions(data: InputData, positions: np.ndarray) -> InputData:
        positions = np.asarray(positions, dtype=int)
        features = np.take(data.features, positions, axis=0)
        target = np.take(data.target, positions, axis=0)
        idx = np.take(data.idx, positions, axis=0)

        categorical_features = None
        if data.categorical_features is not None:
            categorical_features = np.take(data.categorical_features, positions, axis=0)

        return InputData(
            idx=idx,
            features=features,
            target=target,
            task=deepcopy(data.task),
            data_type=data.data_type,
            supplementary_data=data.supplementary_data,
            categorical_features=categorical_features,
            categorical_idx=data.categorical_idx,
            numerical_idx=data.numerical_idx,
            encoded_idx=data.encoded_idx,
            features_names=data.features_names,
        )

    @staticmethod
    def _take_target_slice(target: Any, indices: Any) -> Any:
        if target is None:
            return None
        if isinstance(target, (pd.Series, pd.DataFrame)):
            try:
                return target.loc[indices].to_numpy()
            except Exception:
                return target.iloc[indices].to_numpy()
        return np.asarray(target)[indices]

    @staticmethod
    def _partitions_to_input_data_list(partitions: Dict[str, Any],
                                       original_input_data: InputData) -> InputDataList:
        input_data_list: InputDataList = []

        for partition_name, partition_data in partitions.items():
            if isinstance(partition_data, dict):
                X_partition = partition_data['feature']
                y_partition = partition_data['target']

                if isinstance(X_partition, pd.DataFrame):
                    indices = X_partition.index.values
                    X_values = X_partition.to_numpy()
                else:
                    indices = np.arange(len(X_partition))
                    X_values = np.asarray(X_partition)

                if isinstance(y_partition, (pd.Series, pd.DataFrame)):
                    y_partition = y_partition.to_numpy()
            else:
                indices = partition_data
                if isinstance(original_input_data.features, pd.DataFrame):
                    X_values = original_input_data.features.loc[indices].to_numpy()
                else:
                    X_values = np.asarray(original_input_data.features)[indices]
                if isinstance(original_input_data.target, (pd.Series, pd.DataFrame)):
                    y_partition = original_input_data.target.to_numpy()[indices]
                else:
                    y_partition = original_input_data.target[indices]

            categorical_features = None
            if original_input_data.categorical_idx is not None and len(original_input_data.categorical_idx) > 0:
                categorical_features = X_values[:, original_input_data.categorical_idx]

            partition_input_data = InputData(
                idx=np.asarray(indices),
                features=X_values,
                target=y_partition,
                task=deepcopy(original_input_data.task),
                data_type=original_input_data.data_type,
                supplementary_data=original_input_data.supplementary_data,
                categorical_features=categorical_features,
                categorical_idx=original_input_data.categorical_idx,
                numerical_idx=original_input_data.numerical_idx,
                encoded_idx=original_input_data.encoded_idx,
                features_names=original_input_data.features_names,
            )

            input_data_list.append(partition_input_data)

        return input_data_list

    @staticmethod
    def _raise_if_budget_exceeded(started_at: float, budget_seconds: float) -> None:
        elapsed = time.perf_counter() - started_at
        if elapsed > budget_seconds:
            raise TimeoutError(
                f'Sampling stage exceeded its dynamic cap: elapsed={elapsed:.2f}s, budget={budget_seconds:.2f}s.'
            )

    @staticmethod
    def _remaining_budget(started_at: float, budget_seconds: float) -> float:
        elapsed = time.perf_counter() - started_at
        return max(0.0, budget_seconds - elapsed)
