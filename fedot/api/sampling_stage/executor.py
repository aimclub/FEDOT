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
        self.provider = provider

    def execute(self, train_data: InputData) -> SamplingStageOutput:
        self._validate_task_compatibility(train_data)

        started_at = time.perf_counter()
        budget_seconds = self._compute_budget_seconds()

        provider = self.provider or self._create_provider(self.config.provider)
        if self.config.strategy_kind == 'chunking':
            return self._execute_chunking(train_data, provider, started_at, budget_seconds)
        elif self.config.strategy_kind == 'subset':
            return self.execute_subset(train_data, provider, started_at, budget_seconds)
        else:
            raise ValueError(f'Unknown strategy_kind: {self.config.strategy_kind}')

    def execute_subset(self,
                       train_data: InputData,
                       provider: SamplingProvider,
                       started_at: float,
                       budget_seconds: float) -> SamplingStageOutput:
        effective_size_result = self._select_effective_ratio(train_data, provider, started_at, budget_seconds)

        self._raise_if_budget_exceeded(started_at, budget_seconds)
        remaining_budget = self._remaining_budget(started_at, budget_seconds)
        final_provider_result = provider.sample(
            features=np.asarray(train_data.features),
            target=self._flatten_target(train_data.target),
            strategy=self.config.strategy,
            strategy_params=self.config.strategy_params,
            random_state=self.config.random_state,
            budget_seconds=remaining_budget,
            strategy_kind=self.config.strategy_kind,
            injectable_params={'ratio': effective_size_result['selected_ratio']},
        )

        selected_indices = self._validate_indices(final_provider_result.sample_indices,
                                                  upper_bound=len(train_data.idx),
                                                  data_label='full train data')
        reduced_data = self._subset_by_positions(train_data, selected_indices)

        elapsed_seconds = time.perf_counter() - started_at
        timeout_after_stage = self._compute_updated_timeout(elapsed_seconds)

        metadata = {
            'status': 'applied',
            'provider': self.config.provider,
            'strategy': self.config.strategy,
            'selected_ratio': effective_size_result['selected_ratio'],
            'selected_delta': effective_size_result['selected_delta'],
            'baseline_score': effective_size_result['baseline_score'],
            'selected_score': effective_size_result['selected_score'],
            'rows_before': int(len(train_data.idx)),
            'rows_after': int(len(reduced_data.idx)),
            'elapsed_seconds': elapsed_seconds,
            'budget_seconds': budget_seconds,
            'artifact_mode': self.config.artifact_mode,
            'protocol_trials': effective_size_result['trials'],
            'provider_meta': final_provider_result.meta,
        }

        return SamplingStageOutput(train_data=reduced_data,
                                   metadata=metadata,
                                   elapsed_seconds=elapsed_seconds,
                                   updated_timeout_minutes=timeout_after_stage)

    def _execute_chunking(self,
                          train_data: InputData,
                          provider: SamplingProvider,
                          started_at: float,
                          budget_seconds: float) -> SamplingStageOutput:
        self._raise_if_budget_exceeded(started_at, budget_seconds)
        remaining_budget = self._remaining_budget(started_at, budget_seconds)

        provider_result = provider.sample(
            features=np.asarray(train_data.features),
            target=self._flatten_target(train_data.target),
            strategy=self.config.strategy,
            strategy_params=self.config.strategy_params,
            random_state=self.config.random_state,
            budget_seconds=remaining_budget,
            strategy_kind=self.config.strategy_kind,
            injectable_params=None,
        )

        partitions = provider_result.partitions
        if not isinstance(partitions, dict) or len(partitions) == 0:
            raise ValueError('Chunking strategy did not return partitions.')

        chunked_data = self._partitions_to_input_data_list(partitions, train_data)
        rows_after = sum(len(chunk.idx) for chunk in chunked_data)

        elapsed_seconds = time.perf_counter() - started_at
        timeout_after_stage = self._compute_updated_timeout(elapsed_seconds)

        metadata = {
            'status': 'chunking',
            'provider': self.config.provider,
            'strategy': self.config.strategy,
            'rows_before': int(len(train_data.idx)),
            'rows_after': int(rows_after),
            'elapsed_seconds': elapsed_seconds,
            'budget_seconds': budget_seconds,
            'artifact_mode': self.config.artifact_mode,
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

    def _select_effective_ratio(self,
                                train_data: InputData,
                                provider: SamplingProvider,
                                started_at: float,
                                budget_seconds: float) -> Dict[str, Any]:
        train_split, valid_split = self._split_for_protocol(train_data)

        baseline_score = self._score_light_model(train_split, valid_split)
        sorted_ratios = sorted(self.config.candidate_ratios)
        selected_ratio = None
        selected_delta = None
        selected_score = None
        trials: List[Dict[str, Any]] = []

        for ratio in sorted_ratios:
            self._raise_if_budget_exceeded(started_at, budget_seconds)
            provider_result = provider.sample(
                features=np.asarray(train_split.features),
                target=self._flatten_target(train_split.target),
                strategy=self.config.strategy,
                strategy_params=self.config.strategy_params,
                random_state=self.config.random_state,
                budget_seconds=self._remaining_budget(started_at, budget_seconds),
                strategy_kind=self.config.strategy_kind,
                injectable_params={'ratio': ratio},
            )
            candidate_indices = self._validate_indices(provider_result.sample_indices,
                                                       upper_bound=len(train_split.idx),
                                                       data_label='train split')
            candidate_split = self._subset_by_positions(train_split, candidate_indices)
            candidate_score = self._score_light_model(candidate_split, valid_split)
            delta = self._calculate_delta(baseline_score, candidate_score)

            trials.append({
                'ratio': float(ratio),
                'score': float(candidate_score),
                'delta': float(delta),
                'sample_size': int(len(candidate_indices)),
            })

            if delta <= self.config.delta_metric_threshold:
                selected_ratio = float(ratio)
                selected_delta = float(delta)
                selected_score = float(candidate_score)
                break

        if selected_ratio is None:
            raise ValueError(
                'No candidate ratio satisfied "delta_metric_threshold". '
                f'Checked ratios: {sorted_ratios}, threshold={self.config.delta_metric_threshold}.'
            )

        return {
            'selected_ratio': selected_ratio,
            'selected_delta': selected_delta,
            'selected_score': selected_score,
            'baseline_score': float(baseline_score),
            'trials': trials,
        }

    def _split_for_protocol(self, train_data: InputData) -> Sequence[InputData]:
        indices = np.arange(len(train_data.idx))
        target = self._flatten_target(train_data.target)

        stratify_target = target if self.task_type == TaskTypesEnum.classification else None
        try:
            train_ids, valid_ids = train_test_split(indices,
                                                    test_size=self.config.validation_size,
                                                    random_state=self.config.random_state,
                                                    stratify=stratify_target)
        except ValueError:
            train_ids, valid_ids = train_test_split(indices,
                                                    test_size=self.config.validation_size,
                                                    random_state=self.config.random_state,
                                                    stratify=None)

        train_split = self._subset_by_positions(train_data, train_ids)
        valid_split = self._subset_by_positions(train_data, valid_ids)
        return train_split, valid_split

    def _score_light_model(self, train_data: InputData, valid_data: InputData) -> float:
        x_train_df, x_valid_df = self._prepare_feature_matrices(train_data.features, valid_data.features)
        y_train = self._flatten_target(train_data.target)
        y_valid = self._flatten_target(valid_data.target)

        if self.task_type == TaskTypesEnum.classification:
            model = RandomForestClassifier(n_estimators=100,
                                           random_state=self.config.random_state,
                                           n_jobs=-1)
            model.fit(x_train_df, y_train)
            prediction = model.predict(x_valid_df)
            return float(f1_score(y_valid, prediction, average='macro'))

        model = RandomForestRegressor(n_estimators=100,
                                      random_state=self.config.random_state,
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

    def _compute_budget_seconds(self) -> float:
        if self.config.budget_policy != 'dynamic_cap':
            raise ValueError(f'Unsupported budget_policy={self.config.budget_policy}')

        if self.total_timeout_minutes is None:
            return float(self.config.infinite_timeout_cap_minutes * 60)

        total_seconds = float(self.total_timeout_minutes * 60)
        max_share_seconds = total_seconds * self.config.cap_max_timeout_share
        guaranteed_remaining_seconds = self.config.min_automl_time_minutes * 60
        max_by_remaining = max(0.0, total_seconds - guaranteed_remaining_seconds)
        budget_seconds = min(max_share_seconds, max_by_remaining)

        if budget_seconds <= 0:
            raise ValueError(
                'Sampling stage has zero budget due to timeout constraints. '
                f'Increase timeout or reduce min_automl_time_minutes ({self.config.min_automl_time_minutes}).'
            )

        return float(budget_seconds)

    def _compute_updated_timeout(self, elapsed_seconds: float) -> Optional[float]:
        if self.total_timeout_minutes is None:
            return None

        remaining = float(self.total_timeout_minutes) - elapsed_seconds / 60.0
        return float(max(self.config.min_automl_time_minutes, remaining))

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

    def _calculate_delta(self, baseline_score: float, sampled_score: float) -> float:
        score_drop = max(0.0, baseline_score - sampled_score)
        if self.config.delta_type == 'absolute':
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
    def _take_feature_slice(features: Any, indices: Any) -> Any:
        if isinstance(features, pd.DataFrame):
            try:
                return features.loc[indices]
            except Exception:
                return features.iloc[indices]
        return features[indices]

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
            del partition_name
            if isinstance(partition_data, dict):
                X_partition = partition_data['feature']
                y_partition = partition_data['target']

                if isinstance(X_partition, pd.DataFrame):
                    indices = X_partition.index.values
                else:
                    indices = np.arange(len(X_partition))

                if isinstance(X_partition, pd.DataFrame):
                    X_values = X_partition.values
                else:
                    X_values = X_partition
            else:
                indices = partition_data
                X_values = SamplingStageExecutor._take_feature_slice(original_input_data.features, indices)
                y_partition = SamplingStageExecutor._take_target_slice(original_input_data.target, indices)

            if isinstance(indices, list):
                indices = np.asarray(indices)

            categorical_features = None
            if original_input_data.categorical_features is not None and indices is not None:
                try:
                    categorical_features = SamplingStageExecutor._take_feature_slice(
                        original_input_data.categorical_features, indices
                    )
                except Exception:
                    categorical_features = None

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
