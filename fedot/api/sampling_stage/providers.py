import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Literal, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class SamplingSubsetResult:
    sample_indices: np.ndarray
    sample_scores: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingChunkingResult:
    partitions: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


SamplingProviderResult = Union[SamplingSubsetResult, SamplingChunkingResult]


class SamplingProvider(ABC):
    @abstractmethod
    def sample(self,
               features: np.ndarray,
               target: np.ndarray,
               strategy: str,
               strategy_params: Dict[str, Any],
               random_state: Optional[int],
               budget_seconds: Optional[float],
               strategy_kind: Optional[Literal['subset', 'chunking']] = None,
               injectable_params: Optional[Dict[str, Any]] = None) -> SamplingProviderResult:
        pass


class SamplingZooProvider(SamplingProvider):
    _SAMPLING_MODULE_CANDIDATES = (
        'sampling_zoo.core.api.api_main',
        'sampling_zoo.api.api_main',
        'core.api.api_main',
    )

    def __init__(self):
        self._factory_cls = self._load_factory()

    def sample(self,
               features: np.ndarray,
               target: np.ndarray,
               strategy: str,
               strategy_params: Dict[str, Any],
               random_state: Optional[int],
               budget_seconds: Optional[float],
               strategy_kind: Optional[Literal['subset', 'chunking']] = None,
               injectable_params: Optional[Dict[str, Any]] = None) -> SamplingProviderResult:
        del budget_seconds
        factory = self._factory_cls()
        if strategy_kind is None:
            strategy_kind = self._resolve_strategy_kind(factory, strategy)

        if strategy_kind == 'chunking':
            return self._sample_chunking(factory=factory,
                                         features=features,
                                         target=target,
                                         strategy=strategy,
                                         strategy_params=strategy_params,
                                         random_state=random_state)
        elif strategy_kind == 'subset':
            return self._sample_subset(factory=factory,
                                       features=features,
                                       target=target,
                                       strategy=strategy,
                                       strategy_params=strategy_params,
                                       random_state=random_state,
                                       injectable_params=injectable_params)
        else:
            raise ValueError(f'Unsupported sampling strategy kind: {strategy_kind}')

    def _sample_subset(self,
                       factory: Any,
                       features: np.ndarray,
                       target: np.ndarray,
                       strategy: str,
                       strategy_params: Dict[str, Any],
                       random_state: Optional[int],
                       injectable_params: Optional[Dict[str, Any]]) -> SamplingProviderResult:
        n_rows = int(features.shape[0])

        strategy_kwargs = dict(strategy_params)
        if random_state is not None and 'random_state' not in strategy_kwargs:
            strategy_kwargs['random_state'] = random_state

        strategy_kwargs = self._inject_required_kwargs(
            factory=factory,
            strategy_name=strategy,
            strategy_kwargs=strategy_kwargs,
            n_rows=n_rows,
            injectable_params=injectable_params,
        )

        sample_size = strategy_kwargs.get('sample_size') or n_rows
        strategy_obj = self._create_strategy(factory, strategy, strategy_kwargs)

        data_frame = pd.DataFrame(features)
        self._fit_strategy(strategy_obj, data_frame, target)

        extracted = self._extract_indices(strategy_obj, data_frame, target)
        if extracted.size < sample_size:
            raise ValueError(
                f'Sampling provider returned too few unique indices: {extracted.size}, required at least {sample_size}.'
            )

        rng = np.random.default_rng(random_state)
        sampled = rng.choice(extracted, size=sample_size, replace=False)
        sampled = np.asarray(sampled, dtype=int)

        sample_scores = self._extract_scores(strategy_obj, sampled)
        meta = {
            'provider': 'sampling_zoo',
            'strategy': strategy,
            'strategy_kind': 'subset',
            'sample_size': sample_size,
            'strategy_kwargs': strategy_kwargs,
        }

        return SamplingSubsetResult(sample_indices=sampled,
                                    sample_scores=sample_scores,
                                    meta=meta)

    def _sample_chunking(self,
                         factory: Any,
                         features: np.ndarray,
                         target: np.ndarray,
                         strategy: str,
                         strategy_params: Dict[str, Any],
                         random_state: Optional[int]) -> SamplingProviderResult:
        strategy_kwargs = dict(strategy_params)
        if random_state is not None and 'random_state' not in strategy_kwargs:
            strategy_kwargs['random_state'] = random_state

        data_frame = pd.DataFrame(features)
        partitions = self._fit_transform_partitions(factory=factory,
                                                    strategy=strategy,
                                                    data_frame=data_frame,
                                                    target=target,
                                                    strategy_kwargs=strategy_kwargs)
        if not isinstance(partitions, dict) or len(partitions) == 0:
            raise ValueError('Chunking strategy did not return any partitions.')

        meta = {
            'provider': 'sampling_zoo',
            'strategy': strategy,
            'strategy_kind': 'chunking',
            'strategy_kwargs': strategy_kwargs,
            'n_partitions': len(partitions),
        }

        return SamplingChunkingResult(partitions=partitions,
                                      meta=meta)

    @staticmethod
    def _create_strategy(factory: Any, strategy_name: str, strategy_kwargs: Dict[str, Any]) -> Any:
        try:
            return factory.create_strategy(strategy_name, **strategy_kwargs)
        except TypeError as ex:
            raise ValueError(
                f'Failed to initialize sampling strategy "{strategy_name}" with parameters {strategy_kwargs}: {ex}'
            )

    @staticmethod
    def _fit_strategy(strategy_obj: Any, data_frame: pd.DataFrame, target: np.ndarray) -> None:
        fit_method = getattr(strategy_obj, 'fit', None)
        if fit_method is None:
            raise ValueError('Sampling strategy object has no "fit" method.')

        calls = (
            lambda: fit_method(data_frame, target=target),
            lambda: fit_method(data_frame, target),
            lambda: fit_method(data_frame),
        )
        last_error = None
        for call in calls:
            try:
                call()
                return
            except TypeError as ex:
                last_error = ex

        raise ValueError(f'Unable to call strategy.fit(...) due to incompatible signature: {last_error}')

    @staticmethod
    def _extract_scores(strategy_obj: Any, selected_indices: np.ndarray) -> Optional[np.ndarray]:
        for attr_name in ('sampling_scores_', 'difficulty_scores_', 'uncertainty_scores_'):
            score_values = getattr(strategy_obj, attr_name, None)
            if score_values is None:
                continue
            try:
                score_values = np.asarray(score_values)
                if score_values.ndim != 1:
                    continue
                return score_values[selected_indices]
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_indices(strategy_obj: Any,
                         data_frame: pd.DataFrame,
                         target: np.ndarray) -> np.ndarray:
        indices = SamplingZooProvider._extract_indices_from_sample_method(strategy_obj)
        if indices is None:
            indices = SamplingZooProvider._extract_indices_from_attrs(strategy_obj)
        if indices is None:
            indices = SamplingZooProvider._extract_indices_from_get_partitions(strategy_obj, data_frame, target)

        if indices is None or len(indices) == 0:
            raise ValueError('Sampling strategy did not return any indices.')

        indices = np.asarray(indices, dtype=int)
        unique_indices = np.unique(indices)
        return unique_indices

    @staticmethod
    def _extract_indices_from_sample_method(strategy_obj: Any) -> Optional[np.ndarray]:
        sample_indices_method = getattr(strategy_obj, 'sample_indices', None)
        if sample_indices_method is None:
            return None

        call_attempts = (
            lambda: sample_indices_method(),
            lambda: sample_indices_method(replace=False),
        )
        for attempt in call_attempts:
            try:
                result = attempt()
                if isinstance(result, tuple):
                    result = result[0]
                arr = np.asarray(result)
                if arr.ndim == 1:
                    return arr.astype(int)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_indices_from_attrs(strategy_obj: Any) -> Optional[np.ndarray]:
        for attr_name in ('sampled_indices', 'sampled_indices_'):
            value = getattr(strategy_obj, attr_name, None)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 1:
                return arr.astype(int)

        for attr_name in ('partitions', 'partitions_'):
            partitions = getattr(strategy_obj, attr_name, None)
            if not isinstance(partitions, dict):
                continue
            values = []
            for part_value in partitions.values():
                parsed = SamplingZooProvider._parse_partition_value(part_value)
                if parsed is not None:
                    values.append(parsed)
            if values:
                return np.concatenate(values)
        return None

    @staticmethod
    def _extract_indices_from_get_partitions(strategy_obj: Any,
                                             data_frame: pd.DataFrame,
                                             target: np.ndarray) -> Optional[np.ndarray]:
        get_partitions = getattr(strategy_obj, 'get_partitions', None)
        if get_partitions is None:
            return None

        calls = (
            lambda: get_partitions(data_frame, target),
            lambda: get_partitions(data_frame),
            lambda: get_partitions(),
        )
        partitions = None
        for call in calls:
            try:
                partitions = call()
                break
            except TypeError:
                continue
            except Exception:
                return None

        if not isinstance(partitions, dict):
            return None

        values = []
        for part_value in partitions.values():
            parsed = SamplingZooProvider._parse_partition_value(part_value)
            if parsed is not None:
                values.append(parsed)

        if not values:
            return None

        return np.concatenate(values)

    @staticmethod
    def _extract_partitions(strategy_obj: Any,
                            data_frame: pd.DataFrame,
                            target: np.ndarray) -> Optional[Dict[str, Any]]:
        get_partitions = getattr(strategy_obj, 'get_partitions', None)
        if get_partitions is not None:
            calls = (
                lambda: get_partitions(data_frame, target),
                lambda: get_partitions(data_frame),
                lambda: get_partitions(),
            )
            for call in calls:
                try:
                    partitions = call()
                    if isinstance(partitions, dict):
                        return partitions
                except TypeError:
                    continue
                except Exception:
                    return None

        for attr_name in ('partitions', 'partitions_'):
            partitions = getattr(strategy_obj, attr_name, None)
            if isinstance(partitions, dict):
                return partitions
        return None

    def _fit_transform_partitions(self,
                                  factory: Any,
                                  strategy: str,
                                  data_frame: pd.DataFrame,
                                  target: np.ndarray,
                                  strategy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        fit_transform = getattr(factory, 'fit_transform', None)
        if fit_transform is not None:
            calls = (
                lambda: fit_transform(strategy_type=strategy,
                                      data=data_frame,
                                      target=target,
                                      strategy_kwargs=strategy_kwargs,
                                      fit_kwargs={}),
                lambda: fit_transform(strategy, data_frame, target, strategy_kwargs, {}),
                lambda: fit_transform(strategy, data_frame, target, strategy_kwargs),
                lambda: fit_transform(strategy, data_frame, target),
                lambda: fit_transform(strategy, data_frame),
            )
            for call in calls:
                try:
                    partitions = call()
                    if isinstance(partitions, dict):
                        return partitions
                except TypeError:
                    continue
                except Exception:
                    break

        strategy_obj = self._create_strategy(factory, strategy, strategy_kwargs)
        self._fit_strategy(strategy_obj, data_frame, target)
        partitions = self._extract_partitions(strategy_obj, data_frame, target)
        if partitions is None or not isinstance(partitions, dict):
            raise ValueError('Chunking strategy did not return partitions.')
        return partitions

    @staticmethod
    def _resolve_strategy_kind(factory: Any,
                               strategy: str) -> Literal['subset', 'chunking']:
        is_chunking = getattr(factory, 'is_chunking_strategy', None)
        if callable(is_chunking):
            try:
                if is_chunking(strategy):
                    return 'chunking'
            except Exception:
                pass

        is_subset = getattr(factory, 'is_subset_strategy', None)
        if callable(is_subset):
            try:
                if is_subset(strategy):
                    return 'subset'
            except Exception:
                pass

        chunking_list = getattr(factory, 'get_chunking_strategies', None)
        if callable(chunking_list):
            try:
                if strategy in chunking_list():
                    return 'chunking'
            except Exception:
                pass

        subset_list = getattr(factory, 'get_subset_strategies', None)
        if callable(subset_list):
            try:
                if strategy in subset_list():
                    return 'subset'
            except Exception:
                pass

        return 'subset'

    @staticmethod
    def _parse_partition_value(part_value: Any) -> Optional[np.ndarray]:
        if isinstance(part_value, np.ndarray) and part_value.ndim == 1 and np.issubdtype(part_value.dtype, np.number):
            return part_value.astype(int)
        if isinstance(part_value, (list, tuple)):
            arr = np.asarray(part_value)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                return arr.astype(int)
        if isinstance(part_value, dict):
            idx = part_value.get('indices')
            if idx is not None:
                arr = np.asarray(idx)
                if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                    return arr.astype(int)

            for key in ('feature', 'target'):
                part_data = part_value.get(key)
                if isinstance(part_data, (pd.DataFrame, pd.Series)):
                    index_values = np.asarray(part_data.index)
                    if index_values.ndim == 1 and np.issubdtype(index_values.dtype, np.number):
                        return index_values.astype(int)
        return None

    @staticmethod
    def _inject_required_kwargs(factory: Any,
                               strategy_name: str,
                               strategy_kwargs: Dict[str, Any],
                               n_rows: int,
                               injectable_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        updated_kwargs = dict(strategy_kwargs)
        if injectable_params is None:
            return updated_kwargs

        strategy_map = getattr(factory, 'strategy_map', None)
        strategy_cls = strategy_map.get(strategy_name) if isinstance(strategy_map, dict) else None
        if strategy_cls is None:
            return updated_kwargs

        try:
            signature = inspect.signature(strategy_cls)
        except (TypeError, ValueError):
            return updated_kwargs

        if 'sample_size' in signature.parameters \
            and 'sample_size' not in updated_kwargs \
            and injectable_params.get('ratio'):
            updated_kwargs['sample_size'] = max(1, round(injectable_params.get('ratio') * n_rows))

        return updated_kwargs

    def _load_factory(self):
        for module_name in self._SAMPLING_MODULE_CANDIDATES:
            try:
                module = import_module(module_name)
                factory_cls = getattr(module, 'SamplingStrategyFactory', None)
                if factory_cls is not None:
                    return factory_cls
            except ModuleNotFoundError:
                continue

        raise ModuleNotFoundError(
            'SamplingZoo provider is unavailable. Install optional dependencies for Sampling Zoo '
            '(for example: pip install "fedot[sampling_zoo]").'
        )
