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
        self._factory_cls = self.load_factory()

    def sample(self,
               features: np.ndarray,
               target: np.ndarray,
               strategy: str,
               strategy_params: Dict[str, Any],
               random_state: Optional[int],
               budget_seconds: Optional[float],
               strategy_kind: Optional[Literal['subset', 'chunking']] = None,
               injectable_params: Optional[Dict[str, Any]] = None) -> SamplingProviderResult:
        factory = self._factory_cls()
        if strategy_kind is None:
            strategy_kind = self._resolve_strategy_kind(factory, strategy)

        if strategy_kind == 'chunking':
            return self._sample_chunking(
                factory=factory,
                features=features,
                target=target,
                strategy=strategy,
                strategy_params=strategy_params,
                random_state=random_state
            )
        elif strategy_kind == 'subset':
            return self._sample_subset(
                factory=factory,
                features=features,
                target=target,
                strategy=strategy,
                strategy_params=strategy_params,
                random_state=random_state,
                injectable_params=injectable_params
            )
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
        strategy_obj, indices = self._apply_strategy(
            factory=factory,
            strategy=strategy,
            data_frame=pd.DataFrame(features),
            target=target,
            strategy_kwargs=strategy_kwargs
        )
        if indices is None or len(indices) == 0:
            raise ValueError('Sampling strategy did not return any indices.')

        indices = np.unique(np.asarray(indices, dtype=int))
        if indices.size < sample_size:
            raise ValueError(
                f'Sampling provider returned too few unique indices: {indices.size}, required at least {sample_size}.'
            )

        rng = np.random.default_rng(random_state)
        sampled = rng.choice(indices, size=sample_size, replace=False)
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
        strategy_obj, partitions = self._apply_strategy(
            factory=factory,
            strategy=strategy,
            data_frame=data_frame,
            target=target,
            strategy_kwargs=strategy_kwargs
        )
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
    def _apply_strategy(factory: Any,
                        strategy: str,
                        data_frame: pd.DataFrame,
                        target: np.ndarray,
                        strategy_kwargs: Dict[str, Any]) -> tuple:
        fit_transform = getattr(factory, 'fit_transform', None)
        if fit_transform is None:
            raise ValueError('Sampling strategy object has no "fit_transform" method.')
        try:
            strategy, result = fit_transform(strategy_type=strategy,
                                             data=data_frame,
                                             target=target,
                                             strategy_kwargs=strategy_kwargs,
                                             fit_kwargs={},
                                             return_strategy=True)

            return strategy, result
        except Exception as e:
            raise ValueError("Error during sampling strategy apply") from e

    @staticmethod
    def _resolve_strategy_kind(factory: Any,
                               strategy: str) -> Literal['subset', 'chunking']:
        is_chunking = getattr(factory, 'is_chunking_strategy', None)
        if callable(is_chunking) and is_chunking(strategy):
            return 'chunking'

        is_subset = getattr(factory, 'is_subset_strategy', None)
        if callable(is_subset) and is_subset(strategy):
            return 'subset'

        return 'subset'

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

    def load_factory(self):
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
