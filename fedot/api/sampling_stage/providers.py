import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union
import numpy as np
import pandas as pd

FeatureMatrix = Union[np.ndarray, pd.DataFrame]


@dataclass
class SamplingSubsetResult:
    sample_indices: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingChunkingResult:
    partitions: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    partition_predictor: Optional[Any] = None


SamplingProviderResult = Union[SamplingSubsetResult, SamplingChunkingResult]


class SamplingProvider(ABC):
    @abstractmethod
    def sample(self,
               features: FeatureMatrix,
               target: np.ndarray,
               strategy: str,
               strategy_params: Dict[str, Any],
               strategy_kind: Optional[Literal['subset', 'chunking']] = None,
               injectable_params: Optional[Dict[str, Any]] = None) -> SamplingProviderResult:
        pass


class SamplingZooProvider(SamplingProvider):
    def __init__(self):
        self._factory_cls = self.load_factory()

    def sample(self,
               features: FeatureMatrix,
               target: np.ndarray,
               strategy: str,
               strategy_params: Dict[str, Any],
               strategy_kind: Optional[Literal['subset', 'chunking']] = None,
               injectable_params: Optional[Dict[str, Any]] = None) -> SamplingProviderResult:
        factory = self._factory_cls()
        if strategy_kind is None:
            strategy_kind = self._resolve_strategy_kind(factory, strategy)

        if strategy_kind == 'chunking':
            return SamplingZooProvider._sample_chunking(
                factory=factory,
                features=features,
                target=target,
                strategy=strategy,
                strategy_params=strategy_params,
            )
        if strategy_kind == 'subset':
            return SamplingZooProvider._sample_subset(
                factory=factory,
                features=features,
                target=target,
                strategy=strategy,
                strategy_params=strategy_params,
                injectable_params=injectable_params,
            )
        raise ValueError(f'Unsupported sampling strategy kind: {strategy_kind}')

    @staticmethod
    def _sample_subset(factory: Any,
                       features: FeatureMatrix,
                       target: np.ndarray,
                       strategy: str,
                       strategy_params: Dict[str, Any],
                       injectable_params: Optional[Dict[str, Any]]) -> SamplingProviderResult:
        n_rows = int(features.shape[0])
        strategy_kwargs = dict(strategy_params)
        strategy_kwargs = SamplingZooProvider._inject_required_kwargs(
            factory=factory,
            strategy_name=strategy,
            strategy_kwargs=strategy_kwargs,
            n_rows=n_rows,
            injectable_params=injectable_params,
        )

        sample_size = strategy_kwargs.get('sample_size') or n_rows
        _, indices = SamplingZooProvider._apply_strategy(
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

        rng = np.random.default_rng(strategy_kwargs.get('random_state'))
        sampled = rng.choice(indices, size=sample_size, replace=False)
        sampled = np.asarray(sampled, dtype=int)

        meta = {
            'provider': 'sampling_zoo',
            'strategy': strategy,
            'strategy_kind': 'subset',
            'sample_size': sample_size,
            'strategy_kwargs': strategy_kwargs,
        }

        return SamplingSubsetResult(sample_indices=sampled,
                                    meta=meta)

    @staticmethod
    def _sample_chunking(factory: Any,
                         features: FeatureMatrix,
                         target: np.ndarray,
                         strategy: str,
                         strategy_params: Dict[str, Any]) -> SamplingProviderResult:
        strategy_kwargs = dict(strategy_params)

        data_frame = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        strategy_obj, partitions = SamplingZooProvider._apply_strategy(
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
            'budget_policy': getattr(strategy_obj, 'budget_policy_', {'applied': False}),
        }

        return SamplingChunkingResult(partitions=partitions,
                                      meta=meta,
                                      partition_predictor=strategy_obj)

    @staticmethod
    def _apply_strategy(factory: Any,
                        strategy: str,
                        data_frame: pd.DataFrame,
                        target: np.ndarray,
                        strategy_kwargs: Dict[str, Any]) -> tuple:
        return factory.fit_transform(strategy_type=strategy,
                                     data=data_frame,
                                     target=target,
                                     strategy_kwargs=strategy_kwargs,
                                     fit_kwargs={},
                                     return_strategy=True)

    @staticmethod
    def _resolve_strategy_kind(factory: Any,
                               strategy: str) -> Literal['subset', 'chunking']:
        if factory.is_chunking_strategy(strategy):
            return 'chunking'
        if factory.is_subset_strategy(strategy):
            return 'subset'
        raise ValueError(f'Unknown sampling strategy: {strategy}')

    @staticmethod
    def _inject_required_kwargs(factory: Any,
                               strategy_name: str,
                               strategy_kwargs: Dict[str, Any],
                               n_rows: int,
                               injectable_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        updated_kwargs = dict(strategy_kwargs)
        if injectable_params is None:
            return updated_kwargs

        signature = inspect.signature(factory.strategy_map[strategy_name])

        if 'sample_size' in signature.parameters \
            and 'sample_size' not in updated_kwargs \
            and injectable_params.get('ratio'):
            updated_kwargs['sample_size'] = max(1, round(injectable_params.get('ratio') * n_rows))

        return updated_kwargs

    def load_factory(self):
        try:
            from sampling_zoo.core.api.api_main import SamplingStrategyFactory
        except ModuleNotFoundError as ex:
            raise ModuleNotFoundError(
                'SamplingZoo provider is unavailable. Install optional dependencies for Sampling Zoo '
                '(for example: pip install "fedot[sampling_zoo]").'
            ) from ex
        return SamplingStrategyFactory
