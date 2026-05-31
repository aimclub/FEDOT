import numpy as np
import pandas as pd

from fedot.api.sampling_stage.providers import SamplingZooProvider


class FakeChunkingFactory:
    def __init__(self, partitions, strategy=None):
        self.partitions = partitions
        self.strategy = strategy or object()

    def fit_transform(self, **kwargs):
        return self.strategy, self.partitions


class FakeBudgetedStrategy:
    budget_policy_ = {'applied': True, 'selected_size': 5}


def test_chunking_provider_preserves_feature_target_partition_contract():
    feature_chunk = pd.DataFrame({'a': [10, 20]}, index=[4, 7])
    target_chunk = pd.Series([0, 1], index=[4, 7])
    partitions = {
        'chunk_0': {
            'feature': feature_chunk,
            'target': target_chunk,
        },
    }

    result = SamplingZooProvider._sample_chunking(
        factory=FakeChunkingFactory(partitions),
        features=np.array([[1], [2]]),
        target=np.array([0, 1]),
        strategy='random',
        strategy_params={},
    )

    assert result.partitions == partitions
    assert result.meta['strategy_kind'] == 'chunking'


def test_chunking_provider_exposes_strategy_budget_metadata():
    partitions = {
        'chunk_0': {
            'feature': pd.DataFrame({'a': [1]}),
            'target': pd.Series([0]),
        },
    }

    result = SamplingZooProvider._sample_chunking(
        factory=FakeChunkingFactory(partitions, strategy=FakeBudgetedStrategy()),
        features=np.array([[1]]),
        target=np.array([0]),
        strategy='random',
        strategy_params={},
    )

    assert result.meta['budget_policy'] == {'applied': True, 'selected_size': 5}
