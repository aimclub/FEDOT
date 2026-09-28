"""Real CPU model lifecycle across the fitted preparation boundary."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fedot import Fedot, create_data
from fedot.core.caching.cache_loader import Loader
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline


@pytest.mark.integration
@pytest.mark.parametrize('string_labels', [False, True])
def test_predefined_model_reuses_preparation_without_cache(
        string_labels, isolated_cache_dir, monkeypatch):
    features = np.array([[0., 1.], [1., 0.], [0., 2.],
                        [2., 0.]], dtype=np.float32)
    target = np.array(['cat', 'dog', 'cat', 'dog']
                      if string_labels else [0, 1, 0, 1])
    original_features, original_target = features.copy(), target.copy()
    train = create_data(features, target=target, use_cache=False)
    train_before = deepcopy(train)
    state_before = train.preparation_state
    node = PrimaryNode('torch_linear')
    node.parameters = {'epochs': 3, 'learning_rate': 0.01}
    model = Fedot(problem='classification', use_cache=False, seed=7, n_jobs=1)
    pipeline = model.fit(train, predefined_model=Pipeline(node))
    assert pipeline.is_fitted

    def forbid_disk_state(*args, **kwargs):
        raise AssertionError(
            'prediction must use training-owned preparation, not disk state')

    monkeypatch.setattr(Loader, 'load', forbid_disk_state)
    monkeypatch.setattr(TraceBuilder, 'from_trace_uuid', forbid_disk_state)
    test = create_data(features[:2], from_data=train, use_cache=False)
    test_before = deepcopy(test)
    first = model.predict(test)
    second = model.predict(test)
    assert first.predict.shape == (2,)
    assert np.array_equal(np.asarray(first.predict),
                          np.asarray(second.predict))
    assert set(np.asarray(first.predict)) <= set(target)
    assert train == train_before
    assert test == test_before
    assert train.preparation_state == state_before
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(target, original_target)
    torch.testing.assert_close(train.features, train_before.features)
