import fedot.api.api_utils.api_composer as composer_module
from fedot.api.api_utils.api_composer import ApiComposer


class _FakeCache:
    def __init__(self, cache_dir=None, use_stats=False):
        self.cache_dir = cache_dir
        self.use_stats = use_stats
        self.was_reset = False

    def reset(self):
        self.was_reset = True


class _FakeParams(dict):
    timeout = 1
    n_jobs = -1


def test_api_composer_init_cache_uses_typed_cache_plan(monkeypatch):
    monkeypatch.setattr(composer_module, 'OperationsCache', _FakeCache)
    monkeypatch.setattr(composer_module, 'PreprocessingCache', _FakeCache)
    monkeypatch.setattr(composer_module, 'PredictionsCache', _FakeCache)

    params = _FakeParams(
        use_operations_cache=True,
        use_preprocessing_cache=True,
        use_predictions_cache=True,
        use_input_preprocessing=False,
        cache_dir='cache_dir',
        use_stats=True,
    )

    composer = ApiComposer(params, metrics=['f1'])

    assert isinstance(composer.operations_cache, _FakeCache)
    assert composer.operations_cache.was_reset is True
    assert composer.preprocessing_cache is None
    assert isinstance(composer.predictions_cache, _FakeCache)
    assert composer.predictions_cache.was_reset is True
