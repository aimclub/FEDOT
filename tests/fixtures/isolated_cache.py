import pytest

import fedot.core.caching.cache_cleaner as cache_cleaner_module
import fedot.core.caching.index_db as index_db_module
import fedot.core.caching.inmemory_operations as inmemory_operations
import fedot.core.caching.normalization as normalization_module
import fedot.core.caching.tools as cache_tools
import fedot.core.caching.tracer as tracer_module


CACHE_DIR_MODULES = (
    index_db_module,
    inmemory_operations,
    cache_tools,
    cache_cleaner_module,
    normalization_module,
    tracer_module,
)


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """Redirect FEDOT disk cache to a per-test temporary directory."""
    cache_dir = tmp_path / "cache"
    for module in CACHE_DIR_MODULES:
        monkeypatch.setattr(module, "CACHE_DIR", cache_dir)
    return cache_dir
