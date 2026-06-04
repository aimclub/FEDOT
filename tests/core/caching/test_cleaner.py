import json
import os
import sqlite3
import time

import pytest

from fedot.core.caching.cache_cleaner import CacheCleaner
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.enums import CacheModeEnum
from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.normalization import (
    DEFAULT_N_FIRST_TENSOR_DATA,
    get_all_tensor_data_hashes,
    normilize_cleaning_strategy,
)


def _tensor_cache_rows(cache_dir):
    with sqlite3.connect(cache_dir / "index.sqlite3") as conn:
        cur = conn.cursor()
        cur.execute("SELECT output_hash, path FROM tensor_data_cache;")
        return cur.fetchall()


def _seed_tensor_artifact(cache_dir, output_hash: str, input_hash: str = "input", operation_hash: str = "op"):
    tensor_dir = cache_dir / "tensor_data"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    path = tensor_dir / f"{output_hash}.pt"
    path.write_text("tensor")
    return path


def _seed_tensor_index(
        index_db: CacheIndexDB,
        cache_dir,
        output_hash: str,
        input_hash: str = "input",
        operation_hash: str = "op"):
    path = _seed_tensor_artifact(cache_dir, output_hash, input_hash, operation_hash)
    return index_db.add_tensor_data(
        input_hash=input_hash,
        output_hash=output_hash,
        operation_hash=operation_hash,
        path=path,
    )


def _touch_older(path, age_seconds: float):
    old_time = time.time() - age_seconds
    os.utime(path, (old_time, old_time))


@pytest.mark.unit
def test_normalize_cleaning_strategy_all_mode():
    params = normilize_cleaning_strategy(mode=CacheModeEnum.ALL, tensor_data_hashes=None, ratio_first_tensor_data=None)

    assert params.mode == CacheModeEnum.ALL
    assert params.tensor_data_hashes is None
    assert params.ratio_first_tensor_data is None


@pytest.mark.unit
def test_normalize_cleaning_strategy_invalid_mode_defaults_to_tensor_data(isolated_cache_dir):
    _seed_tensor_artifact(isolated_cache_dir, "hash-a")

    params = normilize_cleaning_strategy(mode="unknown", tensor_data_hashes=None, ratio_first_tensor_data=None)

    assert params.mode == CacheModeEnum.TENSOR_DATA
    assert params.tensor_data_hashes == ["hash-a"]


@pytest.mark.unit
def test_normalize_cleaning_strategy_tensor_data_collects_all_hashes(isolated_cache_dir):
    _seed_tensor_artifact(isolated_cache_dir, "one")
    _seed_tensor_artifact(isolated_cache_dir, "two")

    params = normilize_cleaning_strategy(mode="tensor_data", tensor_data_hashes=None, ratio_first_tensor_data=None)

    assert params.mode == CacheModeEnum.TENSOR_DATA
    assert set(params.tensor_data_hashes) == {"one", "two"}


@pytest.mark.unit
def test_normalize_cleaning_strategy_tensor_data_wraps_single_hash():
    params = normilize_cleaning_strategy(
        mode=CacheModeEnum.TENSOR_DATA,
        tensor_data_hashes="only-one",
        ratio_first_tensor_data=None,
    )

    assert params.tensor_data_hashes == ["only-one"]


@pytest.mark.unit
def test_normalize_cleaning_strategy_first_n_with_explicit_count(isolated_cache_dir):
    _seed_tensor_artifact(isolated_cache_dir, "a")
    _seed_tensor_artifact(isolated_cache_dir, "b")
    _seed_tensor_artifact(isolated_cache_dir, "c")

    params = normilize_cleaning_strategy(
        mode=CacheModeEnum.FIRST_N_TENSOR_DATA,
        tensor_data_hashes=None,
        ratio_first_tensor_data=2,
    )

    assert params.mode == CacheModeEnum.FIRST_N_TENSOR_DATA
    assert params.ratio_first_tensor_data == 2


@pytest.mark.unit
def test_normalize_cleaning_strategy_first_n_fraction_ratio(isolated_cache_dir):
    for name in ("a", "b", "c", "d"):
        _seed_tensor_artifact(isolated_cache_dir, name)

    params = normilize_cleaning_strategy(
        mode="first_n_tensor_data",
        tensor_data_hashes=None,
        ratio_first_tensor_data=0.5,
    )

    assert params.ratio_first_tensor_data == 2


@pytest.mark.unit
def test_normalize_cleaning_strategy_first_n_percent_ratio(isolated_cache_dir):
    for name in ("a", "b", "c", "d", "e"):
        _seed_tensor_artifact(isolated_cache_dir, name)

    params = normilize_cleaning_strategy(
        mode=CacheModeEnum.FIRST_N_TENSOR_DATA,
        tensor_data_hashes=None,
        ratio_first_tensor_data=40.0,
    )

    assert params.ratio_first_tensor_data == 2


@pytest.mark.unit
def test_normalize_cleaning_strategy_first_n_default_ratio(isolated_cache_dir):
    _seed_tensor_artifact(isolated_cache_dir, "only")

    params = normilize_cleaning_strategy(
        mode=CacheModeEnum.FIRST_N_TENSOR_DATA,
        tensor_data_hashes=None,
        ratio_first_tensor_data=None,
    )

    assert params.ratio_first_tensor_data == min(1, DEFAULT_N_FIRST_TENSOR_DATA)


@pytest.mark.unit
def test_get_all_tensor_data_hashes(isolated_cache_dir):
    _seed_tensor_artifact(isolated_cache_dir, "x")
    _seed_tensor_artifact(isolated_cache_dir, "y")

    assert set(get_all_tensor_data_hashes()) == {"x", "y"}


@pytest.mark.unit
def test_cache_cleaner_clear_tensor_data_removes_files_and_index(isolated_cache_dir):
    index_db = CacheIndexDB()
    _seed_tensor_index(index_db, isolated_cache_dir, "keep-me")
    _seed_tensor_index(index_db, isolated_cache_dir, "remove-me", input_hash="in2", operation_hash="op2")

    CacheCleaner(index_db).clear_tensor_data(["remove-me"])

    assert (isolated_cache_dir / "tensor_data" / "remove-me.pt").exists() is False
    assert (isolated_cache_dir / "tensor_data" / "keep-me.pt").exists()
    assert index_db.get_tensor_data_by_output_hash("remove-me") is None
    assert index_db.get_tensor_data_by_output_hash("keep-me") is not None
    assert len(_tensor_cache_rows(isolated_cache_dir)) == 1


@pytest.mark.unit
def test_clear_tensor_data_updates_trace_manifest(isolated_cache_dir):
    index_db = CacheIndexDB()
    output_hash = "cleared-out"
    path = _seed_tensor_artifact(isolated_cache_dir, output_hash)
    index_db.add_tensor_data("input", output_hash, "op", path)

    trace_path = isolated_cache_dir / "traces" / "trace-1.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "trace_id": "trace-1",
                "trace_hash": "old",
                "raw_fingerprint": "input",
                "final_output_hash": output_hash,
                "created_at": "2020-01-01T00:00:00+00:00",
                "stages": [
                    {
                        "stage": "obligatory_preprocessing",
                        "operation_hash": "op",
                        "input_hash": "input",
                        "output_hash": output_hash,
                        "tensor_data_path": str(path),
                        "operation_path": None,
                        "models": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    CacheCleaner(index_db).clear_tensor_data([output_hash])

    with open(trace_path, encoding="utf-8") as file:
        trace = json.load(file)

    assert trace["stages"][0]["tensor_data_path"] == ""
    assert trace["trace_hash"] != "old"


@pytest.mark.unit
def test_cache_cleaner_clear_tensor_data_orphan_file_without_index(isolated_cache_dir):
    path = _seed_tensor_artifact(isolated_cache_dir, "orphan")

    CacheCleaner(CacheIndexDB()).clear_tensor_data(["orphan"])

    assert not path.exists()


@pytest.mark.unit
def test_cache_cleaner_clear_first_n_tensor_data_by_mtime(isolated_cache_dir):
    index_db = CacheIndexDB()
    old_path = _seed_tensor_artifact(isolated_cache_dir, "old")
    mid_path = _seed_tensor_artifact(isolated_cache_dir, "mid")
    new_path = _seed_tensor_artifact(isolated_cache_dir, "new")
    _touch_older(old_path, 300)
    _touch_older(mid_path, 200)

    index_db.add_tensor_data("i1", "old", "op", old_path)
    index_db.add_tensor_data("i2", "mid", "op", mid_path)
    index_db.add_tensor_data("i3", "new", "op", new_path)

    CacheCleaner(index_db).clear_first_n_tensor_data(2)

    assert not old_path.exists()
    assert not mid_path.exists()
    assert new_path.exists()
    assert index_db.get_tensor_data_by_output_hash("old") is None
    assert index_db.get_tensor_data_by_output_hash("mid") is None
    assert index_db.get_tensor_data_by_output_hash("new") is not None


@pytest.mark.unit
def test_cache_cleaner_clear_first_n_zero_is_noop(isolated_cache_dir):
    path = _seed_tensor_artifact(isolated_cache_dir, "stay")
    index_db = CacheIndexDB()
    index_db.add_tensor_data("in", "stay", "op", path)

    CacheCleaner(index_db).clear_first_n_tensor_data(0)

    assert path.exists()
    assert index_db.get_tensor_data_by_output_hash("stay") is not None


@pytest.mark.unit
def test_cache_cleaner_clear_all_wipes_cache_tree(isolated_cache_dir):
    index_db = CacheIndexDB()
    _seed_tensor_index(index_db, isolated_cache_dir, "td")
    traces_dir = isolated_cache_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    (traces_dir / "trace.json").write_text("{}")

    CacheCleaner(index_db).clear_all()

    assert not (isolated_cache_dir / "tensor_data" / "td.pt").exists()
    assert not (traces_dir / "trace.json").exists()
    assert not (isolated_cache_dir / "index.sqlite3").exists()


@pytest.mark.unit
def test_cacher_clear_cache_default_removes_all_tensor_data(isolated_cache_dir):
    index_db = CacheIndexDB()
    _seed_tensor_index(index_db, isolated_cache_dir, "one")
    _seed_tensor_index(index_db, isolated_cache_dir, "two", input_hash="in2", operation_hash="op2")

    Cacher(index_db=index_db).clear_cache()

    assert _tensor_cache_rows(isolated_cache_dir) == []
    assert list((isolated_cache_dir / "tensor_data").glob("*.pt")) == []


@pytest.mark.unit
def test_cacher_clear_cache_tensor_data_by_hashes(isolated_cache_dir):
    index_db = CacheIndexDB()
    _seed_tensor_index(index_db, isolated_cache_dir, "drop")
    _seed_tensor_index(index_db, isolated_cache_dir, "keep", input_hash="in2", operation_hash="op2")

    Cacher(index_db=index_db).clear_cache(tensor_data_hashes="drop")

    assert index_db.get_tensor_data_by_output_hash("drop") is None
    assert index_db.get_tensor_data_by_output_hash("keep") is not None


@pytest.mark.unit
def test_cacher_clear_cache_first_n_tensor_data_mode(isolated_cache_dir):
    index_db = CacheIndexDB()
    old_path = _seed_tensor_artifact(isolated_cache_dir, "old")
    new_path = _seed_tensor_artifact(isolated_cache_dir, "new")
    _touch_older(old_path, 100)
    index_db.add_tensor_data("in1", "old", "op", old_path)
    index_db.add_tensor_data("in2", "new", "op", new_path)

    Cacher(index_db=index_db).clear_cache(
        mode=CacheModeEnum.FIRST_N_TENSOR_DATA,
        ratio_first_tensor_data=1,
    )

    assert not old_path.exists()
    assert new_path.exists()


@pytest.mark.unit
def test_cacher_clear_cache_all_mode(isolated_cache_dir):
    index_db = CacheIndexDB()
    _seed_tensor_index(index_db, isolated_cache_dir, "td")
    plan_dir = isolated_cache_dir / "preprocessing_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "plan.pkl").write_text("plan")

    Cacher(index_db=index_db).clear_cache(mode="all")

    assert list((isolated_cache_dir / "tensor_data").glob("*.pt")) == []
    assert not (plan_dir / "plan.pkl").exists()
