import sqlite3
import json

import numpy as np
import pytest
import torch

import fedot.core.caching.index_db as index_db_module
import fedot.core.caching.inmemory_operations as inmemory_operations
import fedot.core.caching.tools as cache_tools
import fedot.core.caching.tracer as tracer_module
from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator


@pytest.fixture()
def isolated_cache_dir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(index_db_module, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(inmemory_operations, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(cache_tools, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(tracer_module, "CACHE_DIR", cache_dir)
    return cache_dir


def _make_features() -> np.ndarray:
    return np.array(
        [
            [0.0, 10.0, 0.0],
            [1.0, 11.0, 1.0],
            [2.0, 12.0, 0.0],
            [3.0, 13.0, 1.0],
        ],
        dtype=np.float32,
    )


def _tensor_cache_rows(cache_dir):
    with sqlite3.connect(cache_dir / "index.sqlite3") as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT input_hash, output_hash, operation_hash, path
            FROM tensor_data_cache;
            """
        )
        return cur.fetchall()


@pytest.mark.unit
def test_tensor_data_creator_first_run_writes_tensor_data_cache(isolated_cache_dir):
    tensor_data = TensorDataCreator.create(_make_features(), backend_name="cpu")

    rows = _tensor_cache_rows(isolated_cache_dir)

    assert isinstance(tensor_data, TensorData)
    assert len(rows) == 1
    _, output_hash, _, path = rows[0]
    assert output_hash
    assert path.endswith(".pt")
    assert (isolated_cache_dir / "tensor_data").exists()
    assert rows[0][3] and tensor_data.features.device.type == "cpu"


@pytest.mark.unit
def test_tensor_data_creator_second_run_returns_cached_tensor_data(isolated_cache_dir, monkeypatch):
    features = _make_features()
    first = TensorDataCreator.create(features, backend_name="cpu")

    def fail_if_regular_tensor_data_build_is_used(self):
        raise AssertionError("TensorData should be loaded from cache on the second create call")

    monkeypatch.setattr(TensorDataCreator, "to_tensor_data", fail_if_regular_tensor_data_build_is_used)

    second = TensorDataCreator.create(features.copy(), backend_name="cpu")

    assert second == first
    assert len(_tensor_cache_rows(isolated_cache_dir)) == 1


@pytest.mark.unit
def test_tensor_data_creator_different_input_creates_separate_cache_record(isolated_cache_dir):
    first_features = _make_features()
    second_features = first_features.copy()
    second_features[0, 0] = 42.0

    first = TensorDataCreator.create(first_features, backend_name="cpu")
    second = TensorDataCreator.create(second_features, backend_name="cpu")

    rows = _tensor_cache_rows(isolated_cache_dir)
    input_hashes = {row[0] for row in rows}
    paths = {row[3] for row in rows}

    assert first != second
    assert len(rows) == 2
    assert len(input_hashes) == 2
    assert len(paths) == 2


@pytest.mark.unit
def test_cache_index_db_is_created_with_tensor_and_model_tables(isolated_cache_dir):
    db = CacheIndexDB()

    with sqlite3.connect(db.db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        tables = {name for (name,) in cur.fetchall()}

    assert db.db_path == isolated_cache_dir / "index.sqlite3"
    assert CacheIndexDB.TENSOR_DATA_TABLE in tables
    assert CacheIndexDB.PREPROCESSING_MODELS_TABLE in tables


@pytest.mark.unit
def test_tensor_data_creator_writes_trace_manifest_for_fit_state(isolated_cache_dir):
    tensor_data = TensorDataCreator.create(_make_features(), backend_name="cpu")
    trace_paths = list((isolated_cache_dir / "traces").glob("*.json"))

    assert len(trace_paths) == 1

    with open(trace_paths[0], encoding="utf-8") as file:
        trace = json.load(file)

    assert trace["raw_fingerprint"] == tensor_data.raw_fingerprint
    assert trace["final_output_hash"] == tensor_data.ready_fingerprint
    assert len(trace["stages"]) == 1
    assert trace["stages"][0]["stage"] == "obligatory_preprocessing"
    assert trace["stages"][0]["input_hash"] == tensor_data.raw_fingerprint
    assert trace["stages"][0]["output_hash"] == tensor_data.ready_fingerprint
    assert trace["stages"][0]["tensor_data_path"].endswith(".pt")
    assert trace["stages"][0]["operation_path"].endswith(".pkl")


@pytest.mark.unit
def test_trace_builder_updates_existing_manifest_by_trace_uuid(isolated_cache_dir):
    index_db = CacheIndexDB()
    index_db.add_preprocessing_plan("plan-1", isolated_cache_dir / "plans" / "plan-1.pkl")
    index_db.add_preprocessing_plan("plan-2", isolated_cache_dir / "plans" / "plan-2.pkl")
    index_db.add_tensor_data(
        input_hash="raw",
        output_hash="after-obligatory",
        operation_hash="plan-1",
        path=isolated_cache_dir / "tensor_data" / "after-obligatory.pt",
    )
    index_db.add_tensor_data(
        input_hash="after-obligatory",
        output_hash="after-optional",
        operation_hash="plan-2",
        path=isolated_cache_dir / "tensor_data" / "after-optional.pt",
    )

    trace_builder = TraceBuilder("raw", index_db=index_db)
    trace_builder.add_stage("obligatory_preprocessing", "raw", "plan-1")
    trace_path = trace_builder.save(final_output_hash="after-obligatory")

    loaded_builder = TraceBuilder.from_trace_uuid(trace_builder.trace_id, index_db=index_db)
    loaded_builder.add_stage("optional_preprocessing", "after-obligatory", "plan-2")
    updated_trace_path = loaded_builder.save(final_output_hash="after-optional")

    assert updated_trace_path == trace_path
    assert len(list((isolated_cache_dir / "traces").glob("*.json"))) == 1

    with open(updated_trace_path, encoding="utf-8") as file:
        trace = json.load(file)

    assert trace["trace_id"] == trace_builder.trace_id
    assert trace["final_output_hash"] == "after-optional"
    assert [stage["stage"] for stage in trace["stages"]] == [
        "obligatory_preprocessing",
        "optional_preprocessing",
    ]
