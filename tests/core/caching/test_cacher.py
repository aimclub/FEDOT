import sqlite3
import json

import numpy as np
import pytest
import torch

from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.planner import PreprocessingPlan


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


def _make_tensor_data_for_cache() -> TensorData:
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        target=torch.tensor([0.0, 1.0]),
    )


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

    assert trace["raw_fingerprint"] == trace["stages"][0]["input_hash"]
    assert trace["final_output_hash"] == tensor_data.fingerprint
    assert len(trace["stages"]) == 1
    assert trace["stages"][0]["stage"] == "obligatory_preprocessing"
    assert trace["stages"][0]["input_hash"] == trace["raw_fingerprint"]
    assert trace["stages"][0]["output_hash"] == tensor_data.fingerprint
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


@pytest.mark.unit
def test_index_db_does_not_overwrite_existing_tensor_record_with_null_path(isolated_cache_dir):
    index_db = CacheIndexDB()
    saved_record = index_db.add_tensor_data(
        input_hash="input",
        output_hash="saved-output",
        operation_hash="operation",
        path=isolated_cache_dir / "tensor_data" / "saved-output.pt",
    )

    null_path_record = index_db.add_tensor_data(
        input_hash="input",
        output_hash="new-output",
        operation_hash="operation",
        path=None,
    )

    assert null_path_record == saved_record
    assert index_db.get_tensor_data("input", "operation") == saved_record


@pytest.mark.unit
def test_cacher_with_disabled_tensor_cache_does_not_overwrite_saved_path(isolated_cache_dir):
    index_db = CacheIndexDB()
    tensor_data = _make_tensor_data_for_cache()
    output_hash = Hasher.hash(tensor_data)
    saved_record = index_db.add_tensor_data(
        input_hash="input",
        output_hash=output_hash,
        operation_hash="operation",
        path=isolated_cache_dir / "tensor_data" / f"{output_hash}.pt",
    )

    null_path_record = Cacher(index_db=index_db, use_cache=False).cache_tensor_data(
        output_data=tensor_data,
        output_hash=output_hash,
        input_hash="input",
        operation_hash="operation",
    )

    assert null_path_record == saved_record
    assert index_db.get_tensor_data("input", "operation") == saved_record


@pytest.mark.unit
def test_cacher_with_enabled_tensor_cache_updates_null_path_record(isolated_cache_dir):
    index_db = CacheIndexDB()
    tensor_data = _make_tensor_data_for_cache()
    output_hash = Hasher.hash(tensor_data)
    null_path_record = index_db.add_tensor_data(
        input_hash="input",
        output_hash=output_hash,
        operation_hash="operation",
        path=None,
    )

    saved_record = Cacher(index_db=index_db, use_cache=True).cache_tensor_data(
        output_data=tensor_data,
        output_hash=output_hash,
        input_hash="input",
        operation_hash="operation",
    )

    assert null_path_record.path is None
    assert saved_record.path is not None
    assert saved_record.path.exists()
    assert index_db.get_tensor_data("input", "operation").path == saved_record.path


@pytest.mark.unit
def test_cacher_indexes_and_traces_tensor_data_without_saving_tensor_artifact(isolated_cache_dir):
    index_db = CacheIndexDB()
    cacher = Cacher(index_db=index_db, use_cache=False)
    raw_features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor_data = _make_tensor_data_for_cache()
    output_hash = Hasher.hash(tensor_data)
    input_hash = Hasher.hash(raw_features)
    operation = PreprocessingPlan()
    operation_hash = Hasher.hash(operation)
    index_db.add_preprocessing_plan(operation_hash, isolated_cache_dir / "preprocessing_plans" / "plan.pkl")

    record = cacher.cache_tensor_data(
        output_data=tensor_data,
        output_hash=output_hash,
        input_hash=input_hash,
        operation_hash=operation_hash,
        trace_stage="obligatory_preprocessing",
    )
    load_response = cacher.load_tensor_data(
        input_data=raw_features,
        operation=operation,
    )
    indexed_record = index_db.get_tensor_data(input_hash, operation_hash)
    trace_paths = list((isolated_cache_dir / "traces").glob("*.json"))

    assert record.path is None
    assert indexed_record.path is None
    assert tensor_data.trace_uuid is not None
    assert not list((isolated_cache_dir / "tensor_data").glob("*.pt"))
    assert load_response.success is False
    assert len(trace_paths) == 1

    with open(trace_paths[0], encoding="utf-8") as file:
        trace = json.load(file)

    assert trace["stages"][0]["tensor_data_path"] is None
    assert trace["stages"][0]["output_hash"] == output_hash


@pytest.mark.unit
def test_cacher_clears_cache(isolated_cache_dir):
    index_db = CacheIndexDB()
    cacher = Cacher(index_db=index_db, use_cache=False)
    cacher.cache_tensor_data(
        output_data=TensorDataCreator.create(np.random.rand(10, 10), backend_name="cpu"),
        output_hash="tensor-hash",
        input_hash="input-hash",
        operation_hash="operation-hash",
    )
    cacher.clear_cache(mode="all")

    assert not (isolated_cache_dir / "tensor_data" / "tensor-hash.pt").exists()
    assert not (isolated_cache_dir / "traces" / "trace-hash.json").exists()
    assert not (isolated_cache_dir / "index.sqlite3").exists()
