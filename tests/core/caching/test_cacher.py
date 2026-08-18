import sqlite3
import json

import numpy as np
import pytest
import torch

from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.cacher import Cacher, ensure_cacher
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.evaluation.operation_implementations.models.torch import TorchLinearClassifier
from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
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
    assert CacheIndexDB.OPERATIONS_TABLE in tables


@pytest.mark.unit
def test_ensure_cacher_returns_existing_or_default_instance():
    existing = Cacher(use_cache=False)
    created = ensure_cacher(use_cache=False)

    assert ensure_cacher(existing) is existing
    assert isinstance(created, Cacher)
    assert created.use_cache is False
    assert ensure_cacher().use_cache is True


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
def test_cacher_indexes_preprocessing_model_without_saving_artifact(isolated_cache_dir):
    index_db = CacheIndexDB()
    cacher = Cacher(index_db=index_db, use_cache=False)
    model = {"fitted": True, "method": "mean"}

    record = cacher.cache_preprocessing_model(
        input_hash="input-hash",
        model=model,
        model_hash="model-hash",
        operation_hash="operation-hash",
        step_order=0,
        step_name="imputation",
        method="mean",
    )
    load_response = cacher.load_preprocessing_model(
        input_hash="input-hash",
        operation_hash="operation-hash",
    )
    indexed_record = index_db.get_preprocessing_model("input-hash", "operation-hash")

    assert record.path is None
    assert indexed_record.path is None
    assert not list((isolated_cache_dir / "preprocessing_models").glob("*.pkl"))
    assert load_response.success is False


@pytest.mark.unit
def test_cacher_with_enabled_cache_saves_preprocessing_model(isolated_cache_dir):
    from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler

    class _StubHandler(AbstractPreprocessingHandler):
        def fit(self, data, features_idx):
            return self

        def transform(self, data):
            return data

    index_db = CacheIndexDB()
    cacher = Cacher(index_db=index_db, use_cache=True)
    model = _StubHandler()

    record = cacher.cache_preprocessing_model(
        input_hash="input-hash",
        model=model,
        model_hash="model-hash",
        operation_hash="operation-hash",
        step_order=0,
        step_name="imputation",
        method="mean",
    )

    assert record.path is not None
    assert record.path.exists()
    assert index_db.get_preprocessing_model("input-hash", "operation-hash").path == record.path


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


def _init_logit_operation(tensor_data: TensorData, cacher: Cacher) -> Model:
    operation = Model("logit", cacher=cacher)
    operation._init(tensor_data.task, n_samples_data=tensor_data.features.shape[0])
    return operation


def _make_torch_linear_train_data() -> TensorData:
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        features=torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ),
        target=torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )


@pytest.mark.unit
def test_pipeline_torch_linear_caches_fitted_operation_as_pkl(isolated_cache_dir):
    tensor_data = _make_torch_linear_train_data()
    cacher = Cacher(use_cache=True)
    node = PipelineNode("torch_linear", cacher=cacher)
    node.parameters = {"epochs": 20, "learning_rate": 0.05}
    pipeline = Pipeline(node, cacher=cacher)

    pipeline.fit(tensor_data)

    fitted = pipeline.root_node.fitted_operation
    operation_files = list((isolated_cache_dir / "operations").glob("*.pkl"))
    tensor_files = list((isolated_cache_dir / "tensor_data").glob("*.pt"))
    loaded = cacher.load_operation(node.operation, tensor_data)

    assert pipeline.is_fitted
    assert isinstance(fitted, TorchLinearClassifier)
    assert fitted.module is not None
    assert len(operation_files) == 1
    assert len(tensor_files) == 1
    assert isinstance(loaded, TorchLinearClassifier)
    assert loaded.module is not None
    assert torch.equal(loaded.module.weight.detach().cpu(), fitted.module.weight.detach().cpu())
    assert torch.equal(loaded.module.bias.detach().cpu(), fitted.module.bias.detach().cpu())


def _fit_torch_linear_pipeline(tensor_data: TensorData, cacher: Cacher) -> Pipeline:
    node = PipelineNode("torch_linear", cacher=cacher)
    node.parameters = {"epochs": 20, "learning_rate": 0.05}
    pipeline = Pipeline(node, cacher=cacher)
    pipeline.fit(tensor_data)
    return pipeline


def _make_torch_linear_test_data(train_data: TensorData) -> TensorData:
    return TensorData(
        task=train_data.task,
        data_type=train_data.data_type,
        features=train_data.features + 0.5,
        target=train_data.target,
    )


@pytest.mark.unit
def test_pipeline_predict_reuses_inmemory_operation_when_cache_disabled(isolated_cache_dir):
    train_data = _make_torch_linear_train_data()
    test_data = _make_torch_linear_test_data(train_data)
    pipeline = _fit_torch_linear_pipeline(train_data, Cacher(use_cache=False))
    fitted = pipeline.root_node.fitted_operation
    predict_calls = {"count": 0}
    original_predict_proba = fitted.predict_proba

    def wrapped_predict_proba(features):
        predict_calls["count"] += 1
        return original_predict_proba(features)

    fitted.predict_proba = wrapped_predict_proba

    result = pipeline.predict(test_data)

    assert pipeline.root_node.fitted_operation is fitted
    assert predict_calls["count"] == 1
    assert result.predict is not None
    assert not list((isolated_cache_dir / "operations").glob("*.pkl"))


@pytest.mark.unit
def test_pipeline_predict_uses_inmemory_operation_before_cache(isolated_cache_dir):
    train_data = _make_torch_linear_train_data()
    test_data = _make_torch_linear_test_data(train_data)
    cacher = Cacher(use_cache=True)
    pipeline = _fit_torch_linear_pipeline(train_data, cacher)
    fitted = pipeline.root_node.fitted_operation
    predict_calls = {"count": 0}
    original_predict_proba = fitted.predict_proba

    def wrapped_predict_proba(features):
        predict_calls["count"] += 1
        return original_predict_proba(features)

    fitted.predict_proba = wrapped_predict_proba

    result = pipeline.predict(test_data)

    assert pipeline.root_node.fitted_operation is fitted
    assert predict_calls["count"] == 1
    assert result.predict is not None
    assert cacher.load_operation(pipeline.root_node.operation, train_data) is not None

    second = pipeline.predict(test_data)
    assert predict_calls["count"] == 1
    assert torch.equal(second.predict.detach().cpu(), result.predict.detach().cpu())


@pytest.mark.unit
def test_predict_loads_operation_from_cache_when_missing_in_memory(isolated_cache_dir):
    train_data = _make_torch_linear_train_data()
    cacher = Cacher(use_cache=True)
    pipeline = _fit_torch_linear_pipeline(train_data, cacher)
    operation = pipeline.root_node.operation
    original_weight = pipeline.root_node.fitted_operation.module.weight.detach().clone()

    result = operation.predict(
        fitted_operation=None,
        data=train_data,
        params=pipeline.root_node.parameters,
    )

    assert result.predict is not None
    restored = cacher.load_operation(operation, train_data)
    assert restored is not None
    assert torch.equal(restored.module.weight.detach().cpu(), original_weight.cpu())


@pytest.mark.unit
def test_cacher_skips_optional_preprocessing_operation(isolated_cache_dir):
    tensor_data = _make_tensor_data_for_cache()
    cacher = Cacher(use_cache=True)
    operation = DataOperation("optional_preprocessing")
    operation.fitted_operation = object()

    assert cacher.cache_operation(operation, tensor_data) is None
    assert cacher.load_operation(operation, tensor_data) is None
    assert not list((isolated_cache_dir / "operations").glob("*.pkl"))


@pytest.mark.unit
def test_cacher_with_disabled_cache_does_not_write_operation_artifact(isolated_cache_dir):
    tensor_data = _make_tensor_data_for_cache()
    cacher = Cacher(use_cache=False)
    operation = _init_logit_operation(tensor_data, cacher)
    operation.fitted_operation = {"weights": [1.0, 2.0]}

    record = cacher.cache_operation(operation, tensor_data)

    assert record is not None
    assert record.path is None
    assert cacher.load_operation(operation, tensor_data) is None
    assert not list((isolated_cache_dir / "operations").glob("*.pkl"))


@pytest.mark.unit
def test_operation_fit_reuses_cached_fitted_model(isolated_cache_dir):
    tensor_data = _make_tensor_data_for_cache()
    cacher = Cacher(use_cache=True)
    fit_calls = {"count": 0}

    def _bind_fake_strategy(operation: Model):
        original_init = operation._init

        def wrapped_init(*args, **kwargs):
            original_init(*args, **kwargs)

            def fake_fit(train_data):
                fit_calls["count"] += 1
                return {"marker": "fitted-logit"}

            def fake_predict_for_fit(trained_operation, predict_data):
                return predict_data

            operation._eval_strategy.fit = fake_fit
            operation._eval_strategy.predict_for_fit = fake_predict_for_fit

        operation._init = wrapped_init

    first = Model("logit", cacher=cacher)
    _bind_fake_strategy(first)
    fitted, _ = first.fit(params=None, data=tensor_data)

    second = Model("logit", cacher=cacher)
    _bind_fake_strategy(second)
    loaded, _ = second.fit(params=None, data=tensor_data)

    assert fitted == {"marker": "fitted-logit"}
    assert loaded == {"marker": "fitted-logit"}
    assert fit_calls["count"] == 1
