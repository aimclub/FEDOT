import os

import numpy as np
import pytest

from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture
def predictions_cache(tmpdir):
    """Fixture with real cache instance and temp directory"""
    cache_dir = str(tmpdir.mkdir("predictions_cache"))
    cache = PredictionsCache(cache_dir=cache_dir)
    yield cache
    # Cleanup
    db_path = os.path.join(cache_dir, "predictions.db")
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture()
def output_table_1d():
    task = Task(TaskTypesEnum.classification)

    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    threshold = 0.5
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)

    data = OutputData(idx=np.arange(0, samples), features=x, predict=classes,
                      task=task, data_type=DataTypesEnum.table)
    return data


def test_save_prediction_calls_db(predictions_cache: PredictionsCache, output_table_1d: OutputData):
    """Test saving and loading predictions"""
    test_data = output_table_1d
    params = {
        "descriptive_id": "node_123",
        "output_mode": "default",
        "fold_id": 0,
        "outputData": test_data,
        "is_fit": False
    }

    predictions_cache.save_node_prediction(**params)

    result = predictions_cache.load_node_prediction(
        descriptive_id="node_123",
        output_mode="default",
        fold_id=0
    )

    assert result is not None


def test_save_ransac_prediction_skipped(predictions_cache: PredictionsCache, output_table_1d: OutputData):
    """Test that ransac predictions are not cached"""
    test_data = output_table_1d
    params = {
        "descriptive_id": "ransac_node_123",
        "output_mode": "default",
        "fold_id": 0,
        "outputData": test_data,
        "is_fit": False
    }

    predictions_cache.save_node_prediction(**params)
    result = predictions_cache.load_node_prediction(
        descriptive_id="ransac_node_123",
        output_mode="default",
        fold_id=0
    )

    assert result is None


def test_load_nonexistent_prediction(predictions_cache):
    """Test loading missing prediction returns None"""
    result = predictions_cache.load_node_prediction(
        descriptive_id="missing_node",
        output_mode="default",
        fold_id=0
    )

    assert result is None


def test_uid_generation_variants(predictions_cache: PredictionsCache, output_table_1d: OutputData):
    """Test different UID generation scenarios"""
    test_cases = [
        {
            "params": {
                "descriptive_id": "node",
                "output_mode": "mode",
                "fold_id": 1,
                "is_fit": True
            },
            "expected_uid": "fit_node_mode_1"
        },
        {
            "params": {
                "descriptive_id": "long_node_id",
                "output_mode": "special_mode",
                "fold_id": 42,
                "is_fit": False
            },
            "expected_uid": "pred_long_node_id_special_mode_42"
        }
    ]

    for case in test_cases:
        save_params = {
            **case["params"],
            "outputData": output_table_1d
        }
        predictions_cache.save_node_prediction(**save_params)

        result = predictions_cache.load_node_prediction(**case["params"])
        assert result is not None
        assert isinstance(result, OutputData)


def test_fit_vs_pred_type_handling(predictions_cache: PredictionsCache, output_table_1d: OutputData):
    """Test proper handling of fit/prediction type differentiation"""
    predictions_cache.save_node_prediction(
        descriptive_id="fit_node",
        output_mode="mode",
        fold_id=0,
        outputData=output_table_1d,
        is_fit=True
    )

    predictions_cache.save_node_prediction(
        descriptive_id="pred_node",
        output_mode="mode",
        fold_id=0,
        outputData=output_table_1d,
        is_fit=False
    )

    fit_result = predictions_cache.load_node_prediction(
        descriptive_id="fit_node",
        output_mode="mode",
        fold_id=0,
        is_fit=True
    )
    pred_result = predictions_cache.load_node_prediction(
        descriptive_id="pred_node",
        output_mode="mode",
        fold_id=0
    )

    assert fit_result is not None
    assert pred_result is not None
