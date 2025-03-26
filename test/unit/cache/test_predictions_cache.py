import pytest
from unittest.mock import MagicMock, ANY


from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.caching.predictions_cache_db import PredictionsCacheDB
from fedot.core.data.data import OutputData


@pytest.fixture
def mock_db() -> MagicMock:
    """Fixture to mock PredictionsCacheDB"""
    return MagicMock(spec=PredictionsCacheDB)


@pytest.fixture
def predictions_cache(mock_db: MagicMock) -> PredictionsCache:
    """Fixture to create PredictionsCache with mocked database"""
    cache = PredictionsCache(cache_dir=None)
    cache._db = mock_db
    cache.log = MagicMock()
    return cache


def test_save_prediction_calls_db(predictions_cache: PredictionsCache, mock_db: MagicMock):
    """Test that saving prediction calls DB with correct parameters"""

    test_data = MagicMock(spec=OutputData)
    params = {
        "descriptive_id": "node_123",
        "output_mode": "default",
        "fold_id": 0,
        "outputData": test_data,
        "is_fit": False
    }

    predictions_cache.save_node_prediction(**params)

    expected_uid = "pred_node_123_default_0"
    mock_db.add_prediction.assert_called_once_with(expected_uid, "pred", test_data)


def test_save_ransac_prediction_skipped(predictions_cache: PredictionsCache, mock_db: MagicMock):
    """Test that ransac predictions are not cached"""
    test_data = MagicMock(spec=OutputData)
    params = {
        "descriptive_id": "ransac_node_123",
        "output_mode": "default",
        "fold_id": 0,
        "outputData": test_data,
        "is_fit": False
    }

    predictions_cache.save_node_prediction(**params)

    mock_db.add_prediction.assert_not_called()


def test_load_prediction_calls_db(predictions_cache: PredictionsCache, mock_db: MagicMock):
    """Test that loading prediction calls DB with correct parameters"""
    params = {
        "descriptive_id": "node_456",
        "output_mode": "test_mode",
        "fold_id": 1,
        "is_fit": True
    }
    mock_db.get_prediction.return_value = MagicMock(spec=OutputData)

    result = predictions_cache.load_node_prediction(**params)

    expected_uid = "fit_node_456_test_mode_1"
    mock_db.get_prediction.assert_called_once_with(expected_uid, "fit")
    assert isinstance(result, OutputData)


def test_load_nonexistent_prediction(predictions_cache: PredictionsCache, mock_db: MagicMock):
    """Test loading missing prediction returns None"""
    mock_db.get_prediction.return_value = None
    params = {
        "descriptive_id": "missing_node",
        "output_mode": "default",
        "fold_id": 0,
        "is_fit": False
    }

    result = predictions_cache.load_node_prediction(**params)

    assert result is None
    predictions_cache.log.debug.assert_called_with("--- MISS prediction cache: pred_missing_node_default_0")


def test_uid_generation_variants(predictions_cache: PredictionsCache):
    """Test different UID generation scenarios"""
    test_cases = [
        (
            {"descriptive_id": "node", "output_mode": "mode", "fold_id": 1, "is_fit": True},
            "fit_node_mode_1"
        ),
        (
            {"descriptive_id": "long_node_id", "output_mode": "special_mode", "fold_id": 42, "is_fit": False},
            "pred_long_node_id_special_mode_42"
        ),
        (
            {"descriptive_id": "node", "output_mode": "", "fold_id": 0, "is_fit": False},
            "pred_node__0"
        )
    ]

    for params, expected_uid in test_cases:
        save_params = {
            **params,
            "outputData": MagicMock(spec=OutputData)
        }

        predictions_cache.save_node_prediction(**save_params)
        predictions_cache._db.add_prediction.assert_called_with(
            expected_uid,
            "fit" if params["is_fit"] else "pred",
            ANY
        )
        predictions_cache._db.reset_mock()

        predictions_cache.load_node_prediction(**params)
        predictions_cache._db.get_prediction.assert_called_with(
            expected_uid,
            "fit" if params["is_fit"] else "pred"
        )
        predictions_cache._db.reset_mock()


def test_cache_logging(predictions_cache: PredictionsCache, mock_db: MagicMock):
    """Test logging of cache operations"""
    test_data = MagicMock(spec=OutputData)
    predictions_cache.save_node_prediction(
        descriptive_id="node",
        output_mode="mode",
        fold_id=0,
        outputData=test_data
    )
    predictions_cache.log.debug.assert_any_call("--- SAVE prediction cache: pred_node_mode_0")

    mock_db.get_prediction.return_value = test_data
    predictions_cache.load_node_prediction(
        descriptive_id="node",
        output_mode="mode",
        fold_id=0
    )
    predictions_cache.log.debug.assert_any_call("--- HIT prediction cache: pred_node_mode_0")

    mock_db.get_prediction.return_value = None
    predictions_cache.load_node_prediction(
        descriptive_id="missing",
        output_mode="mode",
        fold_id=0
    )
    predictions_cache.log.debug.assert_any_call("--- MISS prediction cache: pred_missing_mode_0")


def test_fit_vs_pred_type_handling(predictions_cache: PredictionsCache, mock_db: MagicMock):
    """Test proper handling of fit/prediction type differentiation"""
    predictions_cache.save_node_prediction(
        descriptive_id="fit_node",
        output_mode="mode",
        fold_id=0,
        outputData=MagicMock(spec=OutputData),
        is_fit=True
    )
    mock_db.add_prediction.assert_called_with("fit_fit_node_mode_0", "fit", ANY)

    predictions_cache.save_node_prediction(
        descriptive_id="pred_node",
        output_mode="mode",
        fold_id=0,
        outputData=MagicMock(spec=OutputData),
        is_fit=False
    )
    mock_db.add_prediction.assert_called_with("pred_pred_node_mode_0", "pred", ANY)
