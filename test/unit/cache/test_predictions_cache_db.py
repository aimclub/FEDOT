import pytest
import sqlite3
from contextlib import closing
from typing import List, Tuple

from fedot.core.caching.predictions_cache_db import PredictionsCacheDB


class MockOutputData:
    # Avoid side effects of OutputData changes

    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MockOutputData) and self.value == other.value


@pytest.fixture
def tmp_cache(tmpdir) -> PredictionsCacheDB:
    """Fixture to create a temporary cache database."""
    return PredictionsCacheDB(cache_dir=tmpdir, use_stats=True)


def test_table_creation(tmp_cache: PredictionsCacheDB) -> None:
    """Test if the main and stats tables are created on initialization."""
    with closing(sqlite3.connect(tmp_cache.db_path)) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';")
        assert cursor.fetchone() is not None, "Main table 'predictions' not created."

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stats';")
        assert cursor.fetchone() is not None, "Stats table 'stats' not created."


def test_add_and_get_prediction(tmp_cache: PredictionsCacheDB) -> None:
    """Test adding a prediction and retrieving it successfully."""
    uid = "test_uid"
    data = MockOutputData(42)

    tmp_cache.add_prediction(uid, "pred", data)

    retrieved = tmp_cache.get_prediction(uid, "pred")
    assert retrieved == data, "Retrieved data does not match original data."

    stats: List[Tuple[str, int]] = tmp_cache.retrieve_stats()
    assert (uid, 1) in stats, "Retrieve count not incremented correctly."


def test_get_non_existent_prediction(tmp_cache: PredictionsCacheDB) -> None:
    """Test retrieving a non-existent prediction returns None and updates stats."""
    uid = "non_existent"

    retrieved = tmp_cache.get_prediction(uid, "pred")
    assert retrieved is None, "Non-existent UID should return None."

    stats: List[Tuple[str, int]] = tmp_cache.retrieve_stats()
    assert stats is None or all(entry[0] != uid for entry in stats), "Stats should not track non-existent UID."


def test_add_duplicate_prediction(tmp_cache: PredictionsCacheDB) -> None:
    """Test that duplicate UID predictions are ignored."""
    uid = "test_uid"

    data1 = MockOutputData(42)
    data2 = MockOutputData(43)

    tmp_cache.add_prediction(uid, "pred", data1)
    tmp_cache.add_prediction(uid, "pred", data2)

    retrieved = tmp_cache.get_prediction(uid, "pred")
    assert retrieved == data1, "Duplicate UID should retain original data."


def test_retrieve_stats(tmp_cache: PredictionsCacheDB) -> None:
    """Test retrieve_stats returns correct counts for multiple accesses."""
    uid1, uid2 = "uid1", "uid2"

    data = MockOutputData(42)

    tmp_cache.add_prediction(uid1, "pred", data)
    tmp_cache.add_prediction(uid2, "pred", data)

    tmp_cache.get_prediction(uid1, "pred")
    tmp_cache.get_prediction(uid1, "pred")
    tmp_cache.get_prediction(uid2, "pred")

    stats: List[Tuple[str, int]] = tmp_cache.retrieve_stats()
    expected_stats = [(uid1, 2), (uid2, 1)]
    assert sorted(stats) == sorted(expected_stats), "Retrieve stats do not match expected counts."


def test_multiple_retrieve_types(tmp_cache: PredictionsCacheDB) -> None:
    """Test retrieve_count increments regardless of prediction type."""
    uid = "test_uid"

    data = MockOutputData(42)

    tmp_cache.add_prediction(uid, "pred", data)
    tmp_cache.get_prediction(uid, "pred")
    tmp_cache.get_prediction(uid, "fit")

    stats: List[Tuple[str, int]] = tmp_cache.retrieve_stats()
    assert (uid, 2) in stats, "Retrieve count should increment per UID regardless of type."
