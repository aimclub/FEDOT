from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from fedot.api.api_utils.assumptions.memory_safety import bounded_population_size, memory_safe_operations
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID
from fedot.preprocessing.preprocessing import DataPreprocessor
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import (
    LabelEncodingImplementation, OneHotEncodingImplementation,
)


def _classification_data(n_rows=20, n_columns=4, n_classes=4):
    return InputData(idx=np.arange(n_rows), features=np.ones((n_rows, n_columns), dtype=np.float32),
                     target=np.arange(n_rows)[:, None] % n_classes,
                     task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)


def test_large_multiclass_restricts_model_candidates_but_keeps_other_operations(monkeypatch):
    from fedot.api.api_utils.assumptions import memory_safety

    monkeypatch.setattr(memory_safety, 'MAX_MULTICLASS_TREE_WORK', 100)
    data = _classification_data()
    choices = ['scaling', 'lgbm', 'catboost', 'xgboost', 'rf', 'knn', 'logit']
    assert memory_safe_operations(data, choices) == ['scaling', 'lgbm', 'logit']
    assert choices == ['scaling', 'lgbm', 'catboost', 'xgboost', 'rf', 'knn', 'logit']


def test_population_adapts_to_measured_fit_without_increasing_budget():
    assert bounded_population_size(20, n_jobs=8, composing_seconds=1944, estimated_fit_seconds=407) == 10
    assert bounded_population_size(6, n_jobs=8, composing_seconds=1944, estimated_fit_seconds=407) == 6
    assert bounded_population_size(20, n_jobs=8, composing_seconds=1944, estimated_fit_seconds=0) == 20


@pytest.mark.parametrize('n_classes,operations', [(2, ['lgbm', 'catboost']), (4, ['catboost'])])
def test_memory_safety_preserves_small_or_explicitly_restricted_search(monkeypatch, n_classes, operations):
    from fedot.api.api_utils.assumptions import memory_safety

    monkeypatch.setattr(memory_safety, 'MAX_MULTICLASS_TREE_WORK', 100)
    assert memory_safe_operations(_classification_data(n_classes=n_classes), operations) is operations


@pytest.mark.parametrize('max_arity', [None, 3])
def test_composer_keeps_best_quality_search_and_tuning_for_large_multiclass(monkeypatch, max_arity):
    from fedot.api.api_utils.assumptions import memory_safety
    from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
    from fedot.api.api_utils.api_composer import ApiComposer
    from fedot.api.api_utils.params import ApiParams
    from fedot.api.time import ApiTime

    monkeypatch.setattr(memory_safety, 'MAX_MULTICLASS_TREE_WORK', 100)

    @contextmanager
    def measured_initial_fit(timer, n_folds):
        yield
        timer.assumption_fit_spend_time_single_fold = timedelta(seconds=81.4)
        timer.assumption_fit_spend_time = timedelta(seconds=407)

    monkeypatch.setattr(ApiTime, 'launch_assumption_fit', measured_initial_fit)
    pipeline = Mock()
    pipeline.root_node.operation.operation_type = 'lgbm'
    monkeypatch.setattr(AssumptionsHandler, 'propose_assumptions',
                        lambda self, initial_assumption, available_operations, **kwargs: [pipeline])
    monkeypatch.setattr(AssumptionsHandler, 'fit_assumption_and_check_correctness',
                        lambda self, pipeline, **kwargs: pipeline)
    params = ApiParams({'preset': 'best_quality', 'with_tuning': True, 'max_arity': max_arity,
                        'available_operations': ['scaling', 'lgbm', 'catboost', 'knn', 'logit'],
                        'use_operations_cache': False, 'use_preprocessing_cache': False,
                        'use_predictions_cache': False}, 'classification', timeout=54, n_jobs=8)
    composer = ApiComposer(params, ['roc_auc'])
    composer.timer = ApiTime(time_for_automl=54, with_tuning=True)
    assumptions, _ = composer.propose_and_fit_initial_assumption(_classification_data())

    assert assumptions[0].root_node.operation.operation_type == 'lgbm'
    assert params['max_arity'] == 1
    assert params['pop_size'] == 10
    assert composer.timer.have_time_for_composing(params['pop_size'], params.n_jobs)
    assert params['with_tuning'] is True
    assert params['preset'] == 'best_quality'
    assert 'catboost' not in params['available_operations']


def _categorical_data():
    data = _classification_data()
    data.features = np.array([[str(i), float(i)] for i in range(20)], dtype=object)
    data.categorical_idx = np.array([0])
    data.numerical_idx = np.array([1])
    data.supplementary_data.col_type_ids = {'features': np.array([TYPE_TO_ID[str], TYPE_TO_ID[float]]),
                                            'target': np.array([TYPE_TO_ID[int]])}
    return data


def test_auto_encoder_bounds_dense_allocation_and_reuses_fitted_encoder(monkeypatch):
    from fedot.preprocessing import preprocessing

    monkeypatch.setattr(preprocessing, 'MAX_AUTO_ONE_HOT_BYTES', 200)
    preprocessor = DataPreprocessor()
    data = _categorical_data()
    train = preprocessor._apply_categorical_encoding(data, 'table')
    assert isinstance(preprocessor.features_encoders['table'], LabelEncodingImplementation)
    assert train.features.shape == (20, 2)
    assert train.features.dtype == np.float32

    test = preprocessor._apply_categorical_encoding(_categorical_data(), 'table')
    np.testing.assert_array_equal(test.features, train.features)


def test_auto_encoder_keeps_float32_one_hot_for_small_tables(monkeypatch):
    from fedot.preprocessing import preprocessing

    monkeypatch.setattr(preprocessing, 'MAX_AUTO_ONE_HOT_BYTES', 2000)
    preprocessor = DataPreprocessor()
    data = preprocessor._apply_categorical_encoding(_categorical_data(), 'table')
    assert isinstance(preprocessor.features_encoders['table'], OneHotEncodingImplementation)
    assert data.features.shape == (20, 21)
    assert data.features.dtype == np.float32
