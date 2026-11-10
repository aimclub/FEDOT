import numpy as np
import pytest
from sklearn.datasets import make_classification

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import (
    FedotLightGBMClassificationImplementation,
    FedotLightGBMImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def classification_data(classes):
    features, target = make_classification(n_samples=120, n_features=8, n_informative=5,
                                           n_classes=classes, n_clusters_per_class=1, random_state=42)
    return InputData(idx=np.arange(len(target)), features=features, target=target,
                     task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)


@pytest.mark.parametrize('classes, expected', [(2, 1e-3), (3, 1.0), (8, 1.0)])
def test_default_leaf_hessian_depends_on_task(monkeypatch, classes, expected):
    implementation = FedotLightGBMClassificationImplementation(OperationParameters(use_eval_set=False))
    monkeypatch.setattr(FedotLightGBMImplementation, 'fit', lambda self, input_data: self.model)

    model = implementation.fit(classification_data(classes))

    assert model.get_params()['min_child_weight'] == expected


@pytest.mark.parametrize('parameter', ['min_child_weight', 'min_sum_hessian_in_leaf',
                                     'min_sum_hessian', 'min_hessian', 'min_sum_hessian_per_leaf'])
def test_explicit_leaf_hessian_is_preserved(monkeypatch, parameter):
    implementation = FedotLightGBMClassificationImplementation(OperationParameters(**{parameter: 0.2}))
    original_params = implementation.model.get_params()
    monkeypatch.setattr(FedotLightGBMImplementation, 'fit', lambda self, input_data: self.model)

    implementation.fit(classification_data(3))

    assert implementation.model.get_params() == original_params


def test_repeated_fit_resets_implicit_multiclass_default(monkeypatch):
    implementation = FedotLightGBMClassificationImplementation(OperationParameters())
    monkeypatch.setattr(FedotLightGBMImplementation, 'fit', lambda self, input_data: self.model)
    implementation.fit(classification_data(3))

    implementation.fit(classification_data(2))

    assert implementation.model.get_params()['min_child_weight'] == 1e-3


def test_multioutput_keeps_library_default(monkeypatch):
    data = classification_data(3)
    data.target = np.column_stack([data.target, data.target])
    implementation = FedotLightGBMClassificationImplementation(OperationParameters())
    monkeypatch.setattr(FedotLightGBMImplementation, 'fit', lambda self, input_data: self.model)

    implementation.fit(data)

    assert implementation.model.get_params()['min_child_weight'] == 1e-3


def test_multiclass_model_produces_finite_normalized_probabilities():
    data = classification_data(3)
    implementation = FedotLightGBMClassificationImplementation(
        OperationParameters(n_estimators=5, n_jobs=1, verbose=-1, use_eval_set=False))

    implementation.fit(data)
    probabilities = implementation.predict_proba(data)

    assert probabilities.shape == (len(data.target), 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
