import pickle
import numpy as np
from sklearn.datasets import make_classification, make_regression

from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.blending import (
    BlendingClassifier, BlendingRegressor)


# ================== Test Data Preparation ========================

def get_binclass_data():
    """Generate binary classification data"""
    X, y = make_classification(n_samples=100, n_features=5,
                               n_classes=2, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification))


def get_multiclass_data():
    """Generate multiclass classification data"""
    X, y = make_classification(n_samples=100, n_features=5,
                               n_classes=3, n_informative=3, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification))


def get_regression_data():
    """Generate regression data"""
    X, y = make_regression(n_samples=100, n_features=5,
                           n_informative=3, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.regression))

# ================== Ensemble with blending ========================


def test_blending_classifier_integration():
    """Test for BlendingClassifier via API"""
    input_data = get_multiclass_data()
    train, test = train_test_data_setup(input_data)

    model = Fedot(
        problem='classification',
        timeout=1,
        with_tuning=False,
        metric=['f1'],
        initial_assumption=PipelineBuilder()
        .add_branch('logit', 'rf', 'dt')
        .join_branches('blending')
        .build()
    )
    model.fit(train)

    # Checking predictions
    predictions = model.predict(test)
    assert len(predictions) == len(test.target)

    # Checking metrics
    score = model.get_metrics(test.target)
    assert score['f1'] > 0.5  # better than constant predictor


def test_blending_regressor_integration():
    """Test for BlendingRegressor via API"""
    input_data = get_regression_data()
    train, test = train_test_data_setup(input_data)

    model = Fedot(
        problem='regression',
        timeout=1,
        with_tuning=False,
        metric=['r2'],
        initial_assumption=PipelineBuilder()
        .add_branch('ridge', 'rfr', 'dtreg')
        .join_branches('blendreg')
        .build()
    )
    model.fit(train)

    # Checking predictions
    predictions = model.predict(test)
    assert len(predictions) == len(test.target)

    # Checking metrics
    score = model.get_metrics(test.target)
    assert score['r2'] > 0  # better than constant predictor


def test_blending_with_single_model_fallback():
    """Single-model case processing test in blending"""
    input_data = get_multiclass_data()

    # Pipeline with single branch
    pipeline = PipelineBuilder().add_node('logit').add_node('blending').build()

    pipeline.fit(input_data)
    blending_operation = pipeline.root_node.fitted_operation
    assert isinstance(blending_operation, BlendingClassifier)
    assert len(blending_operation.weights) == 1
    assert np.isclose(sum(blending_operation.weights), 1.0)


def test_blendreg_with_single_model_fallback():
    """Single-model case processing test in blending"""
    input_data = get_regression_data()

    # Pipeline with single branch
    pipeline = PipelineBuilder().add_node('linear').add_node('blendreg').build()

    pipeline.fit(input_data)
    blending_operation = pipeline.root_node.fitted_operation
    assert isinstance(blending_operation, BlendingRegressor)
    assert len(blending_operation.weights) == 1
    assert np.isclose(sum(blending_operation.weights), 1.0)


def test_blending_with_custom_parameters():
    """Test of passing custom parameters to blending"""
    input_data = get_binclass_data()

    pipeline = (PipelineBuilder()
                .add_branch('logit', 'rf')
                .join_branches('blending', params={'n_trials': 50}).build())

    pipeline.fit(input_data)
    blending_operation = pipeline.root_node.fitted_operation
    assert blending_operation.n_trials == 50
    assert isinstance(blending_operation, BlendingClassifier)
    assert len(blending_operation.weights) == 2
    assert np.isclose(sum(blending_operation.weights), 1.0)\



def test_blendreg_with_custom_parameters():
    """Test of passing custom parameters to blending"""
    input_data = get_regression_data()

    pipeline = (PipelineBuilder()
                .add_branch('linear', 'rfr')
                .join_branches('blendreg', params={'n_trials': 50}).build())

    pipeline.fit(input_data)
    blending_operation = pipeline.root_node.fitted_operation
    assert blending_operation.n_trials == 50
    assert isinstance(blending_operation, BlendingRegressor)
    assert len(blending_operation.weights) == 2
    assert np.isclose(sum(blending_operation.weights), 1.0)


def test_blending_are_serializable():
    input_data = get_binclass_data()
    pipeline = PipelineBuilder().add_node('logit').add_node('blending').build()
    pipeline.fit(input_data)
    serialized = pickle.dumps(pipeline, pickle.HIGHEST_PROTOCOL)
    assert isinstance(serialized, bytes)


def test_blendreg_are_serializable():
    input_data = get_regression_data()
    pipeline = PipelineBuilder().add_node('linear').add_node('blendreg').build()
    pipeline.fit(input_data)
    serialized = pickle.dumps(pipeline, pickle.HIGHEST_PROTOCOL)
    assert isinstance(serialized, bytes)
