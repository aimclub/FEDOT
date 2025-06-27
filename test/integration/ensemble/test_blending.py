import pickle
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.models.blending import (
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

# ================== Tests ========================


def test_blending_implementation_no_models_raises_error():
    """Check that an error is raised if there are no models"""
    blending = BlendingClassifier()
    input_data = get_multiclass_data()
    with pytest.raises(ValueError, match="No previous models provided for blending"):
        blending.fit(input_data)


def test_blendreg_implementation_no_models_raises_error():
    """Check that an error is raised if there are no models"""
    blending = BlendingRegressor()
    input_data = get_regression_data()
    with pytest.raises(ValueError, match="No previous models provided for blending"):
        blending.fit(input_data)


def test_blending_implementation_single_model_uses_weight_one():
    """Check that one model uses a weight of 1.0"""
    blending = BlendingClassifier()
    input_data = InputData(idx=np.array([1]), features=np.array([[0.5]]), target=np.array([1]),
                           data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))
    input_data.supplementary_data.previous_operations = ['model1']
    blending.fit(input_data)
    assert np.allclose(blending.weights, [1.0])


def test_blending_classifier_init_task_specific_params():
    """Check initialization of parameters for classification"""
    classifier = BlendingClassifier()
    input_data = get_binclass_data()
    classifier._init_task_specific_params(input_data)
    assert classifier.task == TaskTypesEnum.classification
    assert classifier.n_classes == 2


def test_blending_classifier_divide_predictions_binary():
    """Check the separation of predictions for binary classification"""
    classifier = BlendingClassifier()
    classifier.n_models = 2
    classifier.n_classes = 2
    input_data = InputData(idx=np.arange(0, 2), features=np.array([[0.1, 0.9]]), target=np.array([1]),
                           data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))
    input_data.supplementary_data.previous_operations = ['m1', 'm2']
    predictions = classifier._divide_predictions(input_data)
    assert len(predictions) == 2
    assert np.allclose(predictions[0], [[0.1]])
    assert np.allclose(predictions[1], [[0.9]])


def test_blending_classifier_divide_predictions_multiclass():
    """Check the separation of predictions for multiclass classification"""
    classifier = BlendingClassifier()
    classifier.n_models = 2
    classifier.n_classes = 3
    input_data = InputData(idx=np.arange(0, 2), features=np.array([[0.1, 0.2, 0.7, 0.3, 0.3, 0.4]]),
                           target=np.array([2]),
                           data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))
    input_data.supplementary_data.previous_operations = ['m1', 'm2']
    predictions = classifier._divide_predictions(input_data)
    assert len(predictions) == 2
    assert np.allclose(predictions[0], [[0.1, 0.2, 0.7]])
    assert np.allclose(predictions[1], [[0.3, 0.3, 0.4]])


def test_blending_classifier_blend_predictions_binary():
    """Check blending of predictions for binary classification"""
    classifier = BlendingClassifier()
    classifier.n_classes = 2
    predictions = [np.array([[0.1]]), np.array([[0.9]])]
    weights = [0.4, 0.6]
    blended = classifier._blend_predictions(predictions, weights)
    assert blended[0] == 1  # 0.4*0.1 + 0.6*0.9 = 0.58 > 0.5 => class 1


def test_blending_classifier_blend_predictions_multiclass():
    """Check blending of predictions for multiclass classification"""
    classifier = BlendingClassifier()
    classifier.n_classes = 3
    predictions = [np.array([[0.1, 0.2, 0.7]]), np.array([[0.3, 0.3, 0.4]])]
    weights = [0.5, 0.5]
    blended = classifier._blend_predictions(predictions, weights)
    assert blended[0] == 2  # argmax of [0.2, 0.25, 0.55] is 2


def test_blending_regressor_init_task_specific_params():
    """Check initialization of parameters for regression"""
    regressor = BlendingRegressor()
    input_data = get_regression_data()
    regressor._init_task_specific_params(input_data)
    assert regressor.task == TaskTypesEnum.regression


def test_blending_regressor_divide_predictions():
    """Check the separation of predictions for regression"""
    regressor = BlendingRegressor()
    regressor.n_models = 2
    input_data = InputData(idx=np.arange(0, 2), features=np.array([[1.5, 2.5]]), target=np.array([2.0]),
                           data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.regression))
    input_data.supplementary_data.previous_operations = ['m1', 'm2']
    predictions = regressor._divide_predictions(input_data)
    assert len(predictions) == 2
    assert np.allclose(predictions[0], [[1.5]])
    assert np.allclose(predictions[1], [[2.5]])


def test_blending_regressor_blend_predictions():
    """Check blending of predictions for regression"""
    regressor = BlendingRegressor()
    predictions = [np.array([[1.0]]), np.array([[2.0]])]
    weights = [0.3, 0.7]
    blended = regressor._blend_predictions(predictions, weights)
    assert np.allclose(blended, [[1.7]])  # 0.3*1.0 + 0.7*2.0 = 1.7


def test_blending_classifier_shape_mismatch_raises_error():
    """Check the error handling of the form mismatch for classification."""
    classifier = BlendingClassifier()
    classifier.n_models = 2
    classifier.n_classes = 2
    input_data = InputData(
        idx=np.arange(0, 1),
        features=np.array([[0.1]]),  # Expect (n_samples, 2), but it's (1, 1)
        target=np.array([1]),
        data_type=DataTypesEnum.table,
        task=Task(TaskTypesEnum.classification),
    )
    with pytest.raises(ValueError, match="Shape mismatch for binary classification"):
        classifier._divide_predictions(input_data)


def test_blending_regressor_shape_mismatch_raises_error():
    """Check the error handling of the form mismatch for regression"""
    regressor = BlendingRegressor()
    regressor.n_models = 2
    input_data = InputData(idx=np.arange(0, 1), features=np.array([[1.5]]),  # only 1 row instead of 2
                           target=np.array([2.0]),
                           data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.regression))
    with pytest.raises(ValueError, match="Shape mismatch for regression"):
        regressor._divide_predictions(input_data)


def test_blending_classifier_integration():
    """Test for BlendingClassifier via API as initial assumption"""
    input_data = get_multiclass_data()
    train, test = train_test_data_setup(input_data)

    model = Fedot(
        problem='classification',
        timeout=0.1,
        with_tuning=True,
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
    """Test for BlendingRegressor via API as initial assumption"""
    input_data = get_regression_data()
    train, test = train_test_data_setup(input_data)

    model = Fedot(
        problem='regression',
        timeout=0.1,
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
    assert np.isclose(sum(blending_operation.weights), 1.0)


def test_blendreg_with_custom_parameters():
    """Test of passing custom parameters to blending"""
    input_data = get_regression_data()

    pipeline = (PipelineBuilder()
                .add_branch('linear', 'rfr')
                .join_branches('blendreg', params={'strategy': 'weighted'}).build())

    pipeline.fit(input_data)
    blending_operation = pipeline.root_node.fitted_operation
    assert blending_operation.strategy == 'weighted'
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
