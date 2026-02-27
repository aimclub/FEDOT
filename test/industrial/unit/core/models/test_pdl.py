import pytest
import numpy as np
import pandas as pd
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.model_selection import train_test_split
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data

from fedot_ind.core.models.pdl.pairwise_model import (
    PairwiseDifferenceEstimator,
    PairwiseDifferenceClassifier,
    PairwiseDifferenceRegressor
)

# Fixtures for test data


@pytest.fixture
def classification_data():
    X, y = np.random.rand(50, 50), np.random.randint(0, 2, 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create InputData objects
    train_data = init_input_data(X=X_train, y=y_train, task='classification')
    test_data = init_input_data(X=X_test, y=y_test, task='classification')

    return train_data, test_data


@pytest.fixture
def regression_data():
    X, y = np.random.rand(50, 50), np.random.rand(50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create InputData objects
    train_data = init_input_data(X=X_train, y=y_train, task='regression')
    test_data = init_input_data(X=X_test, y=y_test, task='regression')

    return train_data, test_data

# Tests for PairwiseDifferenceEstimator


class TestPairwiseDifferenceEstimator:

    def test_convert_to_pandas(self):
        pde = PairwiseDifferenceEstimator()
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])

        df1, df2 = pde._convert_to_pandas(arr1, arr2)

        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert df1.shape == arr1.shape
        assert df2.shape == arr2.shape

    def test_pair_input(self):
        pde = PairwiseDifferenceEstimator()
        X1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        X2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

        X_pair, X_pair_sym = pde.pair_input(X1, X2)

        # Test shapes
        assert X_pair.shape[0] == X1.shape[0] * X2.shape[0]  # Cross product
        assert X_pair_sym.shape[0] == X1.shape[0] * X2.shape[0]

        # Test column names
        expected_columns = ['a_x', 'b_x', 'a_y', 'b_y', 'a_diff', 'b_diff']
        assert all(col in X_pair.columns for col in expected_columns)

    def test_pair_output(self):
        pde = PairwiseDifferenceEstimator()
        y1 = pd.Series([1, 2])
        y2 = pd.Series([3, 4])

        y_pair_diff = pde.pair_output(y1, y2)

        # Test shape
        assert len(y_pair_diff) == len(y1) * len(y2)

        # Test values (should be y2 - y1 for all combinations)
        expected = np.array([2, 3, 1, 2])  # [3-1, 4-1, 3-2, 4-2]
        np.testing.assert_array_equal(y_pair_diff, expected)

    def test_pair_output_difference(self):
        pde = PairwiseDifferenceEstimator()
        y1 = pd.Series([0, 1, 2])
        y2 = pd.Series([0, 2])

        y_pair_diff = pde.pair_output_difference(y1, y2, 3)

        # Test shape
        assert len(y_pair_diff) == len(y1) * len(y2)

        # Test values (should be 1 if different, 0 if same)
        expected = np.array([0, 1, 1, 1, 1, 0])  # Comparing [0,0], [0,2], [1,0], [1,2], [2,0], [2,2]
        np.testing.assert_array_equal(y_pair_diff, expected)

# Tests for PairwiseDifferenceClassifier


class TestPairwiseDifferenceClassifier:

    def test_init(self):
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        assert classifier.base_model_params == {'n_estimators': 10}
        assert hasattr(classifier, 'base_model')
        assert hasattr(classifier, 'pde')

    def test_fit_predict(self, classification_data):
        train_data, test_data = classification_data
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        # Test fit
        fitted_classifier = classifier.fit(train_data)
        assert fitted_classifier is classifier
        assert hasattr(classifier, 'num_classes')
        assert hasattr(classifier, 'target')
        assert hasattr(classifier, 'classes_')

        # Test predict
        predictions = classifier.predict(test_data, output_mode='labels')
        assert len(predictions) == len(test_data.target)

        # Test that predictions are valid class indices
        assert np.all(predictions >= 0)
        assert np.all(predictions < classifier.num_classes)

    def test_predict_proba(self, classification_data):
        train_data, test_data = classification_data
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        classifier.fit(train_data)

        # Test predict_proba
        proba = classifier.predict_proba(test_data, output_mode='default')

        # Check shape and probability sums
        assert proba.shape[0] == len(test_data.target)
        assert np.allclose(np.sum(proba, axis=1), np.ones(len(test_data.target)), atol=1e-10)

    def test_score_difference(self, classification_data):
        train_data, test_data = classification_data
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        classifier.fit(train_data)

        # Test score_difference
        score = classifier.score_difference(test_data)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # MAE value should be between 0 and 1

# Tests for PairwiseDifferenceRegressor


class TestPairwiseDifferenceRegressor:

    def test_init(self):
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        assert regressor.base_model_params == {'n_estimators': 10}
        assert hasattr(regressor, 'base_model')
        assert hasattr(regressor, 'pde')

    def test_fit_predict(self, regression_data):
        train_data, test_data = regression_data
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        # Test fit
        fitted_regressor = regressor.fit(train_data)
        assert fitted_regressor is regressor
        assert hasattr(regressor, 'num_classes')
        assert hasattr(regressor, 'target')

        # Test predict
        predictions = regressor.predict(test_data)
        assert len(predictions) == len(test_data.target)

    def test_predict_samples(self, regression_data):
        train_data, test_data = regression_data
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        regressor.fit(train_data)

        # Test _predict_samples
        prediction_samples, pred_diff_samples = regressor._predict_samples(test_data)

        # Check shapes
        assert prediction_samples.shape == (len(test_data.features), len(train_data.features))
        assert pred_diff_samples.shape == (len(test_data.features), len(train_data.features))

    # Skip testing learn_anchor_weights due to complexity
    # This would require mocking _name_to_method_mapping and detailed implementation knowledge

    def test_set_sample_weight(self, regression_data):
        train_data, _ = regression_data
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        regressor.fit(train_data)

        # Create a mock attribute for testing
        regressor.y_train_ = pd.Series(train_data.target, index=pd.RangeIndex(len(train_data.target)))

        # Test setting valid sample weights
        weights = pd.Series([1] * len(train_data.target), index=regressor.y_train_.index)
        regressor.set_sample_weight(weights)
        assert regressor.sample_weight_ is weights

        # Test with invalid weights (should raise error)
        with pytest.raises(ValueError):
            invalid_weights = pd.Series([1, 2])  # Wrong length
            regressor.set_sample_weight(invalid_weights)
