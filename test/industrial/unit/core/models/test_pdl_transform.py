import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from unittest.mock import MagicMock, patch

from fedot_ind.core.models.pdl.pairwise_transform import PDCDataTransformer, SampleWeights
from fedot.core.operations.operation_parameters import OperationParameters


class TestPDCDataTransformer:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe with mixed types for testing."""
        data = {
            'numeric_col1': [1.5, 2.7, 3.2, 4.1, 5.0],
            'numeric_col2': [10, 20, 30, 40, 50],
            'ordinal_col': pd.Categorical(['low', 'medium', 'high', 'medium', 'low'], ordered=True),
            'string_col1': ['A', 'B', 'C', 'D', 'E'],
            'string_col2': pd.Categorical(['cat', 'dog', 'cat', 'bird', 'dog'], ordered=False),
            'bool_col': [True, False, True, False, True]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def y_numeric(self):
        """Sample numeric target variable."""
        return pd.Series([5.1, 6.2, 7.3, 8.4, 9.5], name='target')

    @pytest.fixture
    def y_categorical(self):
        """Sample categorical target variable."""
        return pd.Series(['X', 'Y', 'Z', 'X', 'Y'], name='target')

    def test_init_default(self):
        """Test initialization with default parameters."""
        transformer = PDCDataTransformer()
        assert transformer.numeric_features is None
        assert transformer.ordinal_features is None
        assert transformer.string_features is None
        assert transformer.y_type is None

    def test_init_with_params(self):
        """Test initialization with specified parameters."""
        numeric = ['num1', 'num2']
        ordinal = ['ord1']
        string = ['str1', 'str2']

        transformer = PDCDataTransformer(
            numeric_features=numeric,
            ordinal_features=ordinal,
            string_features=string,
            y_type='numeric'
        )

        assert transformer.numeric_features == numeric
        assert transformer.ordinal_features == ordinal
        assert transformer.string_features == string
        assert transformer.y_type == 'numeric'

    def test_init_invalid_y_type(self):
        """Test initialization with invalid y_type parameter."""
        with pytest.raises(ValueError) as excinfo:
            PDCDataTransformer(y_type='invalid_type')
        assert "y_type must be one of 'numeric', 'ordinal', 'string'" in str(excinfo.value)

    def test_fit_auto_feature_detection(self, sample_dataframe):
        """Test automatic feature type detection during fit."""
        transformer = PDCDataTransformer()
        transformer.fit(sample_dataframe)

        assert set(transformer.numeric_features) == {'numeric_col1', 'numeric_col2', 'bool_col'}

    def test_fit_y_transformers(self, sample_dataframe, y_numeric, y_categorical):
        """Test fitting of y preprocessors."""
        # Test numeric target
        transformer_num = PDCDataTransformer(y_type='numeric')
        transformer_num.fit(sample_dataframe, y_numeric)
        assert isinstance(transformer_num.preprocessing_y_, StandardScaler)

        # Test ordinal target
        transformer_ord = PDCDataTransformer(y_type='ordinal')
        transformer_ord.fit(sample_dataframe, y_categorical)
        assert isinstance(transformer_ord.preprocessing_y_, OrdinalEncoder)

        # Test string target
        transformer_str = PDCDataTransformer(y_type='string')
        transformer_str.fit(sample_dataframe, y_categorical)
        assert isinstance(transformer_str.preprocessing_y_, OneHotEncoder)

    def test_cast_uint(self, sample_dataframe, y_numeric):
        """Test the cast_uint method."""
        transformer = PDCDataTransformer()
        X_cast, y_cast = transformer.cast_uint(sample_dataframe, y_numeric)

        # Check numeric columns were converted to float32
        assert X_cast['numeric_col1'].dtype == np.float32
        assert X_cast['numeric_col2'].dtype == np.float32

        # Check y was converted to float32
        assert y_cast.dtype == np.float32

    @patch.object(PDCDataTransformer, 'cast_uint')
    def test_transform(self, mock_cast_uint, sample_dataframe):
        """Test the transform method with mocked prerequisites."""
        # Setup
        transformer = PDCDataTransformer()
        transformer.preprocessing_ = MagicMock()
        mock_transformed_data = pd.DataFrame(np.random.rand(5, 3))
        transformer.preprocessing_.transform.return_value = mock_transformed_data
        mock_cast_uint.return_value = (sample_dataframe, None)

        # Test
        result = transformer.transform(sample_dataframe)

        # Assertions
        mock_cast_uint.assert_called_once_with(sample_dataframe)
        transformer.preprocessing_.transform.assert_called_once_with(sample_dataframe)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_transform_no_features(self, sample_dataframe):
        """Test transform method when preprocessing returns no features."""
        transformer = PDCDataTransformer()
        transformer.preprocessing_ = MagicMock()
        # Mock transformer returning empty dataframe
        transformer.preprocessing_.transform.return_value = pd.DataFrame()

        # Create a mock to avoid the actual cast_uint implementation
        with patch.object(PDCDataTransformer, 'cast_uint', return_value=(sample_dataframe, None)):
            with pytest.raises(ValueError) as excinfo:
                transformer.transform(sample_dataframe)
            assert 'no features left after pre-processing' in str(excinfo.value)

    def test_transform_sparse_matrix(self, sample_dataframe):
        """Test transform method rejects sparse matrix data."""
        transformer = PDCDataTransformer()
        transformer.preprocessing_ = MagicMock()

        # Create a sparse matrix
        sparse_data = csr_matrix(np.eye(5))

        # Mock transformer returning DataFrame with sparse data
        df_with_sparse = pd.DataFrame({'col1': [sparse_data[0]] * 5})
        transformer.preprocessing_.transform.return_value = df_with_sparse

        # Create a mock to avoid the actual cast_uint implementation
        with patch.object(PDCDataTransformer, 'cast_uint', return_value=(sample_dataframe, None)):
            with pytest.raises(NotImplementedError) as excinfo:
                transformer.transform(sample_dataframe)
            assert 'X contains sparse features' in str(excinfo.value)


class TestSampleWeights:
    @pytest.fixture
    def sample_params(self):
        """Create sample parameters for SampleWeights."""
        params = OperationParameters(**{'method': 'L2'})
        return params

    @pytest.fixture
    def sample_weights_instance(self, sample_params):
        """Create SampleWeights instance with configured parameters."""
        return SampleWeights(params=sample_params)

    @pytest.fixture
    def mock_training_data(self):
        """Create mock training data."""
        X_train = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        return X_train

    @pytest.fixture
    def mock_validation_data(self):
        """Create mock validation data."""
        X_val = pd.DataFrame({
            'feature1': [1.5, 2.5, 3.5],
            'feature2': [0.15, 0.25, 0.35]
        })
        y_val = pd.Series([10.0, 20.0, 30.0])
        return X_val, y_val

    def test_init(self, sample_params):
        """Test initialization with parameters."""
        weights = SampleWeights(params=sample_params)
        assert weights.method == 'L2'
        assert 'L2' in weights.method_dict

    def test_init_default(self):
        """Test initialization with default parameters."""
        weights = SampleWeights(dict())
        assert weights.method == 'L2'  # Default method

    def test_normalize_weights(self, sample_weights_instance):
        """Test normalize_weights method."""
        # Test with varied weights
        weights = pd.Series([1.0, 2.0, 3.0, 4.0])
        normalized = sample_weights_instance._normalize_weights(weights)
        assert np.isclose(normalized.sum(), 1.0)
        assert all(normalized >= 0)

        # Test with uniform weights
        uniform_weights = pd.Series([2.0, 2.0, 2.0, 2.0])
        normalized_uniform = sample_weights_instance._normalize_weights(uniform_weights)
        assert np.isclose(normalized_uniform.sum(), 1.0)
        assert all(normalized_uniform == 0.25)

    def test_normalize_weights_negative(self, sample_weights_instance):
        """Test normalize_weights with negative values."""
        weights = pd.Series([1.0, -2.0, 3.0, 4.0])
        with pytest.raises(AssertionError):
            sample_weights_instance._normalize_weights(weights)

    @patch.object(SampleWeights, '_sample_weight_by_kmeans_prototypes')
    def test_method_kmeans(self, mock_kmeans):
        """Test KMeansClusterCenters method selection."""
        params = OperationParameters(**{'method': 'KMeansClusterCenters'})
        weights = SampleWeights(params=params)

        # Configure mock
        expected_result = pd.Series([0.2, 0.3, 0.5])
        mock_kmeans.return_value = expected_result

        # Setup attributes needed for method execution
        weights.X_train_ = pd.DataFrame({'f1': [1, 2, 3]})

        # Test method call
        result = weights.method_dict['KMeansClusterCenters']()

        # Assertions
        mock_kmeans.assert_called_once()
        assert result.equals(expected_result)

    @patch.object(SampleWeights, '_sample_weight_optimize')
    def test_method_l2(self, mock_optimize):
        """Test L2 method selection and configuration."""
        params = OperationParameters(**{'method': 'L2'})
        weights = SampleWeights(params=params)

        # Test partial function configuration
        partial_func = weights.method_dict['L2']
        assert partial_func.func == weights._sample_weight_optimize
        assert partial_func.keywords == {'l2_lambda': 0.1}

    @patch('fedot_ind.core.models.pdl.pairwise_transform.minimize')
    def _test_sample_weight_optimize(
            self,
            mock_minimize,
            sample_weights_instance,
            mock_training_data,
            mock_validation_data):
        """Test sample_weight_optimize method."""
        # Setup
        X_val, y_val = mock_validation_data
        sample_weights_instance.X_train_ = mock_training_data

        # Configure the _predict_samples mock
        with patch.object(
            SampleWeights,
            '_predict_samples',
            return_value=(pd.DataFrame(np.random.rand(3, 5)), None)
        ):
            # Configure minimize mock
            mock_result = MagicMock()
            mock_result.x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            mock_minimize.return_value = mock_result

            # Test
            result = sample_weights_instance._sample_weight_optimize(X_val, y_val)

            # Assertions
            assert mock_minimize.called
            assert isinstance(result, pd.Series)
            assert len(result) == len(mock_training_data)
            assert np.isclose(result.sum(), 1.0)

    def _test_sample_weight_ordered_votes_from_weights(self, sample_weights_instance):
        """Test _sample_weight_ordered_votes_from_weights static method."""
        received_weights = np.array([0.1, 0.3, 0.2, 0.4])
        result = SampleWeights._sample_weight_ordered_votes_from_weights(received_weights)

        # Should assign weights based on rank (higher values of received_weights get lower ranks)
        expected = np.array([4, 2, 3, 1]) / 10  # Ranks converted to weights
        assert np.allclose(result, expected)

    @patch.object(SampleWeights, '_sample_weight_negative_error')
    def test_sample_weight_ordered_votes(self, mock_neg_error, sample_weights_instance, mock_validation_data):
        """Test _sample_weight_ordered_votes method."""
        X_val, y_val = mock_validation_data

        # Configure mock
        mock_neg_error.return_value = pd.Series([0.1, 0.3, 0.2, 0.4, 0.0])

        # Mock the static method
        with patch.object(
            SampleWeights,
            '_sample_weight_ordered_votes_from_weights',
            return_value=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        ) as mock_ordered:
            # Test
            result = sample_weights_instance._sample_weight_ordered_votes(X_val, y_val)

            # Assertions
            mock_neg_error.assert_called_once_with(X_val, y_val, force_symmetry=True)
            mock_ordered.assert_called_once_with(mock_neg_error.return_value)
            assert np.array_equal(result, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
