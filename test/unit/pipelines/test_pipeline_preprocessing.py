from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from test.unit.test_data_preprocessing import data_with_only_categorical_features, data_with_too_much_nans, \
    data_with_leading_trailing_spaces, data_with_nans_in_target_column, data_with_nans_in_multi_target

# Tests for pipeline fitting correctly with "bad" data as input - checking preprocessing


def test_only_categorical_data_process_correctly():
    """ Check if data with only categorical features processed correctly """
    pipeline = Pipeline(PrimaryNode('ridge'))
    categorical_data = data_with_only_categorical_features()

    pipeline.fit(categorical_data)


def test_nans_columns_process_correctly():
    """ Check if data with nans processed correctly. Columns with nans should be ignored """
    pipeline = Pipeline(PrimaryNode('ridge'))
    data_with_nans = data_with_too_much_nans()

    pipeline.fit(data_with_nans)

    # Ridge should use only one feature to make prediction
    fitted_ridge = pipeline.nodes[0]
    coefficients = fitted_ridge.operation.fitted_operation.coef_
    coefficients_shape = coefficients.shape

    assert 1 == coefficients_shape[1]


def test_spaces_columns_process_correctly():
    """ Train simple pipeline on the dataset with spaces in categorical features.
    For example, ' x ' instead of 'x'.
    """
    pipeline = Pipeline(PrimaryNode('ridge'))
    data_with_spaces = data_with_leading_trailing_spaces()

    pipeline.fit(data_with_spaces)
    coefficients = pipeline.nodes[0].operation.fitted_operation.coef_
    coefficients_shape = coefficients.shape

    assert 2 == coefficients_shape[1]


def test_data_with_nans_in_target_process_correctly():
    """ K-nn model should use 5 samples to train instead of 6 source due to
    one row will be removed.
    """

    pipeline = Pipeline(PrimaryNode('knnreg'))

    # Single target column processing
    single_target_data = data_with_nans_in_target_column()
    pipeline.fit(single_target_data)
    single_hyperparams = pipeline.nodes[0].custom_params

    # Multi-target columns processing
    multi_target_data = data_with_nans_in_multi_target()
    pipeline.fit(multi_target_data)
    multi_hyperparams = pipeline.nodes[0].custom_params

    assert 5 == single_hyperparams['n_samples']
    assert 3 == multi_hyperparams['n_samples']
