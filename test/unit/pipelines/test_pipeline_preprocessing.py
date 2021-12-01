import numpy as np

from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum, Task
from test.unit.data_operations.test_data_operations_implementations import get_mixed_data
from test.unit.test_data_preprocessing import data_with_only_categorical_features, data_with_too_much_nans, \
    data_with_spaces_and_nans_in_features, data_with_nans_in_target_column, data_with_nans_in_multi_target, \
    data_with_categorical_target


def test_only_categorical_data_process_correctly():
    """
    Check if data with only categorical features processed correctly
    Source 3-feature categorical dataset must be transformed into 5-feature
    """
    pipeline = Pipeline(PrimaryNode('ridge'))
    categorical_data = data_with_only_categorical_features()

    pipeline.fit(categorical_data)
    fitted_ridge = pipeline.nodes[0]
    coefficients = fitted_ridge.operation.fitted_operation.coef_
    coefficients_shape = coefficients.shape
    assert 5 == coefficients_shape[1]


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
    data_with_spaces = data_with_spaces_and_nans_in_features()

    pipeline.fit(data_with_spaces)
    coefficients = pipeline.nodes[0].operation.fitted_operation.coef_
    coefficients_shape = coefficients.shape

    assert 2 == coefficients_shape[1]


def test_data_with_nans_in_target_process_correctly():
    """ K-nn model should use 4 samples to train instead of 6 source due to
    two rows will be removed. So, when n_neighbors was corrected, value must be
    replaced using the following: new value = round(4 / 2). So, 2 instead of 3.

    The same test for multi target table.
    """

    knn_node = PrimaryNode('knnreg')
    knn_node.custom_params = {'n_neighbors': 10}
    pipeline = Pipeline(knn_node)

    # Single target column processing
    single_target_data = data_with_nans_in_target_column()
    pipeline.fit(single_target_data)
    single_hyperparams = pipeline.nodes[0].custom_params

    # Multi-target columns processing
    multi_target_data = data_with_nans_in_multi_target()
    pipeline.fit(multi_target_data)
    multi_hyperparams = pipeline.nodes[0].custom_params

    assert 2 == single_hyperparams['n_neighbors']
    assert 2 == multi_hyperparams['n_neighbors']


def test_preprocessing_binary_categorical_train_test_correct():
    """ Generate tabular InputData with categorical features with only two values (binary).
    During preprocessing such a features must be converted into int values. The same mapping
    should be performed on test part.

    The dataset used below has an important property. The first feature in train will always
    be binary (a + b, or a + c, etc.), but a new category (a or c pr b) will appear in the test.
    So it is needed to extend dictionary for Label encoder.
    """
    pipeline = Pipeline(PrimaryNode('ridge'))
    categorical_data = data_with_only_categorical_features()
    train_data, test_data = train_test_data_setup(categorical_data)

    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)
    assert prediction is not None


def test_pipeline_with_imputer():
    """
    Check the correctness of pipeline fitting on data with gaps and categorical features.
    Pipeline has only imputation operation in it's structure. So encoding must be performed
    as preprocessing.
    """
    imputation_node = PrimaryNode('simple_imputation')
    final_node = SecondaryNode('ridge', nodes_from=[imputation_node])
    pipeline = Pipeline(final_node)

    mixed_input = get_mixed_data(task=Task(TaskTypesEnum.regression),
                                 extended=True)
    pipeline.fit(mixed_input)

    # Coefficients for ridge regression
    coefficients = pipeline.nodes[0].operation.fitted_operation.coef_
    # Linear must use 17 features - several of them are encoded ones
    assert 17 == coefficients.shape[1]


def test_pipeline_with_encoder():
    """
    Check the correctness of pipeline fitting on data with gaps and categorical features.
    Pipeline has only encoding operation in it's structure. So imputation must be performed
    as preprocessing.
    """
    encoding_node = PrimaryNode('one_hot_encoding')
    final_node = SecondaryNode('knnreg', nodes_from=[encoding_node])
    final_node.custom_params = {'n_neighbors': 20}
    pipeline = Pipeline(final_node)

    mixed_input = get_mixed_data(task=Task(TaskTypesEnum.regression),
                                 extended=True)
    # Train pipeline with knn model and then check
    pipeline.fit(mixed_input)
    knn_params = pipeline.nodes[0].custom_params

    # The number of neighbors must be equal to half of the objects in the table.
    # This means that the row with nan has been adequately processed
    assert 3 == knn_params['n_neighbors']


def test_pipeline_target_encoding_correct():
    """
    The correct processing of the categorical target at the Pipeline
    is tested. Moreover, target contains nans and has incorrect shape.
    Source and predicted labels should not differ.
    """
    classification_data = data_with_categorical_target(with_nan=True)

    pipeline = Pipeline(PrimaryNode('dt'))
    pipeline.fit(classification_data)
    predicted = pipeline.predict(classification_data, output_mode='labels')
    predicted_labels = predicted.predict

    assert predicted_labels[0] == 'blue'
    assert predicted_labels[-1] == 'di'


def test_pipeline_target_encoding_for_probs():
    """
    Pipeline's ability to correctly make predictions when probabilities return
    is being tested for categorical target
    """
    classification_data = data_with_categorical_target(with_nan=False)

    pipeline = Pipeline(PrimaryNode('dt'))
    pipeline.fit(classification_data)
    predicted = pipeline.predict(classification_data, output_mode='probs')
    predicted_probs = predicted.predict

    # label 'blue' - after LabelEncoder 1
    assert np.isclose(predicted_probs[0, 1], 1)
    # label 'da' - after LabelEncoder 2
    assert np.isclose(predicted_probs[1, 2], 1)
    # label 'ba' - after LabelEncoder 0
    assert np.isclose(predicted_probs[2, 0], 1)
    # label 'di' - after LabelEncoder 3
    assert np.isclose(predicted_probs[3, 3], 1)
