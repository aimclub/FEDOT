from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.preprocessing.preprocessing import DataPreprocessor
from fedot.preprocessing.structure import StructureExplorer
from test.unit.test_data_preprocessing import data_with_only_categorical_features, data_with_too_much_nans, \
    data_with_spaces_and_nans_in_features, data_with_nans_in_target_column, data_with_nans_in_multi_target

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


def test_non_correct_pipeline_encoder_imputer_validation():
    """
    DataPreprocessor should correctly identify is pipeline has needed operations
    (encoding, imputation) in right order or not.
    In the case presented below incorrect unimodal pipeline generated.
    """
    first_imputation = PrimaryNode('simple_imputation')
    first_encoder = PrimaryNode('one_hot_encoding')

    second_rfr = SecondaryNode('rfr', nodes_from=[first_imputation])
    second_ridge = SecondaryNode('ridge', nodes_from=[first_encoder])
    second_encoder = SecondaryNode('one_hot_encoding', nodes_from=[first_imputation])

    third_imputer = SecondaryNode('simple_imputation', nodes_from=[second_ridge])
    root = SecondaryNode('linear', nodes_from=[second_rfr, third_imputer, second_encoder])
    pipeline = Pipeline(root)

    encoding_correct = StructureExplorer().check_structure_by_tag(pipeline, tag_to_check='encoding')
    imputer_correct = StructureExplorer().check_structure_by_tag(pipeline, tag_to_check='imputation')

    assert encoding_correct is False
    assert imputer_correct is False


def test_correct_pipeline_encoder_imputer_validation():
    """ Validate correct pipeline """
    first_source = PrimaryNode('data_source_table')
    second_imputer = SecondaryNode('simple_imputation', nodes_from=[first_source])
    third_encoder = SecondaryNode('one_hot_encoding', nodes_from=[second_imputer])
    fourth_imputer = SecondaryNode('simple_imputation', nodes_from=[third_encoder])
    root = SecondaryNode('linear', nodes_from=[fourth_imputer])
    pipeline = Pipeline(root)

    encoding_correct = StructureExplorer().check_structure_by_tag(pipeline, tag_to_check='encoding')
    imputer_correct = StructureExplorer().check_structure_by_tag(pipeline, tag_to_check='imputation')

    assert encoding_correct is True
    assert imputer_correct is True


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
