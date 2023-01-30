import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.preprocessing.categorical import BinaryCategoricalPreprocessor
from fedot.preprocessing.data_types import TableTypesCorrector
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME
from test.unit.data_operations.test_data_operations_implementations import get_mixed_data
from test.unit.preprocessing.test_preprocessing_through_api import data_with_only_categorical_features, \
    data_with_too_much_nans, data_with_spaces_and_nans_in_features, data_with_nans_in_target_column, \
    data_with_nans_in_multi_target, data_with_categorical_target


def data_with_mixed_types_in_each_column(multi_output: bool = False):
    """
    Generate dataset with columns, which contain several data types (int, float or str).
    Moreover, columns can contain nans and target columns have the same problems also.

    :param multi_output: is there a need to generate multi-output target
    """
    task = Task(TaskTypesEnum.classification)
    features = np.array([[np.nan, '1', np.nan, '6', 'b'],
                         [np.nan, '2', 1, '5', 'a'],
                         [np.nan, np.nan, 2.1, 4.1, 4],
                         [np.nan, '3', 3, 3.5, 3],
                         [np.nan, 8, 4, 2, 2],
                         [np.nan, 8, 'a', 1, 1],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [np.nan, '4', 'b', 0, 0],
                         [np.nan, '5', 1, -1, -1]], dtype=object)
    if multi_output:
        # Multi-label classification problem solved
        target = np.array([['label_1', 2],
                           ['label_1', 3],
                           ['label_0', 4],
                           ['label_0', 5],
                           ['label_0', 5],
                           [1, '6'],
                           [0, '7'],
                           [1, '8'],
                           [0, '9']], dtype=object)
    else:
        target = np.array(['label_1', 'label_1', 'label_0', 'label_0', 'label_0', 1, 0, 1, 0], dtype=object)
    input_data = InputData(idx=np.arange(9), features=features,
                           target=target, task=task, data_type=DataTypesEnum.table,
                           supplementary_data=SupplementaryData())
    return input_data


def correct_preprocessing_params(pipeline, categorical_max_uniques_th: int = None):
    """
    Correct preprocessing classes parameters

    :param pipeline: pipeline without initialized preprocessors
    :param categorical_max_uniques_th: if number of unique values in the column lower, than
    threshold - convert column into categorical feature
    """
    table_corrector = TableTypesCorrector()

    if categorical_max_uniques_th is not None:
        table_corrector.categorical_max_uniques_th = categorical_max_uniques_th
    pipeline.preprocessor.types_correctors.update({DEFAULT_SOURCE_NAME: table_corrector})
    pipeline.preprocessor.binary_categorical_processors.update({DEFAULT_SOURCE_NAME: BinaryCategoricalPreprocessor()})

    return pipeline


def test_only_categorical_data_process_correctly():
    """
    Check if data with only categorical features processed correctly
    Source 3-feature categorical dataset must be transformed into 5-feature
    """
    pipeline = Pipeline(PipelineNode('ridge'))
    categorical_data = data_with_only_categorical_features()

    pipeline.fit(categorical_data)
    fitted_ridge = pipeline.nodes[0]
    coefficients = fitted_ridge.operation.fitted_operation.coef_
    coefficients_shape = coefficients.shape
    assert 5 == coefficients_shape[1]


def test_nans_columns_process_correctly():
    """ Check if data with nans processed correctly. Columns with nans should be ignored """
    pipeline = Pipeline(PipelineNode('ridge'))
    data_with_nans = data_with_too_much_nans()

    pipeline = correct_preprocessing_params(pipeline, categorical_max_uniques_th=5)
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
    pipeline = Pipeline(PipelineNode('ridge'))
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

    knn_node = PipelineNode('knnreg')
    knn_node.parameters = {'n_neighbors': 10}
    pipeline = Pipeline(knn_node)

    # Single target column processing
    single_target_data = data_with_nans_in_target_column()
    pipeline.fit(single_target_data)
    single_hyperparams = pipeline.nodes[0].parameters
    # Multi-target columns processing
    multi_target_data = data_with_nans_in_multi_target()
    pipeline.unfit()
    pipeline.fit(multi_target_data)
    multi_hyperparams = pipeline.nodes[0].parameters
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
    pipeline = Pipeline(PipelineNode('ridge'))
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
    imputation_node = PipelineNode('simple_imputation')
    final_node = PipelineNode('ridge', nodes_from=[imputation_node])
    pipeline = Pipeline(final_node)

    pipeline = correct_preprocessing_params(pipeline, categorical_max_uniques_th=5)

    mixed_input = get_mixed_data(task=Task(TaskTypesEnum.regression),
                                 extended=True)
    pipeline.fit(mixed_input)

    # Coefficients for ridge regression
    coefficients = pipeline.nodes[0].operation.fitted_operation.coef_
    # Linear must use 12 features - several of them are encoded ones
    assert coefficients.shape[1] == 12


def test_pipeline_with_encoder():
    """
    Check the correctness of pipeline fitting on data with gaps and categorical features.
    Pipeline has only encoding operation in it's structure. So imputation must be performed
    as preprocessing.
    """
    encoding_node = PipelineNode('one_hot_encoding')
    final_node = PipelineNode('knnreg', nodes_from=[encoding_node])
    final_node.parameters = {'n_neighbors': 20}
    pipeline = Pipeline(final_node)

    mixed_input = get_mixed_data(task=Task(TaskTypesEnum.regression),
                                 extended=True)
    # Train pipeline with knn model and then check
    pipeline.fit(mixed_input)
    knn_params = pipeline.nodes[0].parameters

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

    pipeline = Pipeline(PipelineNode('dt'))
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

    pipeline = Pipeline(PipelineNode('dt'))
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


def test_data_with_mixed_types_per_column_processed_correctly():
    """
    Check if dataset with columns, which contain several data types can be
    processed correctly.
    """
    input_data = data_with_mixed_types_in_each_column()
    train_data, test_data = train_test_data_setup(input_data, split_ratio=0.9)

    pipeline = Pipeline(PipelineNode('dt'))
    pipeline = correct_preprocessing_params(pipeline, categorical_max_uniques_th=5)
    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data)

    importances = pipeline.nodes[0].operation.fitted_operation.feature_importances_

    # Finally, seven features were used to give a forecast
    assert len(importances) == 7
    # Target must contain 4 labels
    assert predicted.predict.shape[-1] == 4
