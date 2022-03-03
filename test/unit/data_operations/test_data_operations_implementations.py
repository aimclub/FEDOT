import os

import numpy as np
from examples.simple.classification.classification_with_tuning import get_classification_dataset
from examples.simple.regression.regression_with_tuning import get_regression_dataset
from examples.simple.time_series_forecasting.gapfilling import generate_synthetic_data
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.operations.evaluation.operation_implementations.data_operations. \
    sklearn_transformations import ImputationImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    CutImplementation, LaggedTransformationImplementation
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.preprocessing.data_types import NAME_CLASS_FLOAT, NAME_CLASS_INT, \
    NAME_CLASS_STR
from test.unit.preprocessing.test_preprocessing_though_api import data_with_only_categorical_features

np.random.seed(2021)


def get_small_regression_dataset():
    """ Function returns features and target for train and test regression models """
    features_options = {'informative': 2, 'bias': 2.0}
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options=features_options,
                                                              samples_amount=70,
                                                              features_amount=4)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_small_classification_dataset():
    """ Function returns features and target for train and test classification models """
    features_options = {'informative': 1, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_train, y_train, x_test, y_test = get_classification_dataset(features_options=features_options,
                                                                  samples_amount=70,
                                                                  features_amount=4,
                                                                  classes_amount=2)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # Define regression task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_time_series():
    """ Function returns time series for time series forecasting task """
    len_forecast = 5
    synthetic_ts = generate_synthetic_data(length=80)

    train_data = synthetic_ts[:-len_forecast]
    test_data = synthetic_ts[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, test_data


def get_multivariate_time_series():
    """ Generate several time series in one InputData block """
    ts_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((-1, 1))
    ts_2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).reshape((-1, 1))
    several_ts = np.hstack((ts_1, ts_2))

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=2))
    train_input = InputData(idx=np.arange(0, len(several_ts)),
                            features=several_ts, target=np.ravel(ts_1),
                            task=task, data_type=DataTypesEnum.ts)
    return train_input


def get_nan_inf_data():
    supp_data = SupplementaryData(column_types={'features': [NAME_CLASS_FLOAT]*4})
    train_input = InputData(idx=[0, 1, 2, 3],
                            features=np.array([[1, 2, 3, 4],
                                               [2, np.nan, 4, 5],
                                               [3, 4, 5, np.inf],
                                               [-np.inf, 5, 6, 7]]),
                            target=np.array([1, 2, 3, 4]),
                            task=Task(TaskTypesEnum.regression),
                            data_type=DataTypesEnum.table,
                            supplementary_data=supp_data)

    return train_input


def get_single_feature_data(task=None):
    supp_data = SupplementaryData(column_types={'features': [NAME_CLASS_INT],
                                                'target': [NAME_CLASS_INT]})
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5],
                            features=np.array([[1], [2], [3], [7], [8], [9]]),
                            target=np.array([[0], [0], [0], [1], [1], [1]]),
                            task=task,
                            data_type=DataTypesEnum.table,
                            supplementary_data=supp_data)

    return train_input


def get_mixed_data(task=None, extended=False):
    """ Generate InputData with five categorical features. The categorical features
    are created in such a way that in any splitting there will be categories in the
    test part that were not in the train.
    """
    if extended:
        features = np.array([[1, '0', '1', 1, '5', 'blue', 'blue'],
                             [2, '1', '0', 0, '4', 'blue', 'da'],
                             [3, '1', '0', 1, '3', 'blue', 'ba'],
                             [np.nan, np.nan, '1', np.nan, '2', 'not blue', 'di'],
                             [8, '1', '1', 0, '1', 'not blue', 'da bu'],
                             [9, '0', '0', 0, '0', 'not blue', 'dai']], dtype=object)
        features_types = [NAME_CLASS_INT, NAME_CLASS_STR, NAME_CLASS_STR, NAME_CLASS_INT,
                          NAME_CLASS_STR, NAME_CLASS_STR, NAME_CLASS_STR]
        supp_data = SupplementaryData(column_types={'features': features_types,
                                                    'target': [NAME_CLASS_INT]})
    else:
        features = np.array([[1, '0', 1],
                             [2, '1', 0],
                             [3, '1', 0],
                             [7, '1', 1],
                             [8, '1', 1],
                             [9, '0', 0]], dtype=object)
        features_types = [NAME_CLASS_INT, NAME_CLASS_STR, NAME_CLASS_INT]
        supp_data = SupplementaryData(column_types={'features': features_types,
                                                    'target': [NAME_CLASS_INT]})

    train_input = InputData(idx=[0, 1, 2, 3, 4, 5],
                            features=features,
                            target=np.array([[0], [0], [0], [1], [1], [1]]),
                            task=task,
                            data_type=DataTypesEnum.table,
                            supplementary_data=supp_data)

    return train_input


def get_nan_binary_data(task=None):
    """ Generate table with two numerical and one categorical features.
    Both them contain nans, which need to be filled in.

    Binary int columns must be processed as "almost categorical". Current dataset
    For example, nan object in [1, nan, 0, 0] must be filled as 0, not as 0.33
    """
    features_types = [NAME_CLASS_INT, NAME_CLASS_STR, NAME_CLASS_INT]
    supp_data = SupplementaryData(column_types={'features': features_types})
    features = np.array([[1, '0', 0],
                         [np.nan, np.nan, np.nan],
                         [0, '2', 1],
                         [1, '1', 1],
                         [5, '1', 1]], dtype=object)

    input_data = InputData(idx=[0, 1, 2, 3], features=features,
                           target=np.array([[0], [0], [1], [1]]),
                           task=task, data_type=DataTypesEnum.table,
                           supplementary_data=supp_data)

    return input_data


def data_with_binary_int_features_and_equal_categories():
    """
    Generate table with binary integer features and nans there. Such a columns
    must be processed as "almost categorical". Current dataset
    For example, nan object in [1, nan, 0, 0] must be filled as 0, not as 0.33
    """
    supp_data = SupplementaryData(column_types={'features': [NAME_CLASS_INT, NAME_CLASS_INT]})
    task = Task(TaskTypesEnum.classification)
    features = np.array([[1, 10],
                         [np.nan, np.nan],
                         [np.nan, np.nan],
                         [0, 0]])
    target = np.array([['not-nan'], ['nan'], ['nan'], ['not-nan']])
    train_input = InputData(idx=[0, 1, 2, 3], features=features, target=target,
                            task=task, data_type=DataTypesEnum.table,
                            supplementary_data=supp_data)

    return train_input


def test_regression_data_operations():
    train_input, predict_input, y_test = get_small_regression_dataset()

    operation_names, _ = OperationTypesRepository('data_operation').suitable_operation(
        task_type=TaskTypesEnum.regression)

    for data_operation in operation_names:
        node_data_operation = PrimaryNode(data_operation)
        node_final = SecondaryNode('linear', nodes_from=[node_data_operation])
        pipeline = Pipeline(node_final)

        # Fit and predict for pipeline
        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(predict_input)
        predicted = predicted_output.predict

        assert len(predicted) == len(y_test)


def test_classification_data_operations():
    train_input, predict_input, y_test = get_small_classification_dataset()

    operation_names, _ = OperationTypesRepository('data_operation').suitable_operation(
        task_type=TaskTypesEnum.classification)

    for data_operation in operation_names:
        node_data_operation = PrimaryNode(data_operation)
        node_final = SecondaryNode('logit', nodes_from=[node_data_operation])
        pipeline = Pipeline(node_final)

        # Fit and predict for pipeline
        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(predict_input)
        predicted = predicted_output.predict

        assert len(predicted) == len(y_test)


def test_ts_forecasting_lagged_data_operation():
    train_input, predict_input, y_test = get_time_series()

    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_ridge)

    pipeline.fit_from_scratch(train_input)
    predicted_output = pipeline.predict(predict_input)
    predicted = np.ravel(predicted_output.predict)

    assert len(predicted) == len(np.ravel(y_test))


def test_ts_forecasting_cut_data_operation():
    train_input, predict_input, y_test = get_time_series()
    horizon = train_input.task.task_params.forecast_length
    operation_cut = CutImplementation(cut_part=0.5)

    transformed_input = operation_cut.transform(train_input, is_fit_pipeline_stage=True)
    assert train_input.idx.shape[0] == 2 * transformed_input.idx.shape[0] - horizon


def test_ts_forecasting_smoothing_data_operation():
    train_input, predict_input, y_test = get_time_series()

    model_names, _ = OperationTypesRepository().operations_with_tag(tags=['smoothing'])

    for smoothing_operation in model_names:
        node_smoothing = PrimaryNode(smoothing_operation)
        node_lagged = SecondaryNode('lagged', nodes_from=[node_smoothing])
        node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
        pipeline = Pipeline(node_ridge)

        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(predict_input)
        predicted = np.ravel(predicted_output.predict)

        assert len(predicted) == len(np.ravel(y_test))


def test_inf_and_nan_absence_after_imputation_implementation_fit_transform():
    input_data = get_nan_inf_data()
    output_data = ImputationImplementation().fit_transform(input_data)

    assert np.sum(np.isinf(output_data.predict)) == 0
    assert np.sum(np.isnan(output_data.predict)) == 0


def test_inf_and_nan_absence_after_imputation_implementation_fit_and_transform():
    input_data = get_nan_inf_data()
    imputer = ImputationImplementation()
    imputer.fit(input_data)
    output_data = imputer.transform(input_data)

    assert np.sum(np.isinf(output_data.predict)) == 0
    assert np.sum(np.isnan(output_data.predict)) == 0


def test_inf_and_nan_absence_after_pipeline_fitting_from_scratch():
    train_input = get_nan_inf_data()

    model_names, _ = OperationTypesRepository().suitable_operation(task_type=TaskTypesEnum.regression)

    for model_name in model_names:
        node_data_operation = PrimaryNode(model_name)
        node_final = SecondaryNode('linear', nodes_from=[node_data_operation])
        pipeline = Pipeline(node_final)

        # Fit and predict for pipeline
        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(train_input)
        predicted = predicted_output.predict

        assert np.sum(np.isinf(predicted)) == 0
        assert np.sum(np.isnan(predicted)) == 0


def test_feature_selection_of_single_features():
    for task_type in [TaskTypesEnum.classification, TaskTypesEnum.regression]:
        model_names, _ = OperationTypesRepository(operation_type='data_operation') \
            .suitable_operation(tags=['feature_selection'], task_type=task_type)

        task = Task(task_type)

        for data_operation in model_names:
            node_data_operation = PrimaryNode(data_operation)

            assert node_data_operation.fitted_operation is None

            # Fit and predict for pipeline
            train_input = get_single_feature_data(task)
            node_data_operation.fit(train_input)
            predicted_output = node_data_operation.predict(train_input)
            predicted = predicted_output.predict

            assert node_data_operation.fitted_operation is not None
            assert predicted.shape == train_input.features.shape


def test_one_hot_encoding_new_category_in_test():
    """ Check if One Hot Encoding can correctly predict data with new categories
    (which algorithm were not process during train stage)
    """
    cat_data = get_mixed_data(task=Task(TaskTypesEnum.classification),
                              extended=True)
    train, test = train_test_data_setup(cat_data)

    # Create pipeline with encoding operation
    one_hot_node = PrimaryNode('one_hot_encoding')
    final_node = SecondaryNode('dt', nodes_from=[one_hot_node])
    pipeline = Pipeline(final_node)

    pipeline.fit(train)
    predicted = pipeline.predict(test)

    assert predicted is not None


def test_knn_with_float_neighbors():
    """
    Check pipeline with k-nn fit and predict correctly if n_neighbors value
    is float value
    """
    node_knn = PrimaryNode('knnreg')
    node_knn.custom_params = {'n_neighbors': 2.5}
    pipeline = Pipeline(node_knn)

    input_data = get_single_feature_data(task=Task(TaskTypesEnum.regression))

    pipeline.fit(input_data)
    pipeline.predict(input_data)


def test_imputation_with_binary_correct():
    """
    Check if SimpleImputer can process mixed data with both numerical and categorical
    features correctly. Moreover, check if the imputer swaps the columns (it shouldn't)
    """
    nan_data = get_nan_binary_data(task=Task(TaskTypesEnum.classification))

    # Create node with imputation operation
    imputation_node = PrimaryNode('simple_imputation')
    imputation_node.fit(nan_data)
    predicted = imputation_node.predict(nan_data)

    assert np.isclose(predicted.predict[1, 0], 1.75)
    assert predicted.predict[1, 1] == '1'
    assert np.isclose(predicted.predict[1, 2], 1)


def test_imputation_binary_features_with_equal_categories_correct():
    """
    The correctness of the gap-filling algorithm is checked on data with binary
    features. The number of known categories in each column is equal. Consequently,
    there is no possibility to insert the majority class label into the gaps.
    Instead of that the mean value is inserted.
    """
    nan_data = data_with_binary_int_features_and_equal_categories()

    imputation_node = PrimaryNode('simple_imputation')
    imputation_node.fit(nan_data)
    predicted = imputation_node.predict(nan_data)

    assert np.isclose(predicted.predict[1, 0], 0.5)
    assert np.isclose(predicted.predict[1, 1], 5.0)


def test_label_encoding_correct():
    """
    Check if LabelEncoder can perform transformations correctly. Also the dataset
    is generated so that new categories appear in the test sample.
    """
    cat_data = data_with_only_categorical_features()
    train_data, test_data = train_test_data_setup(cat_data)

    encoding_node = PrimaryNode('label_encoding')
    encoding_node.fit(train_data)

    predicted_train = encoding_node.predict(train_data)
    predicted_test = encoding_node.predict(test_data)

    # Label 'a' was in the training sample - convert it into 0
    assert predicted_train.predict[0, 0] == 0
    # Label 'b' was in the training sample - convert it into 1
    assert predicted_train.predict[1, 0] == 1
    # Label 'c' was not in the training sample - convert it into 2
    assert predicted_test.predict[0, 0] == 2


def test_lagged_with_multivariate_time_series():
    """
    Checking the correct processing of multivariate time series in the lagged operation
    """
    correct_fit_output = np.array([[0., 1., 10., 11.],
                                   [1., 2., 11., 12.],
                                   [2., 3., 12., 13.],
                                   [3., 4., 13., 14.],
                                   [4., 5., 14., 15.],
                                   [5., 6., 15., 16.],
                                   [6., 7., 16., 17.]])
    correct_predict_output = np.array([[8, 9, 18, 19]])

    input_data = get_multivariate_time_series()
    lagged = LaggedTransformationImplementation(**{'window_size': 2})

    transformed_for_fit = lagged.transform(input_data, is_fit_pipeline_stage=True)
    transformed_for_predict = lagged.transform(input_data, is_fit_pipeline_stage=False)

    # Check correctness on fit stage
    lagged_features = transformed_for_fit.predict
    assert lagged_features.shape == correct_fit_output.shape
    assert np.all(np.isclose(lagged_features, correct_fit_output))

    # Check correctness on predict stage
    lagged_predict = transformed_for_predict.predict
    assert lagged_predict.shape == correct_predict_output.shape
    assert np.all(np.isclose(lagged_predict, correct_predict_output))


def test_poly_features_on_big_datasets():
    """
    Use a table with a large number of features to run a poly features operation.
    For a large number of features the operation should not greatly increase the
    number of columns.
    """
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_classification.csv')
    train_input = InputData.from_csv(os.path.join(test_file_path, file),
                                     task=Task(TaskTypesEnum.classification))

    # Take only small number of rows from dataset
    train_input.features = train_input.features[5: 20, :]
    train_input.idx = np.arange(len(train_input.features))
    train_input.target = train_input.target[5: 20].reshape((-1, 1))

    poly_node = Pipeline(PrimaryNode('poly_features'))
    poly_node.fit(train_input)
    transformed_features = poly_node.predict(train_input)

    n_rows, n_cols = transformed_features.predict.shape
    assert n_cols == 85
