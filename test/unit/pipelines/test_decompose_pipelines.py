from itertools import product

import numpy as np

from fedot.core.pipelines.pipeline import nodes_with_operation
from data.data_manager import get_iris_data, get_ts_data, get_classification_data
from data.pipeline_manager import generate_pipeline_with_decomposition,\
    generate_pipeline_with_filtering, generate_cascade_decompose_pipeline, get_refinement_pipeline


def test_order_by_data_flow_len_correct():
    """ The function checks whether the current version of data flow length
    counters can allow for decompose implementation to determine how the nodes
    in the graph are located
    """
    input_data = get_iris_data()

    data_operations = ['scaling', 'normalization', 'pca', 'poly_features']
    model_operations = ['lda', 'knn', 'logit']
    list_with_operations = list(product(data_operations, model_operations))

    for data_operation, model_operation in list_with_operations:
        # Generate pipeline with different operations in the nodes with decomposition
        pipeline = generate_pipeline_with_decomposition(data_operation,
                                                        model_operation)
        pipeline.fit(input_data)

        # Get one node with decompose operation in it
        decompose_nodes = nodes_with_operation(pipeline, 'class_decompose')
        decompose_node = decompose_nodes[0]
        # Predict from decompose must be the same as predict from Data parent
        dec_output = decompose_node.predict(input_data)

        # Get data parent operation for node
        data_node = nodes_with_operation(pipeline, data_operation)[0]
        data_output = data_node.predict(input_data)

        if tuple(data_output.predict.shape) != tuple(dec_output.predict.shape):
            raise ValueError('Data parent is not identified correctly for the decompose operation')


def test_correctness_filter_pipeline_decomposition():
    """ The function runs an example of classification task using an outlier
    filtering algorithm (RANSAC) in its structure in the regression branch
    """
    # Generate synthetic dataset for binary classification task
    train_input, predict_input = get_classification_data(classes_amount=2)

    # Distort labels in targets
    train_input.target = np.array(train_input.target) + 2
    predict_input.target = np.array(predict_input.target) + 2

    # Get pipeline
    pipeline = generate_pipeline_with_filtering()
    pipeline.fit(train_input)
    predicted_output = pipeline.predict(predict_input)

    is_pipeline_worked_correctly = True
    return is_pipeline_worked_correctly


def test_multiclass_classification_decomposition():
    # Generate synthetic dataset for multiclass classification task
    train_input, predict_input = get_classification_data(classes_amount=4)

    # Distort labels targets
    train_input.target = np.array(train_input.target) + 1
    predict_input.target = np.array(predict_input.target) + 1

    # Get pipeline
    pipeline = generate_pipeline_with_decomposition('scaling', 'logit')
    pipeline.fit(train_input)
    predicted_output = pipeline.predict(predict_input)

    is_pipeline_worked_correctly = True
    return is_pipeline_worked_correctly


def test_cascade_classification_decomposition():
    # Generate synthetic dataset for multiclass classification task
    train_input, predict_input = get_classification_data(classes_amount=4)

    # Distort labels targets
    train_input.target = np.array(train_input.target) + 3
    predict_input.target = np.array(predict_input.target) + 3

    # Get pipeline
    pipeline = generate_cascade_decompose_pipeline()
    pipeline.fit(train_input)
    predicted_output = pipeline.predict(predict_input)

    is_pipeline_worked_correctly = True
    return is_pipeline_worked_correctly


def test_ts_forecasting_decomposition():
    """ The function checks whether, after the decompose operation, the pipeline
    actually models the original target (not decomposed) for the time series
    forecasting task
    """
    # Generate synthetic data for time series forecasting
    train_data, _ = get_ts_data(forecast_length=5)
    # Distort original values
    train_data.features = train_data.features + 150
    train_data.target = train_data.target + 150

    _, pipeline_decompose_finish, pipeline = get_refinement_pipeline(lagged=10)

    pipeline.fit(train_data)
    pipeline_decompose_finish.fit(train_data)

    full_output = pipeline.predict(train_data)
    decompose_output = pipeline_decompose_finish.predict(train_data)

    full_level = np.mean(full_output.predict)
    decompose_level = np.mean(decompose_output.predict)

    assert full_level > (decompose_level + 100)
