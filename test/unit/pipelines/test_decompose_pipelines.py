from itertools import product

import numpy as np

from examples.simple.classification.classification_with_tuning import get_classification_dataset
from examples.advanced.decompose.refinement_forecast_example import get_refinement_pipeline
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.preprocessing import DataPreprocessor
from test.unit.tasks.test_classification import get_iris_data
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.preprocessing.test_preprocessors import data_with_complicated_types


def generate_pipeline_with_decomposition(primary_operation, secondary_operation):
    """ The function generates a pipeline in which there is an operation of
    decomposing the target variable into residuals
                     secondary_operation
    primary_operation                       rf
                     class_decompose -> rfr

    :param primary_operation: name of operation to place in primary node
    :param secondary_operation: name of operation to place in secondary node
    """

    node_first = PipelineNode(primary_operation)
    node_second = PipelineNode(secondary_operation, nodes_from=[node_first])
    node_decompose = PipelineNode('class_decompose', nodes_from=[node_second, node_first])
    node_rfr = PipelineNode('rfr', nodes_from=[node_decompose])
    node_rf = PipelineNode('rf', nodes_from=[node_rfr, node_second])
    full_pipeline = Pipeline(node_rf)
    return full_pipeline


def generate_pipeline_with_filtering():
    """ Return 5-level pipeline with decompose and filtering operations
           logit
    scaling                                 xgboost
           class_decompose -> RANSAC -> rfr
    """

    node_scaling = PipelineNode('scaling')
    node_logit = PipelineNode('logit', nodes_from=[node_scaling])
    node_decompose = PipelineNode('class_decompose', nodes_from=[node_logit, node_scaling])
    node_ransac = PipelineNode('ransac_lin_reg', nodes_from=[node_decompose])
    node_rfr = PipelineNode('rfr', nodes_from=[node_ransac])
    node_rf = PipelineNode('rf', nodes_from=[node_rfr, node_logit])
    full_pipeline = Pipeline(node_rf)
    return full_pipeline


def generate_cascade_decompose_pipeline():
    """ The function of generating a multi-stage model with many connections
    and solving many problems (regression and classification)
    """

    node_scaling = PipelineNode('scaling')
    node_second = PipelineNode('logit', nodes_from=[node_scaling])
    node_decompose = PipelineNode('class_decompose', nodes_from=[node_second, node_scaling])
    node_rfr = PipelineNode('rfr', nodes_from=[node_decompose])
    node_rf = PipelineNode('rf', nodes_from=[node_rfr, node_second])
    node_decompose_new = PipelineNode('class_decompose', nodes_from=[node_rf, node_scaling])
    node_rfr_2 = PipelineNode('rfr', nodes_from=[node_decompose_new])
    node_final = PipelineNode('logit', nodes_from=[node_rfr_2, node_rf])
    pipeline = Pipeline(node_final)
    return pipeline


def get_classification_data(
        classes_amount: int = 2,
        samples_amount: int = 800,
        features_amount: int = 4,
):
    """ Function generate synthetic dataset for classification task

    :param classes_amount: amount of classes to predict

    :return train_input: InputData for model fit
    :return predict_input: InputData for predict stage
    """

    # Define options for dataset with 800 objects
    features_options = {'informative': 2, 'redundant': 1,
                        'repeated': 1, 'clusters_per_class': 1}
    x_train, y_train, x_test, y_test = get_classification_dataset(
        features_options,
        samples_amount=samples_amount,
        features_amount=features_amount,
        classes_amount=classes_amount
    )
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # Define classification task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train and validate the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train, target=y_train,
                            task=task, data_type=DataTypesEnum.table)
    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test, target=y_test,
                              task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input


def test_finding_side_root_node():
    """
    The function tests finding side root node in pipeline with the following structure:

    ↑    ->   logit      ->
    ↑          ↓                  ↓
   scaling     ↓                   xgboost -> final prediction
    ↓          ↓                  ↑
    ↓    -> decompose -> rfr ->

    Where logit - logistic regression, rfr - random forest regression, xgboost - xg boost classifier
    """

    reg_root_node = 'rfr'

    pipeline = generate_pipeline_with_decomposition('scaling', 'logit')
    reg_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.regression)

    assert reg_pipeline.root_node.operation.operation_type == reg_root_node


def test_pipeline_for_side_task_predict():
    """ Checks whether the pipeline for the side task gives correct predictions """

    pipeline = generate_pipeline_with_decomposition('scaling', 'logit')

    train_data, test_data = data_with_complicated_types()
    pipeline.fit_from_scratch(train_data)
    predicted_labels = pipeline.predict(test_data)
    preds = predicted_labels.predict

    reg_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.regression)
    reg_predicted_labels = reg_pipeline.predict(test_data)
    reg_preds = reg_predicted_labels.predict

    assert reg_predicted_labels is not None
    assert not (preds == reg_preds).all()


def test_order_by_data_flow_len_correct():
    """ The function checks whether the current version of data flow length
    counters can allow for decompose implementation to determine how the nodes
    in the graph are located
    """
    data_operations = ['scaling', 'normalization', 'pca', 'poly_features']
    model_operations = ['lda', 'knn', 'logit']
    list_with_operations = list(product(data_operations, model_operations))

    for data_operation, model_operation in list_with_operations:
        input_data = get_iris_data()
        input_data = DataPreprocessor().obligatory_prepare_for_fit(input_data)

        # Generate pipeline with different operations in the nodes with decomposition
        pipeline = generate_pipeline_with_decomposition(data_operation,
                                                        model_operation)
        pipeline.fit(input_data)

        # Get one node with decompose operation in it
        decompose_nodes = pipeline.get_nodes_by_name('class_decompose')
        decompose_node = decompose_nodes[0]
        # Predict from decompose must be the same as predict from Data parent
        dec_output = decompose_node.predict(input_data)

        # Get data parent operation for node
        data_node = pipeline.get_nodes_by_name(data_operation)[0]
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
    prediction = pipeline.predict(predict_input)

    assert prediction is not None


def test_multiclass_classification_decomposition():
    # Generate synthetic dataset for multiclass classification task
    train_input, predict_input = get_classification_data(classes_amount=4)

    # Distort labels targets
    train_input.target = np.array(train_input.target) + 1
    predict_input.target = np.array(predict_input.target) + 1

    # Get pipeline
    pipeline = generate_pipeline_with_decomposition('scaling', 'logit')
    pipeline.fit(train_input)
    prediction = pipeline.predict(predict_input)

    assert prediction is not None


def test_cascade_classification_decomposition():
    # Generate synthetic dataset for multiclass classification task
    train_input, predict_input = get_classification_data(classes_amount=4)

    # Distort labels targets
    train_input.target = np.array(train_input.target) + 3
    predict_input.target = np.array(predict_input.target) + 3

    # Get pipeline
    pipeline = generate_cascade_decompose_pipeline()
    pipeline.fit(train_input)
    prediction = pipeline.predict(predict_input)

    assert prediction is not None


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
