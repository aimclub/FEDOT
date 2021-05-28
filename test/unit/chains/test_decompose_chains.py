import numpy as np
from itertools import product

from fedot.core.chains.chain import Chain, nodes_with_operation
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data import InputData

from examples.classification_with_tuning_example import get_classification_dataset
from examples.decompose.refinement_forecast_example import get_refinement_chain

from test.unit.tasks.test_classification import get_iris_data
from test.unit.tasks.test_forecasting import get_synthetic_ts_data_period


def generate_chain_with_decomposition(primary_operation, secondary_operation):
    """ The function generates a chain in which there is an operation of
    decomposing the target variable into residuals
                     secondary_operation
    primary_operation                       xgboost
                     class_decompose -> rfr

    :param primary_operation: name of operation to place in primary node
    :param secondary_operation: name of operation to place in secondary node
    """

    node_first = PrimaryNode(primary_operation)
    node_second = SecondaryNode(secondary_operation, nodes_from=[node_first])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_second, node_first])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_second])
    full_chain = Chain(node_xgboost)
    return full_chain


def generate_chain_with_filtering():
    """ Return 5-level chain with decompose and filtering operations
           logit
    scaling                                 xgboost
           class_decompose -> RANSAC -> rfr
    """

    node_scaling = PrimaryNode('scaling')
    node_logit = SecondaryNode('logit', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_logit, node_scaling])
    node_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_decompose])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_ransac])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_logit])
    full_chain = Chain(node_xgboost)
    return full_chain


def generate_cascade_decompose_chain():
    """ The function of generating a multi-stage model with many connections
    and solving many problems (regression and classification)
    """

    node_scaling = PrimaryNode('scaling')
    node_second = SecondaryNode('logit', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_second, node_scaling])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_second])
    node_decompose_new = SecondaryNode('class_decompose', nodes_from=[node_xgboost, node_scaling])
    node_rfr_2 = SecondaryNode('rfr', nodes_from=[node_decompose_new])
    node_final = SecondaryNode('logit', nodes_from=[node_rfr_2, node_xgboost])
    chain = Chain(node_final)
    return chain


def get_classification_data(classes_amount: int):
    """ Function generate synthetic dataset for classification task

    :param classes_amount: amount of classes to predict

    :return train_input: InputData for model fit
    :return predict_input: InputData for predict stage
    """

    # Define options for dataset with 800 objects
    features_options = {'informative': 2, 'redundant': 1,
                        'repeated': 1, 'clusters_per_class': 1}
    x_train, y_train, x_test, y_test = get_classification_dataset(features_options,
                                                                  800, 4,
                                                                  classes_amount)
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


def test_order_by_data_flow_len_correct():
    """ The function checks whether the current version of data flow length
    counters can allow for decompose implementation to determine how the nodes
    in the chain are located
    """
    input_data = get_iris_data()

    data_operations = ['scaling', 'normalization', 'pca', 'poly_features']
    model_operations = ['lda', 'knn', 'logit']
    list_with_operations = list(product(data_operations, model_operations))

    for data_operation, model_operation in list_with_operations:
        # Generate chain with different operations in the nodes with decomposition
        chain = generate_chain_with_decomposition(data_operation,
                                                  model_operation)
        chain.fit(input_data)

        # Get one node with decompose operation in it
        decompose_nodes = nodes_with_operation(chain, 'class_decompose')
        decompose_node = decompose_nodes[0]
        # Predict from decompose must be the same as predict from Data parent
        dec_output = decompose_node.predict(input_data)

        # Get data parent operation for node
        data_node = nodes_with_operation(chain, data_operation)[0]
        data_output = data_node.predict(input_data)

        if tuple(data_output.predict.shape) != tuple(dec_output.predict.shape):
            raise ValueError('Data parent is not identified correctly for the decompose operation')


def test_correctness_filter_chain_decomposition():
    """ The function runs an example of classification task using an outlier
    filtering algorithm (RANSAC) in its structure in the regression branch
    """
    # Generate synthetic dataset for binary classification task
    train_input, predict_input = get_classification_data(classes_amount=2)

    # Distort labels in targets
    train_input.target = np.array(train_input.target) + 2
    predict_input.target = np.array(predict_input.target) + 2

    # Get chain
    chain = generate_chain_with_filtering()
    chain.fit(train_input)
    predicted_output = chain.predict(predict_input)

    is_chain_worked_correctly = True
    return is_chain_worked_correctly


def test_multiclass_classification_decomposition():
    # Generate synthetic dataset for multiclass classification task
    train_input, predict_input = get_classification_data(classes_amount=4)

    # Distort labels targets
    train_input.target = np.array(train_input.target) + 1
    predict_input.target = np.array(predict_input.target) + 1

    # Get chain
    chain = generate_chain_with_decomposition('scaling', 'logit')
    chain.fit(train_input)
    predicted_output = chain.predict(predict_input)

    is_chain_worked_correctly = True
    return is_chain_worked_correctly


def test_cascade_classification_decomposition():
    # Generate synthetic dataset for multiclass classification task
    train_input, predict_input = get_classification_data(classes_amount=4)

    # Distort labels targets
    train_input.target = np.array(train_input.target) + 3
    predict_input.target = np.array(predict_input.target) + 3

    # Get chain
    chain = generate_cascade_decompose_chain()
    chain.fit(train_input)
    predicted_output = chain.predict(predict_input)

    is_chain_worked_correctly = True
    return is_chain_worked_correctly


def test_ts_forecasting_decomposition():
    """ The function checks whether, after the decompose operation, the chain
    actually models the original target (not decomposed) for the time series
    forecasting task
    """
    # Generate synthetic data for time series forecasting
    train_data, _ = get_synthetic_ts_data_period(forecast_length=5)
    # Distort original values
    train_data.features = train_data.features + 150
    train_data.target = train_data.target + 150

    _, chain_decompose_finish, chain = get_refinement_chain(lagged=10)

    chain.fit(train_data)
    chain_decompose_finish.fit(train_data)

    full_output = chain.predict(train_data)
    decompose_output = chain_decompose_finish.predict(train_data)

    full_level = np.mean(full_output.predict)
    decompose_level = np.mean(decompose_output.predict)

    assert full_level > (decompose_level + 100)
