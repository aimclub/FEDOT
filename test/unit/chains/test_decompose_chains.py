import numpy as np
from itertools import product

from fedot.core.chains.chain import Chain, nodes_with_operation
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data import InputData
from examples.classification_with_tuning_example import get_classification_dataset


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


def test_order_by_descriptive_id():
    """
    The function checks the order of nodes by its descriptive_id property
    """
    # Generate chain with defined operation in the nodes
    full_chain = generate_chain_with_decomposition(primary_operation='scaling',
                                                   secondary_operation='logit')
    desc_decompose = '((/n_scaling_default_params;)/n_logit_default_params;;' \
                     '/n_scaling_default_params;)/n_class_decompose_default_params'

    for node in full_chain.nodes:
        if node.operation.operation_type == 'class_decompose':
            assert node.descriptive_id == desc_decompose


def test_order_by_descriptive_correct():
    """ The function checks whether the current version of descriptive_id can
    determine how the nodes in the chain are located
    """

    data_operations = ['scaling', 'normalization', 'pca', 'dt', 'poly_features']
    model_operations = ['lda', 'qda', 'knn', 'logit']
    list_with_operations = list(product(data_operations, model_operations))

    for data_operation, model_operation in list_with_operations:
        # Generate chain with different operations in the nodes with decomposition
        chain = generate_chain_with_decomposition(data_operation,
                                                  model_operation)

        # Get nodes with decompose operation in it
        decompose_nodes = nodes_with_operation(chain, 'class_decompose')
        decompose_node = decompose_nodes[0]

        # Get parents for decompose node
        parent_nodes = decompose_node._nodes_from_with_fixed_order()
        parent_nodes = np.array(parent_nodes, dtype=str)

        # Get orders for parent operations
        data_parent_id = int(np.argwhere(parent_nodes == data_operation))
        model_parent_id = int(np.argwhere(parent_nodes == model_operation))

        # Data parent must be always in the end of the list
        if data_parent_id != 1 and model_parent_id != 0:
            raise ValueError('Parent nodes for decompose operation has incorrect order')


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


def test_cascade_decompose_classification_decomposition():
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
