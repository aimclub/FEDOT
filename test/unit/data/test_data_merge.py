import numpy as np

from fedot.core.data.data import InputData
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data import OutputData
from fedot.core.data.merge import DataMerger

from examples.regression_with_tuning_example import get_regression_dataset

np.random.seed(2021)


def test_data_merge_in_chain():
    """ Test check is the chain can correctly work with dynamic changes in
    tables during the fit process
    """

    #   ridge
    #  /     \ (merge operation)
    # |   ransac_lin_reg (remove several lines in table)
    #  \     /
    #  scaling

    node_scaling = PrimaryNode('scaling')

    node_lin_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_final = SecondaryNode('ridge', nodes_from=[node_lin_ransac, node_scaling])
    chain = Chain(node_final)

    features_options = {'informative': 2, 'bias': 2.0}
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options=features_options,
                                                              samples_amount=100,
                                                              features_amount=5)
    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    # Fit and predict
    chain.fit_from_scratch(train_input)
    prediction = chain.predict(train_input)

    print(prediction)
    assert prediction is not None


def test_data_merge_function():
    """ Test check is the merge function can find appropriate intersections of
    indices. Set {idx_2} ∈ set {idx_1}, so intersection must be = idx_2
    """

    idx_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    idx_2 = [2, 3, 4, 5, 6, 7, 8, 9]

    task = Task(TaskTypesEnum.regression)
    generated_target = np.random.sample((len(idx_1), 1))
    generated_features = np.random.sample((len(idx_1), 2))

    list_with_outputs = []
    for idx in [idx_1, idx_2]:
        output_data = OutputData(idx=idx,
                                 features=generated_features[idx, :],
                                 predict=generated_target[idx, :],
                                 task=task,
                                 target=generated_target[idx, :],
                                 data_type=DataTypesEnum.table)
        list_with_outputs.append(output_data)

    idx, features, target = DataMerger(list_with_outputs).merge()

    assert tuple(idx) == tuple(idx_2)
