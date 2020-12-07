import tensorflow as tf
import random
from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.chain import Chain
from fedot.core.composer.node import PrimaryNode, SecondaryNode
from fedot.core.models.data import InputData
from fedot.core.models.preprocessing import EmptyStrategy
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def get_composite_chain() -> Chain:
    node_first = PrimaryNode('cnn')
    node_second = PrimaryNode('cnn')
    node_final = SecondaryNode('rf', nodes_from=[node_first, node_second])

    chain = Chain(node_final)

    return chain


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_image_recognitation_problem(train_dataset: Union[str, tuple],
                                    test_dataset: Union[str, tuple],
                                    augmentation_flag: bool = False,
                                    is_visualise: bool = False):
    task = Task(TaskTypesEnum.image_classification)

    if type(train_dataset) is tuple:
        X_train, y_train = train_dataset[0], train_dataset[1]
        X_test, y_test = test_dataset[0], test_dataset[1]

    dataset_to_train = InputData.from_image(images=X_train, labels=y_train, task=task, aug_flag=augmentation_flag)
    dataset_to_validate = InputData.from_image(images=X_test, labels=y_test, task=task, aug_flag=augmentation_flag)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    # available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    chain_simple = PrimaryNode(model_type='cnn')
    chain_simple.manual_preprocessing_func = EmptyStrategy
    chain_simple.fit(input_data=dataset_to_train, verbose=False)
    predictions = chain_simple.predict(dataset_to_validate)
    roc_auc_on_valid_simple = calculate_validation_metric(chain_simple.predict(dataset_to_validate),
                                                          dataset_to_validate)
    print(f'RMSE simple: {roc_auc_on_valid_simple}')

    return roc_auc_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition
    # load MNIST dataset
    training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    run_image_recognitation_problem(train_dataset=training_set,
                                    test_dataset=testing_set,
                                    is_visualise=True)
