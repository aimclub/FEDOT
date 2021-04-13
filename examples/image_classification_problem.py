import tensorflow as tf
import random
from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def get_composite_chain() -> Chain:
    node_first = PrimaryNode('cnn')
    node_first.custom_params = {'architecture': 'deep',
                                'num_classes': 10,
                                'epochs': 15}
    node_second = PrimaryNode('cnn')
    node_second.custom_params = {'image_shape': (28, 28, 1),
                                 'num_classes': 10,
                                 'epochs': 10,
                                 'batch_size': 128,
                                 'architecture_type': 'shallow'}
    node_final = SecondaryNode('rf', nodes_from=[node_first, node_second])

    chain = Chain(node_final)

    return chain


def calculate_validation_metric(predicted: OutputData, dataset_to_validate: InputData) -> float:
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict,
                            multi_class="ovo")
    return roc_auc_value


def run_image_classification_problem(train_dataset: tuple,
                                     test_dataset: tuple):
    task = Task(TaskTypesEnum.classification)

    x_train, y_train = train_dataset[0], train_dataset[1]
    x_test, y_test = test_dataset[0], test_dataset[1]

    dataset_to_train = InputData.from_image(images=x_train,
                                            labels=y_train,
                                            task=task)
    dataset_to_validate = InputData.from_image(images=x_test,
                                               labels=y_test,
                                               task=task)

    chain = get_composite_chain()
    chain.fit(input_data=dataset_to_train)
    predictions = chain.predict(dataset_to_validate)
    roc_auc_on_valid_simple = calculate_validation_metric(predictions,
                                                          dataset_to_validate)
    print(f'ROCAUC simple: {roc_auc_on_valid_simple}')

    return roc_auc_on_valid_simple


if __name__ == '__main__':
    training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    run_image_classification_problem(train_dataset=training_set,
                                     test_dataset=testing_set)
