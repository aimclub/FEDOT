import random

import numpy as np

from fedot.utilities.requirements_notificator import warn_requirement

try:
    import tensorflow as tf
except ModuleNotFoundError:
    warn_requirement('tensorflow')

from sklearn.metrics import roc_auc_score as roc_auc

from examples.simple.classification.classification_pipelines import cnn_composite_pipeline
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(predicted: OutputData, dataset_to_validate: InputData) -> float:
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict,
                            multi_class="ovo")
    return roc_auc_value


def run_image_classification_problem(train_dataset: tuple,
                                     test_dataset: tuple,
                                     composite_flag: bool = True):
    task = Task(TaskTypesEnum.classification)

    x_train, y_train = train_dataset[0], train_dataset[1]
    x_test, y_test = test_dataset[0], test_dataset[1]

    dataset_to_train = InputData.from_image(images=x_train,
                                            labels=y_train,
                                            task=task)
    dataset_to_validate = InputData.from_image(images=x_test,
                                               labels=y_test,
                                               task=task)

    pipeline = cnn_composite_pipeline(composite_flag)
    pipeline.fit(input_data=dataset_to_train)
    predictions = pipeline.predict(dataset_to_validate)
    roc_auc_on_valid = calculate_validation_metric(predictions,
                                                   dataset_to_validate)
    return roc_auc_on_valid, dataset_to_train, dataset_to_validate


if __name__ == '__main__':
    training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_problem(
        train_dataset=training_set,
        test_dataset=testing_set)
