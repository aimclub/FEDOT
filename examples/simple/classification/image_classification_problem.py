from typing import Any

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score as roc_auc

from examples.simple.classification.classification_pipelines import cnn_composite_pipeline
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.models.keras import check_input_array
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import set_random_seed


def calculate_validation_metric(predicted: OutputData, dataset_to_validate: InputData) -> float:
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict,
                            multi_class="ovo")
    return roc_auc_value


def cnn_model_fit(idx: np.array, features: np.array, target: np.array, params: dict):
    x_train, y_train = features, target
    transformed_x_train, transform_flag = check_input_array(x_train)

    if transform_flag:
        print('Train data set was not scaled. The data was divided by 255.')

    if len(x_train.shape) == 3:
        transformed_x_train = np.expand_dims(x_train, -1)

    if len(target.shape) < 2:
        le = preprocessing.OneHotEncoder()
        y_train = le.fit_transform(y_train.reshape(-1, 1)).toarray()

    optimizer_params = {'loss': "categorical_crossentropy",
                        'optimizer': "adam",
                        'metrics': ["accuracy"]}

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=[28, 28, 1]),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

    model.compile(**optimizer_params)
    model.num_classes = 10

    model.fit(transformed_x_train, y_train, batch_size=1, epochs=1,
              validation_split=0.1)

    return model


#
#
def cnn_model_predict(fitted_model: Any, idx: np.array, features: np.array, params: dict):
    x_test = features
    transformed_x_test, transform_flag = check_input_array(x_test)

    if np.max(transformed_x_test) > 1:
        print('Test data set was not scaled. The data was divided by 255.')

    if len(x_test.shape) == 3:
        transformed_x_test = np.expand_dims(x_test, -1)

    # if output_mode == 'labels':
    #    prediction = np.round(trained_model.predict(transformed_x_test))
    # elif output_mode in ['probs', 'full_probs', 'default']:
    prediction = fitted_model.predict(transformed_x_test)
    # if trained_model.num_classes < 2:
    #     print('Data set contain only 1 target class. Please reformat your data.')
    #     raise ValueError('Data set contain only 1 target class. Please reformat your data.')
    # elif trained_model.num_classes == 2 and output_mode != 'full_probs' and len(prediction.shape) > 1:
    #     prediction = prediction[:, 1]
    # else:
    #    raise ValueError(f'Output model {output_mode} is not supported')

    return prediction, 'table'


#

def preproc_predict(fitted_model: Any, idx: np.array, features: np.array, params: dict):
    # example of custom data pre-processing for predict state
    for i in range(features.shape[0]):
        features[i, :, :] = features[i, :, :] + np.random.normal(0, 30)
    return features, 'image'


def cnn_composite_pipeline(composite_flag: bool = True) -> Pipeline:
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/cnn_composite_pipeline.png
      :width: 55%

    Where cnn - convolutional neural network, rf - random forest

    :param composite_flag:  add additional random forest estimator
    """
    node_first = PipelineNode('custom/preproc_image')
    node_first.parameters = {'model_predict': preproc_predict}

    node_second = PipelineNode('custom/cnn_1', nodes_from=[node_first])
    node_second.parameters = {'model_predict': cnn_model_predict,
                              'model_fit': cnn_model_fit}

    node_final = PipelineNode('rf', nodes_from=[node_second])

    pipeline = Pipeline(node_final)
    return pipeline


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
    set_random_seed(1)

    training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_problem(
        train_dataset=training_set,
        test_dataset=testing_set)

    print(roc_auc_on_valid)
