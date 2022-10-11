import logging
import os
from typing import Optional

import numpy as np

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.requirements_notificator import warn_requirement

try:
    import tensorflow as tf
except ModuleNotFoundError:
    warn_requirement('tensorflow')
    tf = None

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import LoggerAdapter, default_log
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from sklearn import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_input_array(x_train):
    if np.max(x_train) > 1:
        transformed_x_train = x_train.astype("float32") / 255
        transform_flag = True
    else:
        transformed_x_train = x_train
        transform_flag = False

    return transformed_x_train, transform_flag


def create_deep_cnn(input_shape: tuple,
                    num_classes: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def create_simple_cnn(input_shape: tuple,
                      num_classes: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def create_vgg16(input_shape: tuple,
                 num_classes: int):
    model = tf.keras.applications.vgg16.VGG16(include_top=True,
                                              weights=None,
                                              input_shape=input_shape,
                                              classes=num_classes,
                                              classifier_activation='sigmoid')
    return model


def fit_cnn(train_data: InputData,
            model,
            epochs: int = 10,
            batch_size: int = 128,
            optimizer_params: dict = None,
            logger: Optional[LoggerAdapter] = None):
    x_train, y_train = train_data.features, train_data.target
    transformed_x_train, transform_flag = check_input_array(x_train)

    if logger is None:
        logger = default_log(prefix=__name__)

    if transform_flag:
        logger.debug('Train data set was not scaled. The data was divided by 255.')

    if len(x_train.shape) == 3:
        transformed_x_train = np.expand_dims(x_train, -1)

    if len(train_data.target.shape) < 2:
        le = preprocessing.OneHotEncoder()
        y_train = le.fit_transform(y_train.reshape(-1, 1)).toarray()

    if optimizer_params is None:
        optimizer_params = {'loss': "categorical_crossentropy",
                            'optimizer': "adam",
                            'metrics': ["accuracy"]}

    model.compile(**optimizer_params)
    model.num_classes = train_data.num_classes
    if logger is None:
        logger = default_log(prefix=__name__)

    if logger.logging_level > logging.DEBUG:
        verbose = 0
    else:
        verbose = 2

    if epochs is None:
        logger.warning('The number of training epochs was not set. The selected number of epochs is 10.')

    model.fit(transformed_x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.1, verbose=verbose)

    return model


def predict_cnn(trained_model, predict_data: InputData, output_mode: str = 'labels', logger=None) -> OutputData:
    x_test = predict_data.features
    transformed_x_test, transform_flag = check_input_array(x_test)

    if logger is None:
        logger = default_log(prefix=__name__)

    if np.max(transformed_x_test) > 1:
        logger.warning('Test data set was not scaled. The data was divided by 255.')

    if len(x_test.shape) == 3:
        transformed_x_test = np.expand_dims(x_test, -1)

    if output_mode == 'labels':
        prediction = np.round(trained_model.predict(transformed_x_test))
    elif output_mode in ['probs', 'full_probs', 'default']:
        prediction = trained_model.predict(transformed_x_test)
        if trained_model.num_classes < 2:
            logger.error('Data set contain only 1 target class. Please reformat your data.')
            raise NotImplementedError()
        elif trained_model.num_classes == 2 and output_mode != 'full_probs' and len(prediction.shape) > 1:
            prediction = prediction[:, 1]
    else:
        raise ValueError(f'Output model {output_mode} is not supported')
    return prediction


cnn_model_dict = {'deep': create_deep_cnn,
                  'simplified': create_simple_cnn,
                  'vgg16': create_vgg16}


class FedotCNNImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        default_params = {'log': default_log(prefix=__name__),
                          'epochs': 10,
                          'batch_size': 32,
                          'output_mode': 'labels',
                          'architecture_type': 'simplified',
                          'optimizer_parameters': {'loss': "categorical_crossentropy",
                                                   'optimizer': "adam",
                                                   'metrics': ["accuracy"]}}

        complete_params = {**default_params, **self.params.to_dict()}
        self.params.update(**complete_params)

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        # TODO make case for multiclass multioutput task
        # check for multioutput target
        if len(train_data.target.shape) < 2:
            self.classes = np.unique(train_data.target)
        else:
            self.classes = np.arange(train_data.target.shape[1])

        self.model = cnn_model_dict[self.params.get('architecture_type')](input_shape=train_data.features.shape[1:4],
                                                                          num_classes=len(self.classes))

        self.model = fit_cnn(train_data=train_data, model=self.model, epochs=self.params.get('epochs'),
                             batch_size=self.params.get('batch_size'),
                             optimizer_params=self.params.get('optimizer_parameters'), logger=self.params.get('log'))
        return self.model

    def predict(self, input_data):
        """ Method make prediction with labels of classes for predict stage

        :param input_data: data with features to process
        """

        return predict_cnn(trained_model=self.model, predict_data=input_data,
                           output_mode='labels', logger=self.params['log'])

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """

        return predict_cnn(trained_model=self.model, predict_data=input_data, output_mode='probs')

    @property
    def classes_(self):
        return self.classes

    def __deepcopy__(self, memo=None):
        clone_model = tf.keras.models.clone_model(self.model)
        clone_model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)
        clone_model.set_weights(self.model.get_weights())
        return clone_model
