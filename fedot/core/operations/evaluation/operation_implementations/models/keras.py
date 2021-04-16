import numpy as np
import os
import tensorflow as tf
from typing import Optional

from sklearn import preprocessing

from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.log import Log, default_log
from fedot.core.data.data import InputData, OutputData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _create_cnn(input_shape: tuple,
                num_classes: int,
                architecture_type: str = 'deep',
                logger: Log = None):
    if architecture_type == 'deep':
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
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
    elif architecture_type == 'shallow':
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
    else:
        model = None
        logger.error(f'{architecture_type} is incorrect type of NN architecture')

    return model


def fit_cnn(train_data: InputData,
            model,
            epochs: int = 1,
            batch_size: int = 128,
            logger: Log = None):
    x_train, y_train = train_data.features, train_data.target
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    le = preprocessing.OneHotEncoder()
    y_train = le.fit_transform(y_train.reshape(-1, 1)).toarray()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    if logger.verbosity_level < 4:
        verbose = 0
    else:
        verbose = 2

    if epochs is None:
        epochs = 10

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=verbose)

    return model


def predict_cnn(trained_model, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
    x_test, y_test = predict_data.features, predict_data.target
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    if output_mode == 'labels':
        prediction = trained_model.predict(x_test)
    elif output_mode in ['probs', 'full_probs', 'default']:
        prediction = trained_model.predict_proba(x_test)
        if predict_data.num_classes < 2:
            raise NotImplementedError()
        elif predict_data.num_classes == 2 and output_mode != 'full_probs':
            prediction = prediction[:, 1]
    else:
        raise ValueError(f'Output model {output_mode} is not supported')
    return prediction


class CustomCNNImplementation(ModelImplementation):
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.log: Log = default_log(__name__)
        self.params = {'image_shape': (28, 28, 1),
                       'num_classes': 2,
                       'log': self.log,
                       'epochs': 10,
                       'batch_size': 128,
                       'output_mode': 'labels',
                       'architecture_type': 'shallow'}
        if not params:
            self.model = _create_cnn(input_shape=self.params['image_shape'],
                                     num_classes=self.params['num_classes'],
                                     architecture_type=self.params['architecture_type'])
        else:
            self.params = {**params, **self.params}
            self.model = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        self.classes = np.unique(train_data.target)

        if self.model is None:
            self.model = _create_cnn(input_shape=self.params['image_shape'],
                                     num_classes=len(self.classes),
                                     architecture_type=self.params['architecture_type'])

        self.model = fit_cnn(train_data=train_data, model=self.model, epochs=self.params['epochs'],
                             batch_size=self.params['batch_size'], logger=self.log)
        return self.model

    def predict(self, input_data, is_fit_chain_stage: Optional[bool] = None):
        """ Method make prediction with labels of classes

        :param input_data: data with features to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """

        return predict_cnn(trained_model=self.model, predict_data=input_data, output_mode='labels')

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """

        return predict_cnn(trained_model=self.model, predict_data=input_data, output_mode='probs')

    def get_params(self):
        """ Method return parameters, which can be optimized for particular
        operation
        """
        return self.params

    @property
    def classes_(self):
        return self.classes
