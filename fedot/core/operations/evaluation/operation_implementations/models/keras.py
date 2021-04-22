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
    return model


def create_simple_cnn(input_shape: tuple,
                      num_classes: int):
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

    return model


def fit_cnn(train_data: InputData,
            model,
            epochs: int = 10,
            batch_size: int = 128,
            optimizer_params: dict = None,
            logger: Log = None):
    x_train, y_train = train_data.features, train_data.target
    transformed_x_train, transform_flag = check_input_array(x_train)

    if logger is None:
        logger = default_log(__name__)

    if transform_flag:
        logger.warn('Train data set was not scaled. The data was divided by 255.')

    transformed_x_train = np.expand_dims(x_train, -1)
    le = preprocessing.OneHotEncoder()
    y_train = le.fit_transform(y_train.reshape(-1, 1)).toarray()

    if optimizer_params is None:
        optimizer_params = {'loss': "categorical_crossentropy",
                            'optimizer': "adam",
                            'metrics': ["accuracy"]}

    model.compile(**optimizer_params)

    if logger is None:
        logger = default_log(__name__)

    if logger.verbosity_level < 4:
        verbose = 0
    else:
        verbose = 2

    if epochs is None:
        logger.warn('The number of training epochs was not set. The selected number of epochs is 10.')

    model.fit(transformed_x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=verbose)

    return model


def predict_cnn(trained_model, predict_data: InputData, output_mode: str = 'labels', logger=None) -> OutputData:
    x_test, y_test = predict_data.features, predict_data.target
    transformed_x_test, transform_flag = check_input_array(x_test)

    if logger is None:
        logger = default_log(__name__)

    if np.max(transformed_x_test) > 1:
        logger.warn('Test data set was not scaled. The data was divided by 255.')
    transformed_x_test = np.expand_dims(x_test, -1)

    if output_mode == 'labels':
        prediction = trained_model.predict(transformed_x_test)
    elif output_mode in ['probs', 'full_probs', 'default']:
        prediction = trained_model.predict_proba(transformed_x_test)
        if predict_data.num_classes < 2:
            logger.error('Data set contain only 1 target class. Please reformat your data.')
            raise NotImplementedError()
        elif predict_data.num_classes == 2 and output_mode != 'full_probs':
            prediction = prediction[:, 1]
    else:
        raise ValueError(f'Output model {output_mode} is not supported')
    return prediction


cnn_model_dict = {'deep': create_deep_cnn,
                  'simplified': create_simple_cnn}


class CustomCNNImplementation(ModelImplementation):
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = {'image_shape': (28, 28, 1),
                       'num_classes': 2,
                       'log': default_log(__name__),
                       'epochs': 10,
                       'batch_size': 128,
                       'output_mode': 'labels',
                       'architecture_type': 'simplified',
                       'optimizer_parameters': {'loss': "categorical_crossentropy",
                                                'optimizer': "adam",
                                                'metrics': ["accuracy"]}}
        if not params:
            self.model = cnn_model_dict[self.params['architecture_type']](input_shape=self.params['image_shape'],
                                                                          num_classes=self.params['num_classes'])
        else:
            self.params = {**params, **self.params}
            self.model = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        self.classes = np.unique(train_data.target)

        if self.model is None:
            self.model = cnn_model_dict[self.params['architecture_type']](input_shape=self.params['image_shape'],
                                                                          num_classes=len(self.classes))

        self.model = fit_cnn(train_data=train_data, model=self.model, epochs=self.params['epochs'],
                             batch_size=self.params['batch_size'],
                             optimizer_params=self.params['optimizer_parameters'], logger=self.params['log'])
        return self.model

    def predict(self, input_data, is_fit_chain_stage: Optional[bool] = None):
        """ Method make prediction with labels of classes

        :param input_data: data with features to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """

        return predict_cnn(trained_model=self.model, predict_data=input_data,
                           output_mode='labels', logger=self.params['log'])

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
