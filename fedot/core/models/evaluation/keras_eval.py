import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from copy import copy
from datetime import timedelta
from typing import Optional
from fedot.core.data.data import InputData, OutputData
from fedot.core.models.evaluation.evaluation import EvaluationStrategy
from fedot.core.repository.tasks import extract_task_param
from fedot.core.log import Log, default_log

forecast_length = 1


class KerasClassificationStrategy(EvaluationStrategy):
    def __init__(self, model_type: str, params: Optional[dict] = None, log=None):
        self._init_CNN_model_functions(model_type)
        self.architecture_type = 'deep'
        self.epochs = 10
        self.batch_size = 128
        self.output_mode = 'labels'

        if params:
            try:
                self.epochs = params.get('epochs')
                self.architecture_type = params.get('architecture')
            except Exception:
                print('Please choose number of epochs and type of model')

        if not log:
            self.log: Log = default_log(__name__)
        else:
            self.log: Log = log

        super().__init__(model_type, params)

    def _init_CNN_model_functions(self, model_type):
        if model_type != 'cnn':
            raise ValueError(f'Impossible to obtain classififcation strategy for {model_type}')

    def fit(self, train_data: InputData):
        model = fit_cnn(train_data, epochs=self.epochs, batch_size=self.batch_size,
                        logger=self.log, architecture_type=self.architecture_type)
        return model

    def predict(self, trained_model, predict_data: InputData):
        return predict_cnn(trained_model, predict_data, output_mode=self.output_mode)

    def fit_tuned(self, train_data: InputData, iterations: int = 30,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        raise NotImplementedError()


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
        logger.error(f'{architecture_type} is incorrect type of NN architecture')

    return model


def fit_cnn(train_data: InputData,
            image_shape: tuple = (28, 28, 1),
            num_classes: int = 10,
            epochs: int = 1,
            batch_size: int = 128,
            logger: Log = None,
            architecture_type: str = 'deep'):
    x_train, y_train = train_data.features, train_data.target
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)

    model = _create_cnn(input_shape=image_shape,
                        num_classes=num_classes,
                        architecture_type=architecture_type,
                        logger=logger)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    if logger.verbosity_level < 4:
        verbose = 0
    else:
        verbose = 2

    if epochs is None:
        epochs = 10

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=verbose)

    return model


def predict_cnn(trained_model, predict_data: InputData, output_mode:str = 'labels') -> OutputData:
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


# TODO inherit this and similar from custom strategy
class KerasForecastingStrategy(EvaluationStrategy):
    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._init_lstm_model_functions(model_type)

        self.epochs = 10
        if params:
            self.epochs = params.get('epochs', self.epochs)

        super().__init__(model_type, params)

    def _init_lstm_model_functions(self, model_type):
        if model_type != 'lstm':
            raise ValueError(f'Impossible to obtain forecasting strategy for {model_type}')

    def fit(self, train_data: InputData):
        model = fit_lstm(train_data, epochs=self.epochs)
        return model

    def predict(self, trained_model, predict_data: InputData):
        return predict_lstm(trained_model, predict_data)

    def fit_tuned(self, train_data: InputData, iterations: int = 30,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        raise NotImplementedError()


def _rmse_only_last(y_true, y_pred):
    """
    Computes rmse only on the last `prediction_len` values - forecasting
    """
    global forecast_length
    y_true = y_true[:, -forecast_length:]
    y_pred = y_pred[:, -forecast_length:]
    se = tf.keras.backend.square(y_true - y_pred)
    mse = tf.keras.backend.mean(se)
    return tf.keras.backend.sqrt(mse)


def _create_lstm(train_data: InputData):
    reg = 0.001
    shape = train_data.features.shape[-1]
    window_len = train_data.features.shape[-2]

    input_layer = tf.keras.layers.Input(shape=(window_len, shape))
    conv_first = tf.keras.layers.Conv1D(filters=16,
                                        kernel_size=16,
                                        strides=1,
                                        activation='relu',
                                        padding='same',
                                        kernel_regularizer=tf.keras.regularizers.l1(reg))(input_layer)
    conv_second = tf.keras.layers.Conv1D(filters=16,
                                         kernel_size=8,
                                         strides=1,
                                         activation='relu',
                                         padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1(reg))(input_layer)
    concatenation = tf.keras.layers.Concatenate()([input_layer, conv_first, conv_second])

    percent = (train_data.target.max() - train_data.target.min()) / 100
    noise = tf.keras.layers.GaussianNoise(percent)(concatenation)

    lstm_first_layer = tf.keras.layers.LSTM(64, return_sequences=True,
                                            kernel_regularizer=tf.keras.regularizers.l1(reg))(noise)
    lstm_second_layer = tf.keras.layers.LSTM(128, return_sequences=True,
                                             kernel_regularizer=tf.keras.regularizers.l1(reg))(lstm_first_layer)

    concatenation = tf.keras.layers.Concatenate()([concatenation, lstm_second_layer])
    output_layer = tf.keras.layers.TimeDistributed(
        tf.keras.Sequential([
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(train_data.target.shape[-1],
                                  kernel_regularizer=tf.keras.regularizers.l1(reg))
        ]))(concatenation)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


# TODO move hyperparameters to params
def fit_lstm(train_data: InputData, epochs: int = 1):
    global forecast_length

    train_data_3d = _lagged_data_to_3d(train_data)

    ts_length = train_data_3d.features.shape[0]
    # train_data_3d.task.task_params.
    model = _create_lstm(train_data_3d)

    forecast_length = train_data_3d.task.task_params.forecast_length

    model.compile(tf.keras.optimizers.Adam(lr=0.02), loss='mse', metrics=[_rmse_only_last])

    percent = 5 * (train_data_3d.target.max() - train_data_3d.target.min()) / 100
    model.fit(train_data_3d.features, train_data_3d.target, epochs=epochs,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=percent, patience=5),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      monitor='_rmse_only_last', factor=0.2, patience=2, min_delta=0.1, verbose=False),
                  tf.keras.callbacks.TensorBoard(update_freq=ts_length // 10)
              ], verbose=0)

    return model


def predict_lstm(trained_model, predict_data: InputData) -> OutputData:
    window_len, prediction_len = extract_task_param(predict_data.task)

    predict_data_3d = _lagged_data_to_3d(predict_data)

    pred = trained_model.predict(predict_data_3d.features)
    return pred[:, -prediction_len:, 0]


def _lagged_data_to_3d(input_data: InputData) -> InputData:
    transformed_data = copy(input_data)

    # TODO separate proprocessing for exog features
    transformed_data.features = np.asarray(transformed_data.features)[:, :, np.newaxis]
    transformed_data.target = np.asarray(transformed_data.target)[:, :, np.newaxis]

    return transformed_data
