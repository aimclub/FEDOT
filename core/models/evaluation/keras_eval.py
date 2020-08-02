from datetime import timedelta
from typing import Optional

import numpy as np
import tensorflow as tf

from core.models.data import InputData, OutputData
from core.models.evaluation.evaluation import EvaluationStrategy
from core.repository.tasks import extract_task_param

forecast_length = 1


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
        ]))(lstm_second_layer)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


# TODO move hyperparameters to params
def fit_lstm(train_data: InputData, epochs: int = 1):
    global forecast_length

    ts_length = train_data.features.shape[0]
    # train_data.task.task_params.
    model = _create_lstm(train_data)

    forecast_length = train_data.task.task_params.forecast_length

    model.compile(tf.keras.optimizers.SGD(lr=0.01, momentum=0.9,
                                          nesterov=True), loss='mae', metrics=[_rmse_only_last])

    percent = 5 * (train_data.target.max() - train_data.target.min()) / 100
    model.fit(train_data.features, train_data.target, epochs=epochs,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=percent, patience=5),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      monitor='_rmse_only_last', factor=0.2, patience=3, min_delta=0.1, verbose=False),
                  tf.keras.callbacks.TensorBoard(update_freq=ts_length // 10)
              ], verbose=0)

    return model


def predict_lstm(trained_model, predict_data: InputData) -> OutputData:
    window_len, prediction_len = extract_task_param(predict_data.task)

    pred = trained_model.predict(predict_data.features)
    pred = np.r_[np.zeros(window_len + prediction_len - 1), pred[:, -1, 0]]
    return pred
