import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from core.models.data import InputData, OutputData


def _rmse_only_last(y_true, y_pred):
    """
    Computes rmse only on the last predicted value - forecasting
    """
    y_true = y_true[:, -1]
    y_pred = y_pred[:, -1]
    se = tf.keras.backend.square(y_true - y_pred)
    mse = tf.keras.backend.mean(se)
    return tf.keras.backend.sqrt(mse)


def _create_lstm(train_data: InputData):
    shape = train_data.features.shape[-1]
    window_len = train_data.features.shape[-2]

    input_layer = tf.keras.layers.Input(shape=(window_len, shape))
    conv1 = tf.keras.layers.Conv1D(filters=32,
                                   kernel_size=window_len//2,
                                   strides=1,
                                   activation='relu',
                                   padding='same')(input_layer)
    conv2 = tf.keras.layers.Conv1D(filters=32,
                                   kernel_size=window_len//4,
                                   strides=1,
                                   activation='relu',
                                   padding='same')(input_layer)
    conc = tf.keras.layers.Concatenate()([input_layer, conv1, conv2])

    percent = (train_data.features.max() - train_data.features.min()) / 100
    noise = tf.keras.layers.GaussianNoise(percent)(conc)

    lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)(noise)
    lstm2 = tf.keras.layers.LSTM(128, return_sequences=True)(lstm1)

    output_layer = tf.keras.layers.TimeDistributed(
        tf.keras.Sequential([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(train_data.target.shape[-1]),
        ]))(lstm2)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def fit_lstm(train_data: InputData):
    model = _create_lstm(train_data)

    # optimizer = Lookahead(RAdam())
    model.compile('sgd', loss='mse', metrics=[_rmse_only_last])

    percent = (train_data.target.max() - train_data.target.min()) / 100
    model.fit(train_data.features, train_data.target, epochs=1000,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(
                      'lstm_model.h5', monitor='loss', save_best_only=True),
                  tf.keras.callbacks.EarlyStopping(
                      monitor='loss', min_delta=percent, patience=20),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      monitor='loss', factor=0.2),
                  tf.keras.callbacks.TensorBoard(
                      update_freq=train_data.features.shape[0] // 10)
              ])

    return model


def predict_lstm(trained_model, predict_data: InputData) -> OutputData:
    # returns only last predicted value
    pred = trained_model.predict(predict_data.features)
    return pred
