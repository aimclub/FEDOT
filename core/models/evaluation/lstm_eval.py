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
    reg = 0.001
    shape = train_data.features.shape[-1]
    window_len = train_data.features.shape[-2]

    input_layer = tf.keras.layers.Input(shape=(window_len, shape))

    conv1 = tf.keras.layers.Conv1D(filters=16,
                                   kernel_size=16,
                                   strides=1,
                                   activation='relu',
                                   padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l1(reg))(input_layer)
    conv2 = tf.keras.layers.Conv1D(filters=16,
                                   kernel_size=8,
                                   strides=1,
                                   activation='relu',
                                   padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l1(reg))(input_layer)
    conv3 = tf.keras.layers.Conv1D(filters=16,
                                   kernel_size=4,
                                   strides=1,
                                   activation='relu',
                                   padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l1(reg))(input_layer)
    conc = tf.keras.layers.Concatenate()([input_layer, conv1, conv2, conv3])

    percent = (train_data.target.max() - train_data.target.min()) / 100
    noise = tf.keras.layers.GaussianNoise(percent)(conc)

    lstm1 = tf.keras.layers.LSTM(
        64, return_sequences=True, 
        kernel_regularizer=tf.keras.regularizers.l1(reg))(noise)
    lstm2 = tf.keras.layers.LSTM(
        32, return_sequences=True, 
        kernel_regularizer=tf.keras.regularizers.l1(reg))(lstm1)

    conc = tf.keras.layers.Concatenate()([conc, lstm2])
    output_layer = tf.keras.layers.TimeDistributed(
        tf.keras.Sequential([
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(train_data.target.shape[-1],
                kernel_regularizer=tf.keras.regularizers.l1(reg)),
        ]))(conc)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def fit_lstm(train_data: InputData):
    model = _create_lstm(train_data)

    # optimizer = Lookahead(RAdam())
    model.compile(tf.keras.optimizers.SGD(lr=0.01, momentum=0.9,
                                          nesterov=True), loss='mae', metrics=[_rmse_only_last])

    percent = (train_data.target.max() - train_data.target.min()) / 100
    model.fit(train_data.features, train_data.target, epochs=40,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(
                      'lstm_model.h5', monitor='loss', save_best_only=True),
                  tf.keras.callbacks.EarlyStopping(
                      monitor='loss', min_delta=percent, patience=10),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      monitor='_rmse_only_last', factor=0.2, patience=5, min_delta=0.1, verbose=True),
                  tf.keras.callbacks.TensorBoard(
                      update_freq=train_data.features.shape[0] // 10)
              ])

    return model


def predict_lstm(trained_model, predict_data: InputData) -> OutputData:
    # returns only last predicted value
    pred = trained_model.predict(predict_data.features)
    return pred
