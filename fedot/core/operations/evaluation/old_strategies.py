import warnings
import gc

import h2o
from h2o import H2OFrame
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier, TPOTRegressor

from copy import copy
from typing import Optional

import numpy as np
import tensorflow as tf

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.repository.tasks import extract_task_param

warnings.filterwarnings("ignore", category=UserWarning)

forecast_length = 1


# TODO inherit this and similar from custom strategy
class KerasForecastingStrategy(EvaluationStrategy):
    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._init_lstm_model_functions(model_type)

        self.epochs = 10
        if params:
            self.epochs = params.get('epochs', self.epochs)

        super().__init__(model_type, params)

    def fit(self, train_data: InputData):
        model = fit_lstm(train_data, epochs=self.epochs)
        return model

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:

        predicted = predict_lstm(trained_operation, predict_data)
        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=predicted,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.ts)
        return converted

    @staticmethod
    def _init_lstm_model_functions(model_type):
        if model_type != 'lstm':
            raise ValueError(
                f'Impossible to obtain forecasting strategy for {model_type}')


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
        tf.keras.Sequential([tf.keras.layers.Dropout(0.25),
                             tf.keras.layers.Dense(train_data.target.shape[-1],
                                                   kernel_regularizer=tf.keras.regularizers.l1(
                                                       reg))]))(concatenation)
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


class AutoMlStrategy(EvaluationStrategy):
    __operations_by_types = {
        'tpot': None,
        'h2o': None}

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation = self._convert_to_operation(operation_type)
        self.params = params
        super().__init__(operation_type, params)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain AutoMl strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))


def fit_tpot(data: InputData, max_run_time_min: int):
    models_hyperparameters = _get_models_hyperparameters(max_run_time_min)['TPOT']
    estimator = None
    if data.task.task_type == TaskTypesEnum.classification:
        estimator = TPOTClassifier
    elif data.task.task_type == TaskTypesEnum.regression:
        estimator = TPOTRegressor

    model = estimator(generations=models_hyperparameters['GENERATIONS'],
                      population_size=models_hyperparameters['POPULATION_SIZE'],
                      verbosity=2,
                      random_state=42,
                      max_time_mins=models_hyperparameters['MAX_RUNTIME_MINS'])

    model.fit(data.features, data.target)

    return model


def predict_tpot_reg(trained_model, predict_data):
    return trained_model.predict(predict_data.features)


def predict_tpot_class(trained_model, predict_data):
    try:
        return trained_model.predict_proba(predict_data.features)[:, 1]
    except AttributeError:
        # sklearn workaround for tpot
        return trained_model.predict(predict_data.features)


def fit_h2o(train_data: InputData, max_run_time_min: int):
    model_hyperparameters = _get_models_hyperparameters(max_run_time_min)['H2O']

    ip, port = _get_h2o_connect_config()

    h2o.init(ip=ip, port=port, name='h2o_server')

    frame = _data_transform(train_data)

    train_frame, valid_frame = frame.split_frame(ratios=[0.85])

    # make sure that your target column is the last one
    target_name = train_frame.columns[-1]
    predictor_names = train_frame.columns.remove(target_name)
    train_frame[target_name] = train_frame[target_name].asfactor()

    model = H2OAutoML(max_models=model_hyperparameters['MAX_MODELS'],
                      seed=1,
                      max_runtime_secs=model_hyperparameters['MAX_RUNTIME_SECS'])
    model.train(x=predictor_names, y=target_name, training_frame=train_frame, validation_frame=valid_frame)
    best_model = model.leader

    return best_model


def predict_h2o(trained_model, predict_data: InputData) -> np.array:
    test_frame = _data_transform(predict_data)

    target_name = test_frame.columns[-1]
    test_frame[target_name] = test_frame[target_name].asfactor()

    prediction_frame = trained_model.predict(test_frame)

    # return list of values like predict_proba[,:1] in sklearn
    prediction_proba_one: list = prediction_frame['p1'].transpose().getrow()

    return np.array(prediction_proba_one)


def _data_transform(data: InputData) -> H2OFrame:
    conc_data = np.concatenate((data.features, data.target.reshape(-1, 1)), 1)
    frame = H2OFrame(python_obj=conc_data)
    return frame


def _get_models_hyperparameters(timedelta: int = 5) -> dict:
    # MAX_RUNTIME_MINS should be equivalent to MAX_RUNTIME_SECS

    tpot_config = {'MAX_RUNTIME_MINS': timedelta,
                   'GENERATIONS': 50,
                   'POPULATION_SIZE': 10}

    h2o_config = {'MAX_MODELS': 20,
                  'MAX_RUNTIME_SECS': timedelta * 60}

    autokeras_config = {'MAX_TRIAL': 10,
                        'EPOCH': 100}

    space_for_mlbox = {
        'ne__numerical_strategy': {"space": [0, 'mean']},

        'ce__strategy': {"space": ["label_encoding", "random_projection", "entity_embedding"]},

        'fs__strategy': {"space": ["variance", "rf_feature_importance"]},
        'fs__threshold': {"search": "choice", "space": [0.1, 0.2, 0.3, 0.4, 0.5]},

        'est__strategy': {"space": ["LightGBM"]},
        'est__max_depth': {"search": "choice", "space": [5, 6]},
        'est__subsample': {"search": "uniform", "space": [0.6, 0.9]},
        'est__learning_rate': {"search": "choice", "space": [0.07]}

    }

    mlbox_config = {'space': space_for_mlbox, 'max_evals': 40}

    config_dictionary = {'TPOT': tpot_config, 'H2O': h2o_config,
                         'autokeras': autokeras_config, 'MLBox': mlbox_config}
    gc.collect()

    return config_dictionary


def _get_h2o_connect_config():
    IP = '127.0.0.1'
    PORT = 8888
    return IP, PORT


class AutoMLEvaluationStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        'tpot': (fit_tpot, predict_tpot_class),
    }

    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._model_specific_fit, self._model_specific_predict = \
            self._init_benchmark_model_functions(model_type)

        self.max_time_min = 5
        if params:
            self.max_time_min = params.get('max_run_time_sec', self.max_time_min * 60) / 60

        super().__init__(model_type, params)

    def _init_benchmark_model_functions(self, model_type):
        if model_type in self._model_functions_by_type.keys():
            return self._model_functions_by_type[model_type]
        else:
            raise ValueError(f'Impossible to obtain benchmark strategy for {model_type}')

    def fit(self, train_data: InputData):
        benchmark_model = self._model_specific_fit(train_data, self.max_time_min)
        return benchmark_model

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        predicted = self._model_specific_predict(trained_operation, predict_data)
        # Wrap prediction as features for next level
        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=predicted,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.table)

        return converted


class AutoMLRegressionStrategy(AutoMLEvaluationStrategy):
    _model_functions_by_type = {
        'tpot': (fit_tpot, predict_tpot_reg),
    }
