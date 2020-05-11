import gc

import h2o
import numpy as np
from h2o import H2OFrame
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier, TPOTRegressor

from core.models.data import InputData
from core.repository.task_types import MachineLearningTasksEnum


def fit_tpot(data: InputData):
    models_hyperparameters = _get_models_hyperparameters()['TPOT']
    estimator = None
    if data.task_type is MachineLearningTasksEnum.classification:
        estimator = TPOTClassifier
    elif data.task_type is MachineLearningTasksEnum.regression:
        estimator = TPOTRegressor

    model = estimator(generations=models_hyperparameters['GENERATIONS'],
                      population_size=models_hyperparameters['POPULATION_SIZE'],
                      verbosity=2,
                      random_state=42,
                      max_time_mins=models_hyperparameters['MAX_RUNTIME_MINS'])

    model.fit(data.features, data.target)

    return model


def predict_tpot(trained_model, predict_data):
    return trained_model.predict_proba(predict_data.features)[:, 1]


def fit_h2o(train_data: InputData):
    model_hyperparameters = _get_models_hyperparameters()['H2O']

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
                   'POPULATION_SIZE': 10
                   }

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
