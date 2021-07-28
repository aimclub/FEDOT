import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7, 7

import seaborn as sns
sns.set_theme(style="whitegrid")

import warnings
warnings.filterwarnings("ignore")

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def wrap_into_input(features, target):
    """ Функция переводит numpy массивы с табличными данными (признаки и отклики) в
    датакласс InputData

    :param features: массив с предикторами
    :param target: столбец с откликом
    """
    task = Task(TaskTypesEnum.regression)
    input_data = InputData(idx=np.arange(0, len(features)),
                           features=features,
                           target=target,
                           task=task,
                           data_type=DataTypesEnum.table)
    return input_data


def pipeline_fit_predict(obtained_pipeline, train_input, test_input):
    """ Обучение пайплайна и предсказание в виде numpy массива """
    # Обучение пайплайна
    obtained_pipeline.fit_from_scratch(train_input)

    # Предсказание для обучающей выборки
    predicted_output = obtained_pipeline.predict(train_input)
    train_predicted = np.ravel(np.array(predicted_output.predict))

    # Предсказание для тестовой выборки
    predicted_output = obtained_pipeline.predict(test_input)
    test_predicted = np.ravel(np.array(predicted_output.predict))

    return train_predicted, test_predicted


def get_automl_pipeline(features_array, target_array, learning_time):
    """ Функция для получения пайплайна (композитной модели) на основе фреймворка FEDOT
    :param features: массив с предикторами
    :param target: столбец с откликом
    """
    params = {'max_depth': 5,
              'max_arity': 3,
              'pop_size': 30,
              'num_of_generations': 100,
              'learning_time': learning_time,
              'cv_folds': 5,
              'preset': 'light_tun'}

    fedot_model = Fedot(problem='regression', learning_time=learning_time,
                        seed=20, verbose_level=0,
                        composer_params=params)
    pipeline = fedot_model.fit(features_array, target_array)
    return pipeline


def plot_results(test_actuals, test_preds, labels, target_name, palette="mako"):
    if target_name == 'CRS-балл-1-  при поступлении':
        # Требуется дискретизация отклика
        test_preds = np.rint(test_preds)
    df = pd.DataFrame({'Действительные значения': test_actuals,
                       'Предсказанные значения': test_preds,
                       'Номер разбиения': labels})

    target_range = [min(test_actuals), max(test_actuals)]

    sns.scatterplot(data=df, x="Действительные значения", y="Предсказанные значения",
                    hue="Номер разбиения", palette=palette)
    # Прямая линия, выходящая из координатного угла
    plt.plot(target_range, target_range, c='black')
    plt.ylabel('Предсказанные значения')
    plt.xlabel('Действительные значения')
    plt.show()


def print_metrics(test_actuals, test_preds, train_actuals, train_preds, target_name):
    """ Вывод метрик предсказаний """
    if target_name == 'CRS-балл-1-  при поступлении':
        # Требуется дискретизация отклика
        test_preds = np.rint(test_preds)
        train_preds = np.rint(train_preds)

    arrays = [[train_actuals, train_preds, 'обучающая'], [test_actuals, test_preds, 'тестовая']]
    for actuals, preds, label in arrays:
        rmse_value = mean_squared_error(actuals, preds, squared=False)
        mae_value = mean_absolute_error(actuals, preds)
        r2_value = r2_score(actuals, preds)

        print(f'RMSE {label} выборка: {rmse_value:.2f}')
        print(f'MAE {label} выборка: {mae_value:.2f}')
        print(f'R2 {label} выборка: {r2_value:.2f}\n')


def run_cross_validation(dataframe, features, target, learning_time=2, folds=5,
                         function_for_pipeline=get_automl_pipeline):
    """ Осуществляется запуск кросс валидации для AutoML модели

    :param dataframe: таблица с данными
    :param features: названия столбцов-предикторов
    :param target: названия столбца-отклика
    :param learning_time: количество времени в минутах, выделяемое для поиска структуры пайплайна
    :param folds: количество подвыборок, на которых проводится кросс-валидаци
    :param function_for_pipeline: функция для генерации пайплайна
    """
    # Определяем объект, который будет разбивать матрицы
    kf = KFold(n_splits=folds)

    # По названиям столбцов получаем массивы
    features_array = np.array(dataframe[features])
    target_array = np.array(dataframe[target])

    # При помощи AutoML алгоритма получаем композитную модель
    obtained_pipeline = function_for_pipeline(features_array, target_array, learning_time)

    # Разбиваем данные на train и test
    train_actuals = []
    train_preds = []
    test_actuals = []
    test_preds = []
    labels = []
    # Номер фолда
    i = 1
    for train_index, test_index in kf.split(target_array):
        # Признаки
        train_features = features_array[train_index, :]
        test_features = features_array[test_index, :]

        # Отклики
        train_target = target_array[train_index]
        test_target = target_array[test_index]

        # Оборачиваем данные в InputData формат
        train_input = wrap_into_input(train_features, train_target)
        test_input = wrap_into_input(test_features, test_target)

        # Получим предсказание модели
        train_predicted, test_predicted = pipeline_fit_predict(obtained_pipeline, train_input, test_input)

        train_actuals.extend(train_target)
        train_preds.extend(train_predicted)

        test_actuals.extend(test_target)
        test_preds.extend(test_predicted)
        labels.extend([i] * len(test_predicted))
        i += 1

    return obtained_pipeline, test_actuals, test_preds, train_actuals, train_preds, labels
