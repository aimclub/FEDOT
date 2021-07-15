import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data_split import train_test_data_setup

warnings.filterwarnings('ignore')
np.random.seed(2020)

""" Ниже приведен эксперимент с запуском простой нелинейной модели xgboost
Решается задача регрессии с целью предсказать CRS-R-index. 
"""


def simple_pipeline():
    node_encoder = PrimaryNode('one_hot_encoding')
    node_final = SecondaryNode('xgbreg', nodes_from=[node_encoder])
    pipeline = Pipeline(node_final)
    return pipeline


def prepare_data(path):
    dtype_dic = {'пол': str, 'Этиология': str}
    df = pd.read_excel(path, engine='openpyxl', dtype=dtype_dic)
    df = df.dropna(subset=['Продолжительность нарушения сознания в месяцах'])

    column_names = ['возраст', 'пол', 'Этиология',
                    'Продолжительность нарушения сознания в месяцах',
                    'CRS-балл-1-  при поступлении']
    features_array = np.array(df[column_names])
    target_array = np.array(df['CRS-R- index 1 при поступлении'])

    task = Task(TaskTypesEnum.regression)
    input_data = InputData(idx=np.arange(0, len(features_array)),
                           features=features_array,
                           target=target_array,
                           task=task,
                           data_type=DataTypesEnum.table)

    # Разбиение на обучение и тест
    train_data, test_data = train_test_data_setup(input_data, split_ratio=0.8)
    return train_data, test_data


def fit_and_validate(train_data, test_data):
    single_model = simple_pipeline()
    single_model.fit(train_data)

    predicted = single_model.predict(test_data)
    preds = predicted.predict

    rmse_value = mean_squared_error(test_data.target, preds, squared=False)
    mae_value = mean_absolute_error(test_data.target, preds)

    print(f'RMSE - {rmse_value:.2f}')
    print(f'MAE - {mae_value:.2f}\n')

    plt.scatter(preds, test_data.target)
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Действительные значения')
    plt.show()


if __name__ == '__main__':
    train_data, test_data = prepare_data('med_data.xlsx')
    fit_and_validate(train_data, test_data)




