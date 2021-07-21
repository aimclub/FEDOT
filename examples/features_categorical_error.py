# Standard python libraries
import logging
import os
import time
import requests
logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.INFO)

# Installed libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

from fedot.api.main import Fedot


N_THREADS = 8 # threads cnt for lgbm and linear models
N_FOLDS = 5 # folds cnt for AutoML
RANDOM_STATE = 42 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TARGET_NAME = 'TARGET' # Target column name

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)

DATASET_DIR = './example_data/test_data_files'
DATASET_NAME = 'sampled_app_train.csv'
DATASET_FULLNAME = os.path.join(DATASET_DIR, DATASET_NAME)
DATASET_URL = 'https://raw.githubusercontent.com/sberbank-ai-lab/LightAutoML/master/example_data/test_data_files/sampled_app_train.csv'


if not os.path.exists(DATASET_FULLNAME):
    os.makedirs(DATASET_DIR, exist_ok=True)

    dataset = requests.get(DATASET_URL).text
    with open(DATASET_FULLNAME, 'w') as output:
        output.write(dataset)

data = pd.read_csv(DATASET_FULLNAME)
data.head()

data['BIRTH_DATE'] = (np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))).astype(str)
data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
                    ).astype(str)

data['constant'] = 1
data['allnan'] = np.nan

data['report_dt'] = np.datetime64('2018-01-01')

data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)


# columns_drop = ['NAME_EDUCATION_TYPE', 'BIRTH_DATE', 'EMP_DATE', 'report_dt', 'NAME_CONTRACT_TYPE']

# data_copy = data.copy()
# data_copy = data_copy[list(set(list(data_copy.columns[2:20]) + ['TARGET']))]

# for column in columns_drop:
#     try:
#         data_copy.drop(columns=column, axis=1, inplace=True)
#     except:
#         print(f'{column} not in columns')


numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
categorical_columns = list(set(data.columns) - set(numerical_columns))

data_copy = data[numerical_columns]

train_data, test_data = train_test_split(data_copy,
                                         test_size=TEST_SIZE,
                                         stratify=data[TARGET_NAME],
                                         random_state=RANDOM_STATE)

train_target, test_target = train_data[TARGET_NAME], test_data[TARGET_NAME]


logging.info('Data splitted. Parts sizes: train_data = {}, test_data = {}'
              .format(train_data.shape, test_data.shape))
