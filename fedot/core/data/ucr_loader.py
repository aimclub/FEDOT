from scipy.io.arff import loadarff
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Union, Tuple

import urllib.request as request
import zipfile
from pathlib import Path
import shutil

import logging

from fedot.core.utils import fedot_project_root, extract_dataset_name_from_url
from fedot.core.data.data_tools import convert_bytes
from fedot.core.data.complex_types import PathType


PROJECT_PATH = fedot_project_root()
DEFAULT_URL = 'http://www.timeseriesclassification.com/aeon-toolkit/'


class TSLoader:
    """
    Class for downloading UCR datasets.

    e.g.
        name = "AbnormalHeartbeat"
        X_train, y_train, X_test, y_test = TSLoader().download_by_url(dataset_name=name)
    """
    logger = logging.getLogger('DataLoader')

    @staticmethod
    def read_tsv_or_csv(dataset_name: str, data_path: PathType, mode: str = 'tsv') -> tuple:
        """Read ``tsv`` or ``csv`` file that contains data for classification experiment.
        Data must be placed in ``data`` folder with ``.tsv``/``csv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data
            mode: ``tsv`` or ``csv`` file format
        Returns:
            tuple: (x_train, x_test) and (y_train, y_test)
        """
        def load_process_data(path_to_dataset: PathType, sep: str):
            data = pd.read_csv(path_to_dataset, sep=sep, header=None)
            features = data.iloc[:, 1:].values
            target = data[0].values
            return features, target

        dataset_dir = os.path.join(data_path, dataset_name)
        if mode not in ['tsv', 'csv']:
            raise ValueError(f'Invalid mode {mode}. Should be one of "tsv" or "csv"')
        separator = '\t' if mode == 'tsv' else ','
        X_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.{mode}', separator)
        X_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.{mode}', separator)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def read_txt_files(dataset_name: str, data_path: str):
        """
        Reads data from ``.txt`` file.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data

        Returns:
            train and test data tuple
        """
        dataset_dir = os.path.join(data_path, dataset_name)
        data_train = np.genfromtxt(dataset_dir + f'/{dataset_name}_TRAIN.txt')
        data_test = np.genfromtxt(dataset_dir + f'/{dataset_name}_TEST.txt')
        x_train, y_train = data_train[:, 1:], data_train[:, 0]
        x_test, y_test = data_test[:, 1:], data_test[:, 0]
        return x_train, y_train, x_test, y_test


    @staticmethod
    def read_arff_files(dataset_name: str, 
                        data_path: PathType) -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]:
        """
        Reads multivariate data from ``.arff`` file

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data

        Returns:
            x_train: train dataframe of shape (n_samples, dim) with pd.Series of shape (ts_length,)
            y_train: train target array of shape (n_samples,)
            x_test: test dataframe of shape (n_samples, dim) with pd.Series of shape (ts_length,)
            y_test: test target array of shape (n_samples,)

        """
        def load_process_data(path_to_dataset):
            data, meta = loadarff(path_to_dataset)
            data_array = np.asarray([data[name] for name in meta.names()])
            features, target = data_array[:-1].T, data_array[-1]
            features = convert_bytes(features)
            target = convert_bytes(target)
            return features, target
        dataset_dir = os.path.join(data_path, dataset_name)
        x_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.arff')
        x_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.arff')

        return x_train, y_train, x_test, y_test


    def read_train_test_files(self,
                              data_path: PathType,
                              dataset_name: str
            ) -> tuple[bool, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        '''
        Reads train and test data from files.

        Args:
            data_path: path to folder with data
            dataset_name: name of dataset

        Returns:
            tuple: train and test data
        '''
        dataset_dir_path = os.path.join(data_path, dataset_name)
        file_path = dataset_dir_path + f'/{dataset_name}_TRAIN'
        self.logger.info(f'Reading data from {dataset_dir_path}')

        if os.path.isfile(file_path + '.tsv'):
            x_train, y_train, x_test, y_test = TSLoader.read_tsv_or_csv(dataset_name, data_path, mode='tsv')
        elif os.path.isfile(file_path + '.csv'):
            x_train, y_train, x_test, y_test = self.read_tsv_or_csv(dataset_name, data_path, mode='csv')
        elif os.path.isfile(file_path + '.txt'):
            x_train, y_train, x_test, y_test = TSLoader.read_txt_files(dataset_name, data_path)
        elif os.path.isfile(file_path + '.arff'):
            x_train, y_train, x_test, y_test = self.read_arff_files(dataset_name, data_path)
        else:
            self.logger.error(f'Data not found in {dataset_dir_path}')
            return None, None, None


        return x_train, y_train, x_test, y_test

    def extract_data(self, dataset_name: str, data_path: PathType):
        """Unpacks data from downloaded file and saves it into Data folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to folder downloaded data

        Returns:
            tuple: train and test data

        """
        try:
            x_train, y_train, x_test, y_test = self.read_train_test_files(
                data_path, dataset_name)

        except Exception as e:
            self.logger.error(f'Error while unpacking data: {e}')
            return None, None

        return x_train, y_train, x_test, y_test

    def download_by_url(self, dataset_name: str = None, url: str = DEFAULT_URL) -> Tuple:
        '''
        Downloads dataset from UCR archive by URL and extracts it into temporary folder.

        Args:
            dataset_name: name of dataset
            url: URL of the dataset

        Returns:
            tuple: train and test data
        '''
        cache_path = os.path.join(PROJECT_PATH, 'temp_cache/')
        os.makedirs(cache_path, exist_ok=True)
        download_path = cache_path + 'downloads/'
        os.makedirs(download_path, exist_ok=True)
        temp_data_path = cache_path + 'temp_data/'
        os.makedirs(temp_data_path, exist_ok=True)

        if dataset_name is not None:
            url = url + f'{dataset_name}.zip'
        else:
            dataset_name = extract_dataset_name_from_url(url)

        request.urlretrieve(url, download_path + f'temp_data_{dataset_name}')
        try:
            zipfile.ZipFile(download_path + f'temp_data_{dataset_name}').extractall(temp_data_path + dataset_name)
        except zipfile.BadZipFile:
            raise FileNotFoundError(f'Cannot extract data: {dataset_name} dataset not found in {url}')
        
        X_train, y_train, X_test, y_test = self.extract_data(dataset_name, temp_data_path)

        shutil.rmtree(cache_path)

        # TODO: implement caching

        return X_train, y_train, X_test, y_test
