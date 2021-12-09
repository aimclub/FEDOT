import numpy as np

from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import str_columns_check
from fedot.core.repository.dataset_types import DataTypesEnum


class DataAnalyser:
    """ Class to analyse data that comes to FEDOT API.
        All methods are inplace to prevent unnecessary copy of large datasets
        It functionality is:
        1) Cut large datasets to prevent memory stackoverflow
        2) Use label encoder with tree models instead OneHot when summary cardinality of categorical features is high
    """
    def __init__(self, safe_mode, preprocessor: ApiDataProcessor):
        self.safe_mode = safe_mode
        self.max_size = 600000
        self.max_cat_cardinality = 1000
        self.data_preprocessor = preprocessor

    def safe_preprocess(self, input_data: InputData):
        """ Preforms preprocessing to preventing crash due of memory stackoverflow if safe_mode on
            :param input_data - data for preprocessing

        """
        if isinstance(input_data, InputData):
            if self.safe_mode:
                self.control_size(input_data)
                self.control_categorical(input_data)

    def control_size(self, input_data: InputData):
        """ Method cut dataset if size of table (N*M) > threshold
            :param input_data - data for preprocessing

        """
        if input_data.data_type == DataTypesEnum.table:
            if input_data.features.shape[0] * input_data.features.shape[1] > self.max_size:
                border = self.max_size // input_data.features.shape[1]
                input_data.idx = input_data.idx[:border]
                input_data.features = input_data.features[:border]
                input_data.target = input_data.target[:border]

    def control_categorical(self, input_data: InputData):
        """ Method use label encoder instead oneHot if summary cardinality > threshold
            :param input_data - data for preprocessing

        """
        categorical_ids, _ = str_columns_check(input_data.features)
        need_label = False
        all_cardinality = 0
        for idx in categorical_ids:
            all_cardinality += np.unique(input_data.features[:, idx].astype(str)).shape[0]
            if all_cardinality > self.max_cat_cardinality:
                need_label = True
                break

        # Здесь сразу препроцессинг в onehot или label
        self.data_preprocessor.preprocessor.encode_data_for_fit(input_data, need_label)
