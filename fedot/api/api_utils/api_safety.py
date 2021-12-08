import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import str_columns_check
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.preprocessing import DataPreprocessor


class ApiSafety:
    def __init__(self, safe_mode):
        self.safe_mode = safe_mode
        self.max_size = 600000
        self.max_cat_cardinality = 1000
        self.label_encoder = None

    def safe_preprocess(self, input_data: InputData):
        if isinstance(input_data, InputData):
            if self.safe_mode:
                self.control_size_inplace(input_data)
            return self.control_categorical(input_data)
        return input_data

    def control_size_inplace(self, input_data: InputData):
        if input_data.data_type == DataTypesEnum.table:
            if input_data.features.shape[0] * input_data.features.shape[1] > self.max_size:
                border = self.max_size // input_data.features.shape[1]
                input_data.idx = input_data.idx[:border]
                input_data.features = input_data.features[:border]
                input_data.target = input_data.target[:border]

    def control_categorical(self, input_data: InputData):
        categorical_ids, _ = str_columns_check(input_data.features)
        need_label = False
        all_cardinality = 0
        for idx in categorical_ids:
            all_cardinality += np.unique(input_data.features[:, idx].astype(str)).shape[0]
            if all_cardinality > self.max_cat_cardinality:
                need_label = True
                break
        if need_label:
            input_data, self.label_encoder = DataPreprocessor.label_encoding(input_data)
        return input_data
