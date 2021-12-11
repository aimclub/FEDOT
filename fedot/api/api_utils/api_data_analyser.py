import numpy as np
from typing import Dict, Tuple, Any

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import str_columns_check
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum


class DataAnalyser:
    """
    Class to analyse data that comes to FEDOT API.
    All methods are inplace to prevent unnecessary copy of large datasets
    It functionality is:
    1) Cut large datasets to prevent memory stackoverflow
    2) Use label encoder with tree models instead OneHot when summary cardinality of categorical features is high
    """

    def __init__(self, safe_mode: bool):
        self.safe_mode = safe_mode
        self.max_size = 600000
        self.max_cat_cardinality = 50

    def give_recommendation(self, input_data: InputData) -> Dict:
        """ Gives a recommendation of cutting dataset or using label encoding
            :param input_data - data for preprocessing

            :return dict with str recommendations
        """

        recommendations = {}
        if isinstance(input_data, MultiModalData):
            for data_source_name, values in input_data.items():
                recommendations[data_source_name] = self.give_recommendation(input_data[data_source_name])
        elif isinstance(input_data, InputData) and input_data.data_type == DataTypesEnum.table:
            if self.safe_mode:
                result, border = self.control_size(input_data)
                if result:
                    recommendations['cut'] = {'border': border}
                result = self.control_categorical(input_data)
                if result:
                    recommendations['label_encoded'] = {}
        return recommendations

    def control_size(self, input_data: InputData) -> Tuple[bool, Any]:
        """
        Check if size of table (N*M) > threshold and cutting is needed
        :param input_data - data for preprocessing

        :return (is_cut_needed, border) - is cutting is needed | if yes - border of cutting,

        """
        if input_data.data_type == DataTypesEnum.table:
            if input_data.features.shape[0] * input_data.features.shape[1] > self.max_size:
                border = self.max_size // input_data.features.shape[1]
                return True, border
        return False, None

    def control_categorical(self, input_data: InputData) -> bool:
        """
        Check if use label encoder instead oneHot if summary cardinality > threshold
        :param input_data - data for preprocessing

        """
        categorical_ids, _ = str_columns_check(input_data.features)
        all_cardinality = 0
        need_label = False
        for idx in categorical_ids:
            all_cardinality += np.unique(input_data.features[:, idx].astype(str)).shape[0]
            if all_cardinality > self.max_cat_cardinality:
                need_label = True
                break
        return need_label
