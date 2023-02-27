from functools import partial
from inspect import signature

import numpy as np
from typing import Dict, Tuple, Any

from fedot.core.composer.meta_rules import get_cv_folds_number, get_recommended_preset, \
    get_early_stopping_generations
from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import find_categorical_columns
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME


meta_rules = [get_cv_folds_number,
              get_recommended_preset,
              get_early_stopping_generations]


class InputAnalyser:
    """
    Class to analyse input that comes to FEDOT API: input data and params
    All methods are inplace to prevent unnecessary copy of large datasets
    It functionality is:
    1) Cut large datasets to prevent memory stackoverflow
    2) Use label encoder with tree models instead OneHot when summary cardinality of categorical features is high
    3) Give recommendations according to meta rules for more successful optimization process
    """

    def __init__(self, safe_mode: bool):
        self.safe_mode = safe_mode
        self.max_size = 50000000
        self.max_cat_cardinality = 50

    # TODO implement correct logic to process multimodal data
    def give_recommendations(self, input_data: InputData, input_params: Dict) -> Tuple[Dict, Dict]:
        """
        Gives recommendations for data and input parameters.
        :param input_data: data for preprocessing
        :param input_params: input parameters from FEDOT API
        :return : dict with str recommendations
        """

        recommendations_for_data = self._give_recommendations_for_data(input_data=input_data)

        recommendations_for_params = self._give_recommendations_with_meta_rules(input_data=input_data,
                                                                                input_params=input_params)
        if 'label_encoded' in recommendations_for_data.keys():
            recommendations_for_params['label_encoded'] = recommendations_for_data['label_encoded']

        return recommendations_for_data, recommendations_for_params

    def _give_recommendations_for_data(self, input_data: InputData) -> Dict:
        """
        Gives a recommendation of cutting dataset or using label encoding
        :param input_data: data for preprocessing
        :return : dict with str recommendations
        """

        recommendations_for_data = {}
        if isinstance(input_data, MultiModalData):
            for data_source_name, values in input_data.items():
                recommendations_for_data[data_source_name] = self.give_recommendations(input_data[data_source_name],
                                                                                       data_source_name)
        elif isinstance(input_data, InputData) and input_data.data_type == DataTypesEnum.table:
            if self.safe_mode:
                is_cut_needed, border = self.control_size(input_data)
                if is_cut_needed:
                    recommendations_for_data['cut'] = {'border': border}
                is_label_encoding_needed = self.control_categorical(input_data)
                if is_label_encoding_needed:
                    recommendations_for_data['label_encoded'] = {}
        return recommendations_for_data

    @staticmethod
    def _give_recommendations_with_meta_rules(input_data: InputData, input_params: Dict):
        recommendations = dict()
        for rule in meta_rules:
            if 'input_params' in signature(rule).parameters:
                rule = partial(rule, input_params=input_params)
            if 'input_data' in signature(rule).parameters:
                rule = partial(rule, input_data=input_data)
            cur_recommendation = rule()
            # if there is recommendation to change parameter
            if list(cur_recommendation.values())[0]:
                recommendations.update(rule(input_data=input_data))
        return recommendations

    def control_size(self, input_data: InputData) -> Tuple[bool, Any]:
        """
        Check if size of table (N*M) > threshold and cutting is needed
        :param input_data: data for preprocessing

        :return : (is_cut_needed, border) is cutting is needed | if yes - border of cutting,
        """

        if input_data.data_type == DataTypesEnum.table:
            if input_data.features.shape[0] * input_data.features.shape[1] > self.max_size:
                border = self.max_size // input_data.features.shape[1]
                return True, border
        return False, None

    def control_categorical(self, input_data: InputData) -> bool:
        """
        Check if use label encoder instead oneHot if summary cardinality > threshold

        :param input_data: data for preprocessing
        """

        categorical_ids, _ = find_categorical_columns(input_data.features)
        all_cardinality = 0
        need_label = False
        for idx in categorical_ids:
            all_cardinality += np.unique(input_data.features[:, idx].astype(str)).shape[0]
            if all_cardinality > self.max_cat_cardinality:
                need_label = True
                break
        return need_label
