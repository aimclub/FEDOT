from typing import Any, Dict, Tuple, Union

from golem.core.log import default_log

from fedot.api.api_utils.recommendation_rules import (
    RecommendationLimits,
    build_recommendation_bundle,
    build_safe_data_recommendations,
    collect_meta_rule_recommendations,
    estimate_size_cut_border,
    should_use_label_encoding,
)
from fedot.core.composer.meta_rules import get_cv_folds_number, get_early_stopping_generations, get_recommended_preset
from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import find_categorical_columns
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum

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
        self._log = default_log('InputAnalyzer')

    def give_recommendations(self, input_data: Union[InputData, MultiModalData], input_params=None) \
            -> Tuple[Dict, Dict]:
        """
        Gives recommendations for data and input parameters.
        :param input_data: data for preprocessing
        :param input_params: input parameters from FEDOT API
        :return : dict with str recommendations
        """

        if input_params is None:
            input_params = dict()

        recommendations_for_data = dict()
        recommendations_for_params = dict()

        if isinstance(input_data, MultiModalData):
            for data_source_name, values in input_data.items():
                recommendations_for_data[data_source_name], recommendations_for_params[data_source_name] = \
                    self.give_recommendations(input_data[data_source_name],
                                              input_params=input_params)
        elif isinstance(input_data, InputData):
            if input_data.data_type in [DataTypesEnum.table, DataTypesEnum.text]:
                recommendation_bundle = build_recommendation_bundle(
                    input_data=input_data,
                    input_params=input_params,
                    safe_mode=self.safe_mode,
                    limits=self._limits(),
                    categorical_detector=find_categorical_columns,
                    meta_rules=meta_rules,
                    log=self._log,
                )
                recommendations_for_data = recommendation_bundle.data
                recommendations_for_params = recommendation_bundle.params
                if 'label_encoded' in recommendations_for_data:
                    self._log.info('Switch categorical encoder to label encoder')

        return recommendations_for_data, recommendations_for_params

    def _give_recommendations_for_data(self, input_data: InputData) -> Dict:
        """
        Gives a recommendation of cutting dataset or using label encoding
        :param input_data: data for preprocessing
        :return : dict with str recommendations
        """

        return build_safe_data_recommendations(
            input_data=input_data,
            safe_mode=self.safe_mode,
            limits=self._limits(),
            categorical_detector=find_categorical_columns,
        )

    def _give_recommendations_with_meta_rules(self, input_data: InputData, input_params: Dict):
        return collect_meta_rule_recommendations(
            input_data=input_data,
            input_params=input_params,
            rules=meta_rules,
            log=self._log,
        )

    def control_size(self, input_data: InputData) -> Tuple[bool, Any]:
        """
        Check if size of table (N*M) > threshold and cutting is needed
        :param input_data: data for preprocessing

        :return : (is_cut_needed, border) is cutting is needed | if yes - border of cutting,
        """

        border = estimate_size_cut_border(input_data, self.max_size)
        return border is not None, border

    def control_categorical(self, input_data: InputData) -> bool:
        """
        Check if use label encoder instead oneHot if summary cardinality > threshold

        :param input_data: data for preprocessing
        """

        return should_use_label_encoding(
            input_data=input_data,
            max_cat_cardinality=self.max_cat_cardinality,
            categorical_detector=find_categorical_columns,
        )

    def _limits(self) -> RecommendationLimits:
        return RecommendationLimits(
            max_size=self.max_size,
            max_cat_cardinality=self.max_cat_cardinality,
        )
