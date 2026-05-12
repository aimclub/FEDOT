import json
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot.industrial.core.models.classification.stat_domain_clf import StatClassificator
from fedot.industrial.tools.serialisation.path_lib import PATH_TO_DEFAULT_PARAMS


class FrequencyClassificator(StatClassificator):
    """Implementation of a composite of a topological extractor and autoreg atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.transformation_model = params.get(
            'transformation_model', 'fourier_basis')
        self.tuning_params['tuning_iterations'] = 1

    def _define_model(self):
        with open(PATH_TO_DEFAULT_PARAMS) as json_data:
            self.default_operation_params = json.load(json_data)
        self.default_transform_model_params = self.default_operation_params[
            self.transformation_model]
        self.default_channel_model_params = self.default_operation_params[self.channel_model]
        self.default_stat_model_params = self.default_operation_params['quantile_extractor']
        self.model = PipelineBuilder(). \
            add_node(self.transformation_model, params=self.default_transform_model_params). \
            add_node('quantile_extractor', params=self.default_stat_model_params). \
            add_node(self.channel_model,
                     params=self.default_channel_model_params).build()
        return self.model
