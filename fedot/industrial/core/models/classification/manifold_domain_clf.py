import json
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot.industrial.core.models.classification.stat_domain_clf import StatClassificator
from fedot.industrial.tools.serialisation.path_lib import PATH_TO_DEFAULT_PARAMS


class ManifoldClassificator(StatClassificator):
    """Implementation of a composite of a manifold extractor(topological, riemann, reccurence)
    and classification atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.transformation_model = params.get('transformation_model', 'riemann_extractor')

    def _define_model(self):
        with open(PATH_TO_DEFAULT_PARAMS) as json_data:
            self.default_operation_params = json.load(json_data)
        self.default_model_params = self.default_operation_params[self.transformation_model]
        self.default_channel_model_params = self.default_operation_params[self.channel_model]
        self.model = PipelineBuilder().add_node(self.transformation_model, params=self.default_model_params). \
            add_node(self.channel_model, params=self.default_channel_model_params).build()
        return self.model
