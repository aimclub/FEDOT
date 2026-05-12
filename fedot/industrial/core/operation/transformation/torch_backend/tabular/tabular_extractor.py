from typing import Optional
import torch
import numpy as np
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from pymonad.either import Either

from fedot.industrial.core.models.base_extractor import BaseExtractor
from fedot.industrial.core.operation.transformation.data.park_transformation_torch import park_transform_torch
from fedot.industrial.core.repository.constanst_repository import KERNEL_BASELINE_FEATURE_GENERATORS_TORCH
from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels
from fedot.industrial.core.operation.transformation.torch_backend.tabular.reduce_dim import PCA_transformation


class TabularExtractorTorch(BaseExtractor):
    """
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.feature_domain = params.get('feature_domain', 'all')
        self.feature_params = params.get('feature_params', {})
        self.explained_dispersion = params.get('explained_dispersion', .975)
        self.reduce_dimension = params.get('reduce_dimension', True)

        self.repo = IndustrialModels().setup_repository()
        self.custom_tabular_transformation = {'park_transformation':
                                              park_transform_torch}
        self.pca_is_fitted = False
        self.pca = PCA_transformation(self.explained_dispersion)

    def _reduce_dim(self, features: torch.Tensor):
        if self.pca_is_fitted:
            return self.pca.forward(features)
        else:
            self.pca_is_fitted = True
            return self.pca.fit(features).forward(features)

    def _create_from_custom_fg(self, input_data: InputData):
        for model_name, nodes in self.feature_domain.items():
            if model_name.__contains__('custom'):
                transform_method = self.custom_tabular_transformation[nodes[0]]
                ts_representation = transform_method(input_data)
            else:
                model = PipelineBuilder()
                for node in nodes:
                    if isinstance(node, tuple):
                        model.add_node(operation_type=node[0], params=node[1])
                    else:
                        model.add_node(operation_type=node)
                model = model.build()
                ts_representation = model.fit(input_data).predict
            self.feature_list.append(ts_representation)

    def _create_from_default_fg(self, input_data: InputData):
        feature_domain_models = [
            model for model in KERNEL_BASELINE_FEATURE_GENERATORS_TORCH]

        if not self.feature_domain.__contains__('all'):
            feature_domain_models = [model for model in feature_domain_models
                                     if model.__contains__(self.feature_domain)]

        for model_name in feature_domain_models:
            model = KERNEL_BASELINE_FEATURE_GENERATORS_TORCH[model_name]
            model.heads[0].parameters['use_sliding_window'] = self.use_sliding_window
            model = model.build()
            pred = model.fit(input_data).predict
            if isinstance(pred, np.ndarray):
                pred = torch.from_numpy(pred)
            self.feature_list.append(model.fit(input_data).predict)

    def create_feature_matrix(self,
                              feature_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate list of feature tensors into 2D feature matrix.

        Args:
            feature_list: list of torch.Tensor with shape (N, C, T)

        Returns:
            torch.Tensor of shape (N, sum(C*T))
        """
        flattened = [
            torch.from_numpy(x).reshape(x.shape[0], -1)
            for x in feature_list
        ]

        return torch.cat(flattened, dim=1).squeeze()

    def _transform(self, input_data: InputData) -> torch.Tensor:
        """
        Method for feature generation for all series
        """
        feature_list = self.generate_features_from_ts(input_data)
        self.predict = self.create_feature_matrix(feature_list)
        return self.predict if not self.reduce_dimension else self._reduce_dim(self.predict)

    def generate_features_from_ts(self,
                                  input_data: InputData) -> list[torch.Tensor]:
        is_custom_feature_representation = isinstance(
            self.feature_domain, dict)
        self.feature_list = []
        Either(value=input_data,
               monoid=[input_data,
                       is_custom_feature_representation]).either(left_function=self._create_from_default_fg,
                                                                 right_function=self._create_from_custom_fg)
        return self.feature_list
