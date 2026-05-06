import sys
from functools import partial
from itertools import product
from typing import Optional

# import open3d as o3d
import pandas as pd
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import takens_embedding_optimal_parameters
from scipy import stats
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.models.base_extractor import BaseExtractor
from fedot.industrial.core.operation.transformation.data.point_cloud import TopologicalTransformation
from fedot.industrial.core.operation.transformation.representation.topological.topofeatures import \
    PersistenceDiagramsExtractor, TopologicalFeaturesExtractor
from fedot.industrial.core.repository.constanst_repository import PERSISTENCE_DIAGRAM_EXTRACTOR, PERSISTENCE_DIAGRAM_FEATURES

sys.setrecursionlimit(1000000000)


class TopologicalExtractor(BaseExtractor):
    """Class for extracting topological features from time series data.

    Args:
        params: parameters for operation

    Example:
        To use this operation you can create pipeline as follows::

            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from fedot.industrial.tools.loader import DataLoader
            from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('eigen_basis').add_node('topological_extractor').add_node(
                    'rf').build()
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = self.params.get('window_size', 10)
        self.stride = self.params.get('stride', 1)
        self.feature_extractor = TopologicalFeaturesExtractor(
            persistence_diagram_extractor=PERSISTENCE_DIAGRAM_EXTRACTOR,
            persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES
        )
        self.data_transformer = None
        self.save_pcd = False

    def __evaluate_persistence_params(self, ts_data: np.array):
        if self.feature_extractor is None:
            te_dimension, te_time_delay = self.get_embedding_params_from_batch(ts_data=ts_data)

            persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=te_dimension,
                                                                         takens_embedding_delay=te_time_delay,
                                                                         homology_dimensions=(0, 1, 2),
                                                                         parallel=True)

            self.feature_extractor = TopologicalFeaturesExtractor(
                persistence_diagram_extractor=persistence_diagram_extractor,
                persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

    def _generate_vr_mesh(self, pcd):
        # Corresponding matrix of Euclidean pairwise distances
        pairwise_distances = squareform(pdist(pcd))
        # Default parameter for ``metric`` is "euclidean"
        vr_graph = VietorisRipsPersistence(metric="precomputed").fit_transform([pairwise_distances])
        return vr_graph

    def _generate_pcd(self, ts_data, persistence_params):
        window_size_range = list(range(1, 35, 5))
        stride_range = list(range(1, 15, 3))
        list(product(window_size_range, stride_range))
        # for params in pcd_params:
        #     data_transformer = TopologicalTransformation(stride=params[1], persistence_params=persistence_params,
        #                                                  window_length=round(ts_data.shape[0] * 0.01 * params[0]))
        #     point_cloud = data_transformer.time_series_to_point_cloud(input_data=ts_data, use_gtda=True)
        #     # VR_mesh = self._generate_vr_mesh(point_cloud)
        #     for scale in range(1, 15, 3):
        #         numpy2stl(point_cloud,
        #                   f"./stl_scale_{scale}_ws_{params[0]}_stride_{params[1]}.stl",
        #                   max_width=300.,
        #                   max_depth=200.,
        #                   max_height=300.,
        #                   scale=scale,
        #                   min_thickness_percent=0.5,
        #                   solid=False)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(point_cloud)
        #     o3d.io.write_point_cloud(f"./pcd_ws_{params[0]}_stride_{params[1]}.ply", pcd)

    def _generate_features_from_ts(self, ts_data: np.array, persistence_params: dict) -> InputData:
        if self.save_pcd:
            self._generate_pcd(ts_data, persistence_params)
        if self.data_transformer is None:
            self.data_transformer = TopologicalTransformation(
                persistence_params=persistence_params, window_length=round(ts_data.shape[0] * 0.01 * self.window_size))

        point_cloud = self.data_transformer.time_series_to_point_cloud(input_data=ts_data, use_gtda=True)
        topological_features = self.feature_extractor.transform(point_cloud)
        # topological_features = InputData(idx=np.arange(len(topological_features.values)),
        #                                  features=topological_features.values,
        #                                  target='no_target',
        #                                  task='no_task',
        #                                  data_type=DataTypesEnum.table,
        #                                  supplementary_data={'feature_name': topological_features.columns})
        return topological_features.values

    def generate_topological_features(self, ts: np.array, persistence_params: dict = None) -> InputData:
        if persistence_params is not None:
            self.__evaluate_persistence_params(ts)

        if len(ts.shape) == 1:
            aggregation_df = self._generate_features_from_ts(ts, persistence_params)
        else:
            aggregation_df = self._get_feature_matrix(
                partial(self._generate_features_from_ts, persistence_params=persistence_params),
                ts
            )

        return aggregation_df

    def generate_features_from_ts(self, ts_data: np.array, dataset_name: str = None):
        return self.generate_topological_features(ts=ts_data)

    def get_embedding_params_from_batch(self, ts_data: pd.DataFrame, method: str = 'mean') -> tuple:
        """Method for getting optimal Takens embedding parameters.

        Args:
            ts_data: dataframe with time series data
            method: method for getting optimal parameters

        Returns:
            Optimal Takens embedding parameters

        """
        methods = {'mode': self._mode,
                   'mean': np.mean,
                   'median': np.median}

        dim_list, delay_list = list(), list()

        for _ in tqdm(range(len(ts_data)), initial=0, desc='Time series processed: ', unit='ts', colour='black'):
            ts_data = pd.DataFrame(ts_data)
            single_time_series = ts_data.sample(1, replace=False, axis=0).squeeze()
            delay, dim = takens_embedding_optimal_parameters(X=single_time_series,
                                                             max_time_delay=1,
                                                             max_dimension=5,
                                                             n_jobs=-1)
            delay_list.append(delay)
            dim_list.append(dim)

        dimension = int(methods[method](dim_list))
        delay = int(methods[method](delay_list))

        return dimension, delay

    @staticmethod
    def _mode(arr: list) -> int:
        return int(stats.mode(arr)[0][0])
