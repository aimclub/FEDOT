from typing import Optional

import pandas as pd
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from scipy.spatial.distance import cdist
from scipy.stats import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot.industrial.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot.industrial.core.operation.transformation.window_selector import WindowSizeSelector


class FeatureFilter(IndustrialCachableOperationImplementation):
    def __int__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)

    def _init_params(self):
        self.grouping_level = 0.4
        self.fourier_approx = 'exact'
        self.explained_dispersion = 0.9
        self.reduction_dim = None
        self.method_dict = {
            'EigenBasisImplementation': self.filter_dimension_num,
            'FourierBasisImplementation': self.filter_signal,
            'LargeFeatureSpace': self.filter_feature_num}
        self.model = None

    def _transform(self, operation):
        self._init_params()
        if operation.task.task_params is None:
            operation_name = operation.task.task_params
        else:
            operation_name = operation.task.task_params.feature_filter \
                if 'feature_filter' in operation.task.task_params else operation.task.task_params
        if operation_name is None:
            return operation.features
        elif operation_name in self.method_dict.keys():
            method = self.method_dict[operation_name]
            return method(operation)

    def filter_dimension_num(self, data):
        if len(data.features.shape) < 3:
            grouped_components = [self._compute_component_corr(data.features)]
        else:
            grouped_components = list(
                map(self._compute_component_corr, data.features))
        dimension_distrib = [x.shape[0] for x in grouped_components]
        minimal_dim = min(dimension_distrib)
        dominant_dim = stats.mode(dimension_distrib).mode
        if self.reduction_dim is None:
            self.reduction_dim = min(minimal_dim, dominant_dim)
        grouped_predict = [x[:self.reduction_dim, :]
                           for x in grouped_components]
        return np.stack(grouped_predict) if len(
            grouped_predict) > 1 else grouped_predict[0]

    def _compute_component_corr(self, sample):
        component_idx_list = list(range(sample.shape[0]))
        del component_idx_list[0]
        if len(component_idx_list) == 1:
            return sample
        else:
            grouped_predict = sample[0, :].reshape(1, -1)
            tmp = pd.DataFrame(sample[1:, :])
            component_list = []
            correlation_matrix = cdist(
                metric='correlation', XA=tmp.values, XB=tmp.values)
            if (correlation_matrix > self.grouping_level).sum() > 0:
                for index in component_idx_list:
                    if len(component_idx_list) == 0:
                        break
                    else:
                        component_idx_list.remove(index)
                        for correlation_level, component in zip(
                                correlation_matrix, sample[1:, :]):
                            if len(component_idx_list) == 0:
                                break
                            grouped_v = component
                            for cor_level in correlation_level[index:]:
                                if cor_level > self.grouping_level:
                                    component_idx = np.where(
                                        correlation_level == cor_level)[0][0] + 1
                                    grouped_v = grouped_v + \
                                        sample[component_idx, :]
                                    if component_idx in component_idx_list:
                                        component_idx_list.remove(
                                            component_idx)
                                    else:
                                        continue
                            component_list.append(grouped_v)
                    component_list = [x.reshape(1, -1) for x in component_list]
                    grouped_predict = np.concatenate(
                        [grouped_predict, *component_list], axis=0)
                return grouped_predict
            else:
                return sample

    def filter_feature_num(self, data):
        if self.model is None:
            self.model = PipelineNode(
                'pca', params={'n_components': self.explained_dispersion})
            self.model.fit(data)
            prediction = self.model.predict(data)
        else:
            prediction = self.model.predict(data)
        return prediction

    def filter_signal(self, data):
        if self.model is None:
            dominant_window_size = WindowSizeSelector(
                method='dff').get_window_size(data)
            self.model = FourierBasisImplementation(
                params={'threshold': dominant_window_size,
                        'approximation': self.fourier_approx})

        return np.median(data) + self.model.transform(data).features


class FeatureSpaceReducer:
    def __init__(self):
        self.is_fitted = False
        self.feature_mask = None

    def reduce_feature_space(self, features: np.array,
                             var_threshold: float = 0.01,
                             corr_threshold: float = 0.98) -> pd.DataFrame:
        """Method responsible for reducing feature space.

        Args:
            features: dataframe with extracted features.
            corr_threshold: cut-off value for correlation threshold.
            var_threshold: cut-off value for variance threshold.

        Returns:
            Dataframe with reduced feature space.

        """
        features = self._drop_constant_features(features, var_threshold)
        features_new = self._drop_correlated_features(corr_threshold, features)
        self.is_fitted = True
        return features_new

    def _drop_correlated_features(self, corr_threshold, features):
        features_corr = np.corrcoef(features.squeeze().T)
        n_features = features_corr.shape[0]
        identity_matrix = np.eye(n_features)
        features_corr = features_corr - identity_matrix
        correlation_mask = abs(features_corr) > corr_threshold
        correlated_features = list(set(np.where(correlation_mask)[0]))
        percent_of_filtred_feats = (1 - (n_features - len(correlated_features)) / n_features) * 100
        return features if percent_of_filtred_feats > 50 else features

    def _drop_constant_features(self, features, var_threshold):
        try:
            is_2d_data = len(features.shape) <= 2
            variance_reducer = VarianceThreshold(threshold=var_threshold)
            variance_reducer.fit_transform(features.squeeze())
            self.feature_mask = variance_reducer.get_support()
            features = features[:, :, self.feature_mask] if not is_2d_data else features[:, self.feature_mask]
        except ValueError:
            print(
                'Variance reducer has not found any features with low variance')
        return features

    def validate_window_size(self, ts: np.ndarray):
        if self.window_size is None or self.window_size > ts.shape[0] / 2:
            self.logger.info(
                'Window size is not defined or too big (> ts_length/2)')
            self.window_size, _ = WindowSizeSelector(
                time_series=ts).get_window_size()
            self.logger.info(f'Window size was set to {self.window_size}')


class VarianceSelector:
    """
    Class that accepts a dictionary as input, the keys of which are the names of models and the values are arrays
    of data in the np.ndarray format.The class implements an algorithm to determine the "best" set of features and the
    best model in the dictionary.
    """

    def __init__(self, models):
        """
        Initialize the class with the model dictionary.
        """
        self.models = models
        self.principal_components = {}
        self.model_scores = {}

    def get_best_model(self, **model_hyperparams):
        """
        Method to determine the "best" set of features and the best model in the dictionary.
        As an estimation algorithm, use the Principal Component analysis method and the proportion of the explained variance.
        If there are several best models, then a model with a smaller number of principal components and a
        larger value of the explained variance is chosen.
        """
        best_model = None
        best_score = 0
        for model_name, model_data in self.models.items():
            pca = PCA()
            pca.fit(model_data)
            filtered_score = [
                x for x in pca.explained_variance_ratio_ if x > 0.05]
            score = sum(filtered_score)
            self.principal_components.update(
                {model_name: pca.components_[:, :len(filtered_score)]})
            self.model_scores.update(
                {model_name: (score, len(filtered_score))})
            if score > best_score:
                best_score = score
                best_model = model_name
        return best_model

    def transform(self,
                  model_data,
                  principal_components):
        if isinstance(principal_components, str):
            principal_components = self.principal_components[principal_components]
        projected = np.dot(model_data, principal_components)
        return projected

    def select_discriminative_features(self,
                                       model_data,
                                       projected_data,
                                       correlation_level: float = 0.8):
        discriminative_feature = {}
        for PCT in range(projected_data.shape[1]):
            correlation_df = pd.DataFrame.corrwith(
                model_data, pd.Series(projected_data[:, PCT]), axis=0, drop=False)
            discriminative_feature_list = [
                k for k,
                x in zip(
                    correlation_df.index.values,
                    correlation_df.values) if abs(x) > correlation_level]
            discriminative_feature.update(
                {f'{PCT + 1} principal components': discriminative_feature_list})
        return discriminative_feature
