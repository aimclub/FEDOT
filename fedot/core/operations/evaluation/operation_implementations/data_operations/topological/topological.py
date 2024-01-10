from abc import ABC
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from scipy.stats import entropy
import pandas as pd
# -*- coding: utf-8 -*-
from gph import ripser_parallel as ripser
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, PersistenceLandscape, BettiCurve
from gtda.homology import VietorisRipsPersistence


class PersistenceDiagramFeatureExtractor(ABC):
    """Abstract class persistence diagrams features extractor.

    """

    def extract_feature_(self, persistence_diagram):
        pass

    def fit_transform(self, x_pd):
        return self.extract_feature_(x_pd)


class PersistenceDiagramsExtractor:
    """Class to extract persistence diagrams from time series.

    Args:
        homology_dimensions: Homology dimensions to compute

    """

    def __init__(self, homology_dimensions: tuple):
        self.homology_dimensions_ = tuple(sorted(homology_dimensions))

    def transform(self, x_embeddings):
        xdgms = ripser(x_embeddings, maxdim=self.homology_dimensions_[-1],
                       thresh=np.inf, coeff=2, metric='euclidean',
                       metric_params=dict(), n_threads=1,
                       collapse_edges=False)["dgms"]
        x_processed = [_xdgms[(_xdgms[:, 0] < _xdgms[:, 1]) & ~np.isinf(_xdgms[:, 1]), :]
                       if len(_xdgms) > 0 else
                       _xdgms
                       for _xdgms in xdgms]
        return x_processed


class TopologicalFeaturesExtractor:
    def __init__(self, persistence_diagram_extractor, persistence_diagram_features):
        self.persistence_diagram_extractor_ = persistence_diagram_extractor
        self.persistence_diagram_features_ = persistence_diagram_features

    def transform(self, x):
        x_pers_diag = self.persistence_diagram_extractor_.transform(x)
        n = self.persistence_diagram_extractor_.homology_dimensions_[1] + 1
        x_transformed = np.zeros((len(self.persistence_diagram_features_), n))
        for dim in self.persistence_diagram_extractor_.homology_dimensions_:
            if len(x_pers_diag[dim]) > 0:
                for j, feature_model in enumerate(self.persistence_diagram_features_.values()):
                    x_transformed[j, dim] = feature_model.fit_transform(x_pers_diag[dim])
        return x_transformed


class HolesNumberFeature(PersistenceDiagramFeatureExtractor):
    def extract_feature_(self, persistence_diagram):
        return persistence_diagram.shape[0]


class MaxHoleLifeTimeFeature(PersistenceDiagramFeatureExtractor):
    def extract_feature_(self, persistence_diagram):
        return np.max(persistence_diagram[:, 1] - persistence_diagram[:, 0])


class RelevantHolesNumber(PersistenceDiagramFeatureExtractor):
    def __init__(self, ratio=0.7):
        super().__init__()
        self.ratio_ = ratio

    def extract_feature_(self, persistence_diagram):
        lifetime = persistence_diagram[:, 1] - persistence_diagram[:, 0]
        return np.sum(lifetime[lifetime > np.max(lifetime) * self.ratio_])


class AverageHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def extract_feature_(self, persistence_diagram):
        return np.mean(persistence_diagram[:, 1] - persistence_diagram[:, 0])


class SumHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def extract_feature_(self, persistence_diagram):
        return np.sum(persistence_diagram[:, 1] - persistence_diagram[:, 0])


class PersistenceEntropyFeature(PersistenceDiagramFeatureExtractor):
    def extract_feature_(self, persistence_diagram):
        return entropy(persistence_diagram[:, 1] - persistence_diagram[:, 0], base=2)


class SimultaneousAliveHolesFeature(PersistenceDiagramFeatureExtractor):
    @staticmethod
    def get_average_intersection_number_(segments):
        n_segments = segments.shape[0]
        s = n_segments
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                if segments[i, 0] <= segments[j, 0] <= segments[i, 1]:
                    s += 1
                else:
                    break
        return s / n_segments

    def extract_feature_(self, persistence_diagram):
        lifetime = persistence_diagram[:, 1] - persistence_diagram[:, 0]
        persistence_diagram = persistence_diagram[lifetime != 0]
        starts, ends = persistence_diagram[:, 0], persistence_diagram[:, 1]
        segments = persistence_diagram[np.lexsort((starts, ends)), :]
        return SimultaneousAliveHolesFeature.get_average_intersection_number_(segments)


class AveragePersistenceLandscapeFeature(PersistenceDiagramFeatureExtractor):
    def extract_feature_(self, persistence_diagram):
        # As practice shows, only 1st layer of 1st homology dimension plays role
        persistence_landscape = PersistenceLandscape(n_jobs=-1).fit_transform([persistence_diagram])[0, 1, 0, :]
        return np.array([np.sum(persistence_landscape) / persistence_landscape.shape[0]])


class BettiNumbersSumFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(BettiNumbersSumFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        betti_curve = BettiCurve(n_jobs=-1).fit_transform([persistence_diagram])[0]
        return np.array([np.sum(betti_curve[i, :]) for i in range(int(np.max(persistence_diagram[:, 2])) + 1)])


class RadiusAtMaxBNFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(RadiusAtMaxBNFeature).__init__()

    def extract_feature_(self, persistence_diagram, n_bins=100):
        betti_curve = BettiCurve(n_jobs=-1, n_bins=n_bins).fit_transform([persistence_diagram])[0]
        max_dim = int(np.max(persistence_diagram[:, 2])) + 1
        max_bettis = np.array([np.max(betti_curve[i, :]) for i in range(max_dim)])
        return np.array(
            [np.where(betti_curve[i, :] == max_bettis[i])[0][0] / (n_bins * max_dim) for i in range(max_dim)])
