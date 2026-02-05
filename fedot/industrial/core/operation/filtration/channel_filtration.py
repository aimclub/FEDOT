import itertools
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from sklearn.neighbors import NearestCentroid
from sktime.dists_kernels import (
    BasePairwiseTransformerPanel, FlatDist, ScipyDist)

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot.industrial.core.repository.constanst_repository import DISTANCE_METRICS


def _detect_knee_point(values, indices):
    """Find elbow point.The elbow cut method is a method to determine a point in
    a curve where significant change can be observed, e.g., from a steep slope to almost flat curve"""
    n_points = len(values)  # number_of_channels
    # coordinate of each channel projected in chosen centroid
    all_coords = np.vstack((range(n_points), values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point  # line coord from first point to last
    scalar_prod = np.sum(
        vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    # "angle" between each point and line
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # find distance from all points to line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    knee_idx = np.argmax(dist_to_line)
    knee = values[knee_idx]
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem > knee]
    if len(best_dims) == 0:
        return [knee_idx], knee_idx

    return best_dims


def _detect_knee_point_torch(values: torch.Tensor, indices: list) -> list:
    """Find elbow point. The elbow cut method is a method to determine a
    point in a curve where significant change can be observed, e.g., from a
    steep slope to almost flat curve.

    Args:
        values (torch.Tensor): Values that form the curve.
        indices (list): Indices of the values.

    Returns:
        list: Indices of values greater than the knee point.
    """
    device = values.device
    n_points = values.numel()

    x_coords = torch.arange(n_points, device=device, dtype=values.dtype)
    all_coords = torch.stack((x_coords, values), dim=1)
    first_point = all_coords[0]
    last_point = all_coords[-1]
    # direction vector of the line
    line_vec = last_point - first_point
    line_vec_norm = line_vec / torch.norm(line_vec)
    # vectors from first point to all points
    vec_from_first = all_coords - first_point
    # projection length onto the line
    scalar_prod = torch.sum(
        vec_from_first * line_vec_norm, dim=1
    )
    # projection vectors and orthogonal vectors
    vec_from_first_parallel = torch.outer(
        scalar_prod, line_vec_norm
    )
    vec_to_line = vec_from_first - vec_from_first_parallel

    # distance to line
    dist_to_line = torch.norm(vec_to_line, dim=1)
    # max distance to the point (knee)
    knee_idx = torch.argmax(dist_to_line).item()
    knee = values[knee_idx]
    best_dims = [
        idx for val, idx in zip(values.tolist(), indices)
        if val > knee.item()
    ]

    if len(best_dims) == 0:
        return [knee_idx], knee_idx

    return best_dims


class ChannelCentroidFilter(IndustrialCachableOperationImplementation):
    """ChannelCentroidFilter (CCF) transformer to select a subset of channels/variables.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The
    ECS selects the subset of channels using the elbow method, which maximizes the
    distance between the class centroids by aggregating the distance for every
    class pair across each channel.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature. E.g., channel selection = variable selection.

    Attributes:
        distance: sktime pairwise panel transform, str, or callable, optional, default=None
            if panel transform, will be used directly as the distance in the algorithm
            default None = euclidean distance on flattened series, FlatDist(ScipyDist())
            if str, will behave as FlatDist(ScipyDist(distance)) = scipy dist on flat series
            if callable, must be univariate nested_univ x nested_univ -> 2D float np.array

        channels_selected : list of integer
            List of variables/channels selected by the estimator
            integers (iloc reference), referring to variables/channels by order
        distance_frame : np.array
            distance matrix of the class centroids pair and channels.
                ``shape = [n_channels, n_class_centroids_pairs]``

    References:

        ..[1]: Bhaskar Dhariyal et al. "Fast Channel Selection for Scalable Multivariate
        Time Series Classification." AALTD, ECML-PKDD, Springer, 2021
    """

    def __init__(self, params: Optional[OperationParameters] = None):

        super().__init__(params)
        self.distance = params.get('distance', None)  # “manhattan” “chebyshev”
        self.shrink = params.get('shrink', 1e-5)
        self.centroid_metric = params.get('centroid_metric', 'euclidean')
        self.sample_metric = params.get('sample_metric', 'euclidean')
        self.sample_metric = DISTANCE_METRICS[self.sample_metric]
        self.channel_selection_strategy = params.get(
            'selection_strategy', 'sum')
        self.channels_selected = []

        if self.distance is None:
            self.distance_ = FlatDist(ScipyDist())
        elif isinstance(self.distance, str):
            self.distance_ = FlatDist(ScipyDist(metric=self.distance))
        elif isinstance(self.distance, BasePairwiseTransformerPanel):
            self.distance_ = self.distance.clone()
        else:
            self.distance_ = self.distance

    def eval_distance_from_centroid(self, centroid_frame):
        """Create distance matrix."""
        # distance from each class to each without repetitions. Number of pairs
        # is n_cls(n_cls-1)/2
        distance_pair = list(itertools.combinations(
            range(0, centroid_frame.shape[0]), 2))
        # distance_metrics = []
        # for metric in DISTANCE_METRICS.values():
        distance_frame = pd.DataFrame()
        for class_ in distance_pair:
            class_pair = []
            # calculate the distance of centroid here
            for _, (q, t) in enumerate(zip(centroid_frame[class_[0], :],
                                           centroid_frame[class_[1], :], )):
                class_pair.append(self.sample_metric(q, t))
                dict_ = {f"Centroid_{[class_[0]]}_{[class_[1]]}": class_pair}

            distance_frame = pd.concat(
                [distance_frame, pd.DataFrame(dict_)], axis=1)
        # distance_metrics.append(distance_frame)

        return distance_frame

    def create_centroid(self, X, y):
        """Create the centroid for each class."""
        n_samples, n_channels, n_points = X.shape
        centroids = []
        for dim in range(
                n_channels):  # for each channel evaluate distance to class centroid
            # choose channel. Input matrix is n_samples x 1 x n_points
            train = X[:, dim, :]
            clf = NearestCentroid(metric=self.centroid_metric,
                                  shrink_threshold=self.shrink)
            clf.fit(train, y)
            # return matrix n_classes x n_points
            centroids.append(clf.centroids_)

        centroid_frame = np.stack(centroids, axis=1)

        return centroid_frame

    def _channel_sum(self, distance_frame):
        self.distance_frame = pd.Series(distance_frame.sum(axis=1))
        distance = self.distance_frame.sort_values(ascending=False).values
        indices = self.distance_frame.sort_values(ascending=False).index
        channels_selected = _detect_knee_point(distance, indices)[0]
        return channels_selected

    def _channel_pairwise(self, distance_frame):
        self.distance_frame = distance_frame
        channels_selected = []
        for pairdistance in self.distance_frame.items():
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            channels_selected.append(
                _detect_knee_point(distance, indices)[0][0])
        return list(set(channels_selected))

    def __convert_target_for_regression(self, input_data):
        bins = [np.quantile(input_data.target, x)
                for x in np.arange(0, 1, 0.2)]
        labels = [x for x in range(len(bins) - 1)]
        input_data.target = pd.cut(input_data.target,
                                   bins=bins,
                                   labels=labels).codes
        return input_data

    def _transform(self, input_data: InputData):
        """Fit ECS to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        InputData
        """

        have_one_channel = input_data.features.shape[1] == 1
        have_selected_channels = len(self.channels_selected) != 0
        if have_one_channel:
            return input_data.features
        elif have_selected_channels:
            return input_data.features[:, self.channels_selected, :]
        else:
            regression_task = input_data.task.task_type.value == 'regression'
            summation_of_channels = self.channel_selection_strategy == 'sum'

            def get_channels(distance_frame):
                return self._channel_sum(distance_frame) if summation_of_channels \
                    else self._channel_pairwise(distance_frame)

            input_data = Either(input_data,
                                monoid=[input_data, regression_task]). \
                either(left_function=lambda data: data,
                       right_function=lambda data: self.__convert_target_for_regression(data))

            self.channels_selected = Either(value=input_data,
                                            monoid=[input_data, not summation_of_channels]).then(
                lambda data: self.create_centroid(
                    data.features, data.target)).then(lambda centroids_by_channel: self.eval_distance_from_centroid(
                        centroids_by_channel)).then(lambda dist_frame: get_channels(dist_frame)).value

            return input_data.features[:, self.channels_selected, :]
