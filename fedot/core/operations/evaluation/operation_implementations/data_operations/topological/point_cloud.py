import numpy as np
import pandas as pd
from ripser import ripser, Rips
from torch.nn.init import sparse

from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.hankel_matrix import \
    HankelMatrix


class TopologicalTransformation:
    """Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
    recorded at equal intervals.

    Args:
        time_series: Time series to be decomposed.
        max_simplex_dim: Maximum dimension of the simplices to be used in the Rips filtration.
        epsilon: Maximum distance between two points to be considered connected by an edge in the Rips filtration.
        persistence_params: ...
        window_length: Length of the window to be used in the rolling window function.

    Attributes:
        epsilon_range (np.ndarray): Range of epsilon values to be used in the Rips filtration.

    """

    def __init__(self,
                 time_series: np.ndarray = None,
                 max_simplex_dim: int = None,
                 epsilon: int = 10,
                 persistence_params: dict = None,
                 window_length: int = None,
                 stride: int = 1):
        self.time_series = time_series
        self.stride = stride
        self.max_simplex_dim = max_simplex_dim
        self.epsilon_range = self.__create_epsilon_range(epsilon)
        self.persistence_params = persistence_params

        if self.persistence_params is None:
            self.persistence_params = {
                'coeff': 2,
                'do_cocycles': False,
                'verbose': False}

        self.__window_length = window_length

    @staticmethod
    def __create_epsilon_range(epsilon):
        return np.array([y * float(1 / epsilon) for y in range(epsilon)])

    @staticmethod
    def __compute_persistence_landscapes(ts):

        N = len(ts)
        I = np.arange(N - 1)
        J = np.arange(1, N)
        V = np.maximum(ts[0:-1], ts[1::])

        # Add vertex birth times along the diagonal of the distance matrix
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        V = np.concatenate((V, ts))

        # Create the sparse distance matrix
        D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]
        dgm0 = dgm0[dgm0[:, 1] - dgm0[:, 0] > 1e-3, :]

        allgrid = np.unique(dgm0.flatten())
        allgrid = allgrid[allgrid < np.inf]

        xs = np.unique(dgm0[:, 0])
        ys = np.unique(dgm0[:, 1])
        ys = ys[ys < np.inf]

    def time_series_to_point_cloud(self,
                                   input_data: np.array = None,
                                   dimension_embed=2) -> np.array:
        """Convert a time series into a point cloud in the dimension specified by dimension_embed.

        Args:
            input_data: Time series to be converted.
            dimension_embed: dimension of Euclidean space in which to embed the time series into by taking
            windows of dimension_embed length, e.g. if the time series is ``[t_1,...,t_n]`` and dimension_embed
            is ``2``, then the point cloud would be ``[(t_0, t_1), (t_1, t_2),...,(t_(n-1), t_n)]``

        Returns:
            Collection of points embedded into Euclidean space of dimension = dimension_embed, constructed
            in the manner explained above.

        """

        if self.__window_length is None:
            self.__window_length = dimension_embed

        trajectory_transformer = HankelMatrix(time_series=input_data,
                                              window_size=self.__window_length,
                                              strides=self.stride)
        return trajectory_transformer.trajectory_matrix

    def point_cloud_to_persistent_cohomology_ripser(self,
                                                    point_cloud: np.array = None,
                                                    max_simplex_dim: int = 1):

        # ensure epsilon_range is a numpy array
        epsilon_range = self.epsilon_range

        # build filtration
        self.persistence_params['maxdim'] = max_simplex_dim
        filtration = Rips(**self.persistence_params)

        if point_cloud is None:
            point_cloud = self.time_series_to_point_cloud()

        # initialize persistence diagrams
        diagrams = filtration.fit_transform(point_cloud)
        # Instantiate persistence landscape transformer
        # plot_diagrams(diagrams)

        # normalize epsilon distance in diagrams so max is 1
        diagrams = [np.array([dg for dg in diag if np.isfinite(dg).all()]) for diag in diagrams]
        diagrams = diagrams / max(
            [np.array([dg for dg in diag if np.isfinite(dg).all()]).max() for diag in diagrams if diag.shape[0] > 0])

        ep_ran_len = len(epsilon_range)

        homology = {dimension: np.zeros(ep_ran_len).tolist() for dimension in range(max_simplex_dim + 1)}

        for dimension, diagram in enumerate(diagrams):
            if dimension <= max_simplex_dim and len(diagram) > 0:
                homology[dimension] = np.array(
                    [np.array(((epsilon_range >= point[0]) & (epsilon_range <= point[1])).astype(int))
                     for point in diagram
                     ]).sum(axis=0).tolist()

        return homology

    def time_series_to_persistent_cohomology_ripser(self,
                                                    time_series: np.array,
                                                    max_simplex_dim: int) -> dict:
        """Wrapper function that takes in a time series and outputs the persistent homology object, along with other
        auxiliary objects.

        Args:
            time_series: Time series to be converted.
            max_simplex_dim: Maximum dimension of the simplicial complex to be constructed.

        Returns:
            Persistent homology object. Dictionary with keys in ``range(max_simplex_dim)`` and, the value ``hom[i]``
            is an array of length equal to ``len(epsilon_range)`` containing the betti numbers of the ``i-th`` homology
            groups for the Rips filtration.

        """

        homology = self.point_cloud_to_persistent_cohomology_ripser(point_cloud=time_series,
                                                                    max_simplex_dim=max_simplex_dim)
        return homology

    def time_series_rolling_betti_ripser(self, ts):

        point_cloud = self.rolling_window(array=ts, window=self.__window_length)
        homology = self.time_series_to_persistent_cohomology_ripser(point_cloud,
                                                                    max_simplex_dim=self.max_simplex_dim)
        df_features = pd.DataFrame(data=homology)
        cols = ["Betti_{}".format(i) for i in range(df_features.shape[1])]
        df_features.columns = cols
        df_features['Betti_sum'] = df_features.sum(axis=1)
        return df_features

    def rolling_window(self, array, window):
        if window <= 0:
            raise ValueError("Window size must be a positive integer.")
        if window > len(array):
            raise ValueError("Window size cannot exceed the length of the array.")
        return np.array([array[i:i + window] for i in range(len(array) - window + 1)])
