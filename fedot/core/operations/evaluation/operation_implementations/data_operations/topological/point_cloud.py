import numpy as np

from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table


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

    @staticmethod
    def __create_epsilon_range(epsilon):
        return np.array([y / epsilon for y in range(epsilon)])

    @staticmethod
    def time_series_to_point_cloud(
            time_series: np.array = None,
            dimension_embed=None) -> np.array:
        """Convert a time series into a point cloud in the dimension specified by dimension_embed.

        Args:
            time_series: Time series to be converted.
            dimension_embed: dimension of Euclidean space in which to embed the time series into by taking
            windows of dimension_embed length, e.g. if the time series is ``[t_1,...,t_n]`` and dimension_embed
            is ``2``, then the point cloud would be ``[(t_0, t_1), (t_1, t_2),...,(t_(n-1), t_n)]``

        Returns:
            Collection of points embedded into Euclidean space of dimension = dimension_embed, constructed
            in the manner explained above.

        """

        if dimension_embed is None:
            dimension_embed = int(time_series.shape[0] / 5)

        _, table = ts_to_table(idx=np.arange(time_series.shape[0]), time_series=time_series,
                               window_size=dimension_embed)
        return table
