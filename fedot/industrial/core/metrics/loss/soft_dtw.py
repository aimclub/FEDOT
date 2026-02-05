from fedot.industrial.core.architecture.settings.computational import backend_methods as np

from sklearn.metrics.pairwise import euclidean_distances


class SoftDTWLoss:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def _soft_min_argmin(self, a, b, c, gamma):
        """Computes the soft min and argmin of (a, b, c).
        Args:
          a: scalar value.
          b: scalar value.
          c: scalar value.
        Returns:
          softmin, softargmin[0], softargmin[1], softargmin[2]
        """
        a /= -gamma
        b /= -gamma
        c /= -gamma

        max_val = max(max(a, b), c)

        sum_of_probs = 0
        exp_a = np.exp(a - max_val)
        exp_b = np.exp(b - max_val)
        exp_c = np.exp(c - max_val)
        sum_of_probs = exp_a + exp_b + exp_c
        softmin_value = -gamma * (np.log(sum_of_probs) + max_val)
        return softmin_value, exp_a, exp_b, exp_c
        # min_abc = min(a, min(b, c))
        # exp_a = np.exp(min_abc - a)
        # exp_b = np.exp(min_abc - b)
        # exp_c = np.exp(min_abc - c)
        # sum_of_probs = exp_a + exp_b + exp_c
        # exp_a /= sum_of_probs
        # exp_b /= sum_of_probs
        # exp_c /= sum_of_probs
        # val = min_abc - np.log(sum_of_probs)
        # return val, exp_a, exp_b, exp_c

    def _sdtw_C(self, cost_matrix, V, P, gamma):
        """SDTW dynamic programming recursion.
        Args:
          C: cost matrix (input).
          V: intermediate values (output).
          P: transition probability matrix (output).
        """
        size_X, size_Y = cost_matrix.shape

        for i in range(1, size_X + 1):
            for j in range(1, size_Y + 1):
                smin, P[i, j, 0], P[i, j, 1], P[i, j, 2] = \
                    self._soft_min_argmin(
                        V[i - 1, j], V[i - 1, j - 1], V[i, j - 1], gamma=gamma)

                # The cost matrix C is indexed starting from 0.
                V[i, j] = cost_matrix[i - 1, j - 1] + smin
        return cost_matrix, V, P

    def sdtw_C(self, C, gamma=1.0, return_all=True):
        """Computes the soft-DTW value from a cost matrix C.
      Args:
        C: cost matrix, numpy array of shape (size_X, size_Y).
        gamma: regularization strength (scalar value).
        return_all: whether to return intermediate computations.
      Returns:
        sdtw_value if not return_all
        V (intermediate values), P (transition probability matrix) if return_all
      """
        size_X, size_Y = C.shape

        # # Handle regularization parameter 'gamma'.
        # if gamma != 1.0:
        #     C = C / gamma

        # Matrix containing the values of sdtw.
        V = np.zeros((size_X + 1, size_Y + 1))
        V[:, 0] = 1e10
        V[0, :] = 1e10
        V[0, 0] = 0

        # Tensor containing the probabilities of transition.
        P = np.zeros((size_X + 2, size_Y + 2, 3))

        C, V, P = self._sdtw_C(C, V, P, gamma)

        if return_all:
            return V, P
        else:
            return V[size_X, size_Y]

    def sdtw(self, gamma=1.0, return_all=False):
        """Computes the soft-DTW value from time series X and Y.
      The cost is assumed to be the squared Euclidean one.
      Args:
        X: time series, numpy array of shape (size_X, num_dim).
        Y: time series, numpy array of shape (size_Y, num_dim).
        gamma: regularization strength (scalar value).
        return_all: whether to return intermediate computations.
      Returns:
        sdtw_value if not return_all
        V (intermediate values), P (transition probability matrix) if return_all
      """
        cost_matrix = self.squared_euclidean_cost()
        return self.sdtw_C(cost_matrix, gamma=gamma, return_all=return_all)

    def squared_euclidean_cost(self):
        """Computes the squared Euclidean cost.
        """
        return euclidean_distances(self.X, self.Y, squared=True)


if __name__ == "__main__":
    from fedot.industrial.tools.loader import DataLoader
    # Two 3-dimensional time series of lengths 5 and 4, respectively.
    X = np.random.randn(5, 3)
    Y = np.random.randn(5, 3)
    train, test = DataLoader('ECG200').load_data()
    X = train[0].iloc[0:1, ].T.values
    Y = train[0].iloc[1:2, ].T.values
    metric = SoftDTWLoss(X=X, Y=Y)
    dtw_map, dtw_path = metric.sdtw(gamma=0.7)
    path = metric.compute_path(dtw_map)
