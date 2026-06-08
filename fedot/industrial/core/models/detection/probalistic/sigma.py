from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from numpy import array, eye, zeros
from scipy.linalg import block_diag, cholesky


class MerweScaledSigmaPoints:
    """
    Generates sigma points and weights according to Van der Merwe's
    2004 dissertation[1] for the UnscentedKalmanFilter class. It
    parametizes the sigma points using alpha, beta, kappa terms, and
    is the version seen in most publications.
    Unless you know better, this should be your default choice.
    Parameters
    ----------
    n : int
        Dimensionality of the state. 2n+1 weights will be generated.
    alpha : float
        Determins the spread of the sigma points around the mean.
        Usually a small positive value (1e-3) according to [3].
    beta : float
        Incorporates prior knowledge of the distribution of the mean. For
        Gaussian x beta=2 is optimal, according to [3].
    kappa : float, default=0.0
        Secondary scaling parameter usually set to 0 according to [4],
        or to 3-n according to [5].
    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.
        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.
    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.
    Attributes
    ----------
    Wm : np.array
        weight for each sigma point for the mean
    Wc : np.array
        weight for each sigma point for the covariance
    Examples
    --------
    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    References
    ----------
    .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)
    """

    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        # pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
        Parameters
        ----------
        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
        Returns
        -------
        sigmas : np.array, of size (2n+1, n)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.
            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n) * P)

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x, U[k])

        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
        """

        n = self.n
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + \
            (1 - self.alpha ** 2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)


class JulierSigmaPoints(object):
    """
    Generates sigma points and weights according to Simon J. Julier
    and Jeffery K. Uhlmann's original paper[1]. It parametizes the sigma
    points using kappa.
    Parameters
    ----------
    n : int
        Dimensionality of the state. 2n+1 weights will be generated.
    kappa : float, default=0.
        Scaling factor that can reduce high order errors. kappa=0 gives
        the standard unscented filter. According to [Julier], if you set
        kappa to 3-dim_x for a Gaussian x you will minimize the fourth
        order errors in x and P.
    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.
        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.
    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
    Attributes
    ----------
    Wm : np.array
        weight for each sigma point for the mean
    Wc : np.array
        weight for each sigma point for the covariance
    References
    ----------
    .. [1] Julier, Simon J.; Uhlmann, Jeffrey "A New Extension of the Kalman
        Filter to Nonlinear Systems". Proc. SPIE 3068, Signal Processing,
        Sensor Fusion, and Target Recognition VI, 182 (July 28, 1997)
   """

    def __init__(self, n, kappa=0., sqrt_method=None, subtract=None):

        self.n = n
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        r""" Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        kappa is an arbitrary constant. Returns sigma points.
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
        Parameters
        ----------
        x : array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
        kappa : float
            Scaling factor.
        Returns
        -------
        sigmas : np.array, of size (2n+1, n)
            2D array of sigma points :math:`\chi`. Each column contains all of
            the sigmas for one dimension in the problem space. They
            are ordered as:
            .. math::
                :nowrap:
                \begin{eqnarray}
                  \chi[0]    = &x \\
                  \chi[1..n] = &x + [\sqrt{(n+\kappa)P}]_k \\
                  \chi[n+1..2n] = &x - [\sqrt{(n+\kappa)P}]_k
                \end{eqnarray}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)  # dimension of problem

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        sigmas = np.zeros((2 * n + 1, n))

        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = self.sqrt((n + self.kappa) * P)

        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x, U[k])
        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the unscented Kalman filter. In this
        formulation the weights for the mean and covariance are the same.
        """

        n = self.n
        k = self.kappa

        self.Wm = np.full(2 * n + 1, .5 / (n + k))
        self.Wm[0] = k / (n + k)
        self.Wc = self.Wm


class SimplexSigmaPoints(object):
    """
    Generates sigma points and weights according to the simplex
    method presented in [1].
    Parameters
    ----------
    n : int
        Dimensionality of the state. n+1 weights will be generated.
    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm
        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.
    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.
    Attributes
    ----------
    Wm : np.array
        weight for each sigma point for the mean
    Wc : np.array
        weight for each sigma point for the covariance
    References
    ----------
    .. [1] Phillippe Moireau and Dominique Chapelle "Reduced-Order
           Unscented Kalman Filtering with Application to Parameter
           Identification in Large-Dimensional Systems"
           DOI: 10.1051/cocv/2010006
    """

    def __init__(self, n, alpha=1, sqrt_method=None, subtract=None):
        self.n = n
        self.alpha = alpha
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return self.n + 1

    def sigma_points(self, x, P):
        """
        Computes the implex sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
        Parameters
        ----------
        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
        Returns
        -------
        sigmas : np.array, of size (n+1, n)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.
            Ordered by Xi_0, Xi_{1..n}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])
        x = x.reshape(-1, 1)

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        U = self.sqrt(P)

        lambda_ = n / (n + 1)
        Istar = np.array(
            [[-1 / np.sqrt(2 * lambda_), 1 / np.sqrt(2 * lambda_)]])

        for d in range(2, n + 1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1. / np.sqrt(
                lambda_ * d * (d + 1))  # pylint: disable=unsubscriptable-object
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[
                np.c_[
                    Istar,
                    np.zeros(
                        (Istar.shape[0]))],
                row]  # pylint: disable=unsubscriptable-object

        I = np.sqrt(n) * Istar
        scaled_unitary = (U.T).dot(I)

        sigmas = self.subtract(x, -scaled_unitary)
        return sigmas.T

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter. """

        n = self.n
        c = 1. / (n + 1)
        self.Wm = np.full(n + 1, c)
        self.Wc = self.Wm


def order_by_derivative(Q, dim, block_size):
    """
    Given a matrix Q, ordered assuming state space
        [x y z x' y' z' x'' y'' z''...]
    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']
    This works for any covariance matrix or state transition function
    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder
    dim : int >= 1
       number of independent state variables. 3 for x, y, z
    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')
    """

    N = dim * block_size

    D = zeros((N, N))

    Q = array(Q)
    for i, x in enumerate(Q.ravel()):
        f = eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix:ix + block_size, iy:iy + block_size] = f

    return D


def Q_discrete_white_noise(
        dim,
        dt=1.,
        var=1.,
        block_size=1,
        order_by_dim=True):
    """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.
    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.
    Parameters
    -----------
    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)
    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations
    var : float, default=1.0
        variance in the noise
    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.
    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)
        [x x' x'' y y' y'']
        whereas `False` interleaves the dimensions
        [x y z x' y' z' x'' y'' z'']

    """

    if dim not in [2, 3, 4]:
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25 * dt ** 4, .5 * dt ** 3],
             [.5 * dt ** 3, dt ** 2]]
    elif dim == 3:
        Q = [[.25 * dt ** 4, .5 * dt ** 3, .5 * dt ** 2],
             [.5 * dt ** 3, dt ** 2, dt],
             [.5 * dt ** 2, dt, 1]]
    else:
        Q = [[(dt ** 6) / 36, (dt ** 5) / 12, (dt ** 4) / 6, (dt ** 3) / 6],
             [(dt ** 5) / 12, (dt ** 4) / 4, (dt ** 3) / 2, (dt ** 2) / 2],
             [(dt ** 4) / 6, (dt ** 3) / 2, dt ** 2, dt],
             [(dt ** 3) / 6, (dt ** 2) / 2, dt, 1.]]

    if order_by_dim:
        return block_diag(*[Q] * block_size) * var
    return order_by_derivative(array(Q), dim, block_size) * var
