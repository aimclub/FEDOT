import sys
from copy import deepcopy
from functools import partial
from math import log

from numpy import dot, eye, zeros
from sklearn.preprocessing import MinMaxScaler

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.models.detection.probalistic.sigma import MerweScaledSigmaPoints
from fedot.industrial.core.operation.transformation.data.hankel import get_x_y_pairs


def reshape_z(z, dim_z, ndim):
    """ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError(
            "z (shape {}) must be convertible to shape ({}, 1)".format(
                z.shape, dim_z)
        )

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z


class AbstractKalmanFilter:
    def __init__(self, model_hyperparams: dict):
        self.model_hyperparams = model_hyperparams
        # Only computed if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None
        # self.state = zeros((self.input_dim, 1))  # state
        self.state = None  # state
        self.control_transition_matrix = None  # B
        self.state_transition_matrix = None  # F
        self.inv = np.linalg.inv

    def _init_kalman_params(self, input_data):
        self.input_dim = input_data.shape[0]
        self.output_dim = self.input_dim

        self._I = np.eye(self.input_dim)  # identity matrix
        self.uncertainty_covariance = 100. * eye(self.input_dim)  # P
        self.process_uncertainty = 3. * eye(self.input_dim)  # R

        self.measurement_function = eye(self.input_dim)  # H
        self.measurement_uncertainty = 5. * eye(self.input_dim)  # Q
        self.fading_memory_control = 1.  # alpha
        self.process_measurement_cross_correlation = np.zeros(
            (self.input_dim, self.output_dim))  # M
        self.z = np.array([[None] * self.output_dim]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.kalman_gain = np.zeros((self.input_dim, self.output_dim))  # K
        self.residual = zeros((self.output_dim, 1))
        self.system_uncertainty = np.zeros(
            (self.output_dim, self.output_dim))  # S
        self.inversed_system_uncertainty = np.zeros(
            (self.output_dim, self.output_dim))  # SI

    def _fit_model_dynamic(self, train_features: np.ndarray):
        self._init_kalman_params(train_features)
        self.state = train_features
        self.state_mean = np.mean(train_features, axis=1)
        self.uncertainty_covariance = np.cov(train_features)
        self.train_features, self.target = get_x_y_pairs(
            train=train_features, train_periods=1, prediction_periods=1)
        self.measurement_function = MinMaxScaler(feature_range=(0, 1))
        self.measurement_function.fit(self.train_features)
        self.train_features = self.measurement_function.transform(
            self.train_features)
        self.target = self.measurement_function.transform(self.target)
        self.state_transition_matrix = TimeSeriesClassifier(
            model_hyperparams=self.model_hyperparams)
        self.state_transition_matrix.fit(
            train_features=self.train_features, train_target=self.target)

    def fit(
            self,
            train_features,
            fit_settings: dict = {
            'control vector': None}):
        self._fit_model_dynamic(train_features=train_features)
        # x = Fx + Bu
        if self.state is None:
            self.state = train_features
        if self.control_transition_matrix is not None and fit_settings[
                'control vector'] is not None:
            self.state = dot(self.state_transition_matrix,
                             self.state) + dot(self.control_transition_matrix,
                                               fit_settings['control vector'])
        else:
            self.state = dot(self.state_transition_matrix, self.state)

        # P = FPF' + Q
        self.uncertainty_covariance = self.fading_memory_control * dot(
            dot(self.state_transition_matrix, self.uncertainty_covariance),
            self.state_transition_matrix.T) + self.process_uncertainty

        # save prior
        self.state_prior = self.state.copy()
        self.uncertainty_covariance_prior = self.uncertainty_covariance.copy()
        return self.state, self.uncertainty_covariance

    def predict(self, test_features, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        test_features : (self.output_dim, 1): array_like
            measurement for this update. z can be a scalar if self.output_dim is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if test_features is None:
            self.z = np.array([[None] * self.output_dim]).T
            self.state_post = self.state.copy()
            self.uncertainty_covariance_post = self.uncertainty_covariance.copy()
            self.y = zeros((self.output_dim, 1))
            return

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.residual = test_features - \
            self.state_transition_matrix.predict(self.state)

        # common subexpression for speed
        PHT = dot(self.uncertainty_covariance, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.system_uncertainty = dot(
            self.state_transition_matrix, PHT) + self.measurement_uncertainty
        self.inversed_system_uncertainty = self.inv(self.system_uncertainty)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.kalman_gain = dot(PHT, self.inversed_system_uncertainty)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.state = self.state + dot(self.kalman_gain, self.residual)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.kalman_gain, self.measurement_function)
        self.uncertainty_covariance = dot(dot(I_KH,
                                              self.uncertainty_covariance),
                                          I_KH.T) + dot(dot(self.kalman_gain,
                                                            self.measurement_uncertainty),
                                                        self.kalman_gain.T)

        # save measurement and posterior state
        self.measurement = deepcopy(test_features)
        self.state_post = self.state.copy()
        self.uncertainty_covariance_post = self.uncertainty_covariance.copy()

        return self.state, self.uncertainty_covariance

    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        z = reshape_z(z, self.self.output_dim, self.state.ndim)
        return z - dot(self.H, self.state_prior)

    def measurement_of_state(self, x):
        """
        Helper function that converts a state into a measurement.
        Parameters
        ----------
        x : np.array
            kalman state vector
        Returns
        -------
        z : (self.output_dim, 1): array_like
            measurement for this update. z can be a scalar if self.output_dim is 1,
            otherwise it must be convertible to a column vector.
        """

        return dot(self.measurement_function, x)


class UnscentedKalmanFilter(AbstractKalmanFilter):
    def __init__(self, model_hyperparams: dict):
        super().__init__(model_hyperparams)
        self.sigma_points = partial(
            MerweScaledSigmaPoints, alpha=.1, beta=2., kappa=-1)

    def fit(
            self,
            train_features,
            fit_settings: dict = {
            'control vector': None}):
        self._fit_model_dynamic(train_features=train_features)
        # calculate sigma points for given mean and covariance
        self.sigma_distribution = self.sigma_points(self.input_dim)
        self.sigmas_f = zeros(
            (self.sigma_distribution.num_sigmas(), self.input_dim))
        self.sigmas_h = zeros(
            (self.sigma_distribution.num_sigmas(), self.output_dim))

    def update(self):
        r"""
        Performs the predict step of the UKF. On return, self.state and
        self.P contain the predicted state (x) and covariance (P). '
        Important: this MUST be called before update() is called for the first
        time.
        """

        sigmas = self.sigma_distribution.sigma_points(
            self.state_mean, self.uncertainty_covariance)
        sigmas = self.measurement_function.transform(sigmas)
        for i, s in enumerate(sigmas):
            if len(s.shape) < 2:
                s = s.reshape(1, -1)
            self.sigmas_f[i] = self.state_transition_matrix.predict(s)
        self.sigmas_f = self.measurement_function.inverse_transform(
            self.sigmas_f)

        # pass sigmas through the unscented transform to compute prior
        self.state_mean, self.uncertainty_covariance = self.unscented_transform(sigmas=self.sigmas_f,
                                                                                Wm=self.sigma_distribution.Wm,
                                                                                Wc=self.sigma_distribution.Wc,
                                                                                noise_cov=self.process_uncertainty,
                                                                                mean_fn=None,
                                                                                residual_fn=np.subtract)

        # update sigma points to reflect the new variance of the points
        self.sigmas_f = self.sigma_distribution.sigma_points(
            self.state_mean, self.uncertainty_covariance)

        # save prior
        self.state_prior = np.copy(self.state_mean)
        self.uncertainty_covariance_prior = np.copy(
            self.uncertainty_covariance)

    def _predict(self, test_features,
                 measurement_uncertainty=None,
                 measurement_function=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        test_features : measurement for this predict.
        measurement_uncertainty : np.array, scalar, or callable
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        measurement function : np.array, or callable
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """
        # update prior_mean and prior_covariance
        self.update()
        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        if measurement_function is None:
            self.sigmas_h = self.sigmas_f
        else:
            self.sigmas_h = dot(self.measurement_function, self.sigmas_f)

        self.sigmas_h = np.atleast_2d(self.sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        measurement_mean, self.system_uncertainty = self.unscented_transform(sigmas=self.sigmas_h,
                                                                             Wm=self.sigma_distribution.Wm,
                                                                             Wc=self.sigma_distribution.Wc,
                                                                             noise_cov=self.measurement_uncertainty,
                                                                             mean_fn=None,
                                                                             residual_fn=np.subtract)

        self.inversed_system_uncertainty = self.inv(self.system_uncertainty)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(
            self.state_mean, measurement_mean, self.sigmas_f, self.sigmas_h)

        self.kalman_gain = dot(
            Pxz, self.inversed_system_uncertainty)  # Kalman gain
        self.residual = np.subtract(
            test_features, measurement_mean.reshape(-1, 1))
        weighted_residual = dot(self.kalman_gain, self.residual)  # residual
        predicted_state = test_features + weighted_residual
        weighted_mean = np.mean(weighted_residual, axis=1)
        # update Gaussian state estimate (x, P)
        self.state_mean = np.add(self.state_mean, weighted_mean)
        self.uncertainty_covariance = self.uncertainty_covariance - \
            dot(self.kalman_gain, dot(self.system_uncertainty, self.kalman_gain.T))

        # save measurement and posterior state
        self.measurement = deepcopy(test_features)
        self.state_post = self.state.copy()
        self.uncertainty_covariance_post = self.uncertainty_covariance.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None
        return self.residual, predicted_state

    def predict(self, test_features,
                measurement_uncertainty=None,
                measurement_function=None):

        if isinstance(test_features, list):
            list_of_residuals, list_of_states = [], []
            for window_slice in test_features:
                try:
                    current_residual, current_state = self._predict(
                        window_slice, measurement_uncertainty, measurement_function)
                    list_of_residuals.append(current_residual)
                    list_of_states.append(current_state)
                except Exception:
                    _ = 1
            return list_of_residuals, list_of_states
        else:
            return self._predict(
                test_features,
                measurement_uncertainty,
                measurement_function)

    @staticmethod
    def unscented_transform(sigmas,
                            Wm,
                            Wc,
                            noise_cov=None,
                            mean_fn=None,
                            residual_fn=None):
        r"""
        Computes unscented transform of a set of sigma points and weights.
        returns the mean and covariance in a tuple.
        This works in conjunction with the UnscentedKalmanFilter class.
        Parameters
        ----------
        sigmas: ndarray, of size (n, 2n+1)
            2D array of sigma points.
        Wm : ndarray [# sigmas per dimension]
            Weights for the mean.
        Wc : ndarray [# sigmas per dimension]
            Weights for the covariance.
        noise_cov : ndarray, optional
            noise matrix added to the final computed covariance matrix.
        mean_fn : callable (sigma_points, weights), optional
            Function that computes the mean of the provided sigma points
            and weights. Use this if your state variable contains nonlinear
            values such as angles which cannot be summed.
            .. code-block:: Python
                def state_mean(sigmas, Wm):
                    x = np.zeros(3)
                    sum_sin, sum_cos = 0., 0.
                    for i in range(len(sigmas)):
                        s = sigmas[i]
                        x[0] += s[0] * Wm[i]
                        x[1] += s[1] * Wm[i]
                        sum_sin += sin(s[2])*Wm[i]
                        sum_cos += cos(s[2])*Wm[i]
                    x[2] = atan2(sum_sin, sum_cos)
                    return x
        residual_fn : callable (x, y), optional
            Function that computes the residual (difference) between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles (359-1 degreees is 2, not 358). x and y
            are state vectors, not scalars.
            .. code-block:: Python
                def residual(a, b):
                    y = a[0] - b[0]
                    y = y % (2 * np.pi)
                    if y > np.pi:
                        y -= 2*np.pi
                    return y
        Returns
        -------
        x : ndarray [dimension]
            Mean of the sigma points after passing through the transform.
        P : ndarray
            covariance of the sigma points after passing through the transform.
        """

        kmax, n = sigmas.shape

        try:
            if mean_fn is None:
                # new mean is just the sum of the sigmas * weight
                x = np.dot(Wm, sigmas)  # dot = \Sigma^n_1 (W[k]*Xi[k])
            else:
                x = mean_fn(sigmas, Wm)
        except BaseException:
            print(sigmas)
            raise

        # new covariance is the sum of the outer product of the residuals
        # times the weights

        # this is the fast way to do this - see 'else' for the slow way
        if residual_fn is np.subtract or residual_fn is None:
            y = sigmas - x[np.newaxis, :]
            P = np.dot(y.T, np.dot(np.diag(Wc), y))
        else:
            P = np.zeros((n, n))
            for k in range(kmax):
                y = residual_fn(sigmas[k], x)
                P += Wc[k] * np.outer(y, y)

        if noise_cov is not None:
            P += noise_cov

        return x, P

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = np.subtract(sigmas_f[i], x)
            dz = np.subtract(sigmas_h[i], z)
            Pxz += self.sigma_distribution.Wc[i] * np.outer(dx, dz)
        return Pxz
