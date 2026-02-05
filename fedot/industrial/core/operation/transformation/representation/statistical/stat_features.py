import warnings

import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy, linregress
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew as skw, kurtosis as kurt
from fedot.industrial.core.architecture.settings.computational import backend_methods as np

warnings.filterwarnings("ignore")


def lambda_less_zero(array: np.array, axis=None) -> int:
    mask = np.array(list(map(lambda x: x < 0.01, array)), dtype=int)
    return np.sum(mask, axis=axis)


def q5(array: np.array, axis=None) -> float:
    return np.quantile(a=array, q=0.05, axis=axis)


def q25(array: np.array, axis=None) -> float:
    return np.quantile(a=array, q=0.25, axis=axis)


def q75(array: np.array, axis=None) -> float:
    return np.quantile(a=array, q=0.75, axis=axis)


def q95(array: np.array, axis=None) -> float:
    return np.quantile(a=array, q=0.95, axis=axis)


def diff(array: np.array, axis=None) -> float:
    return np.diff(a=array, axis=axis, n=len(array) - 1)[0]


# Extra methods for statistical features extraction
def skewness(array: np.array, axis=None) -> float:
    return skw(a=array, axis=axis)


def kurtosis(array: np.array, axis=None) -> float:
    return kurt(a=array, axis=axis)


def n_peaks(array: np.array, axis=None) -> int:
    if axis == 2:
        return None
    else:
        peaks = find_peaks(array)
        return len(peaks[0])


def mean_ptp_distance(array: np.array, axis=None):
    if axis == 2:
        return None
    else:
        peaks, _ = find_peaks(array)
        return np.mean(a=np.diff(a=peaks, axis=axis), axis=axis)


def slope(array: np.array, axis=None):
    if axis == 2:
        return None
    else:
        return linregress(range(len(array)), array).slope


def ben_corr(x, axis=None):
    """Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when
     compared to the Newcomb-Benford's Law distribution [3][4].

     Args:
            x (np.array): time series to calculate the feature of

     Returns:
            float: the value of this feature


     .. math::

         P(d)=\\log_{10}\\left(1+\\frac{1}{d}\\right)

     where :math:`P(d)` is the Newcomb-Benford distribution for :math:`d` that is the leading digit of the number
     {1, 2, 3, 4, 5, 6, 7, 8, 9}.

     .. rubric:: References

     |  [1] A Statistical Derivation of the Significant-Digit Law, Theodore P. Hill, Statistical Science, 1995
     |  [2] The significant-digit phenomenon, Theodore P. Hill, The American Mathematical Monthly, 1995
     |  [3] The law of anomalous numbers, Frank Benford, Proceedings of the American philosophical society, 1938
     |  [4] Note on the frequency of use of the different digits in natural numbers, Simon Newcomb, American Journal of
     |  mathematics, 1881

    """
    if axis == 2:
        return None
    else:
        x = np.asarray(x)

        # retrieve first digit from data
        x = np.array(
            [int(str(np.format_float_scientific(i))[:1])
             for i in np.abs(np.nan_to_num(x))]
        )

        # benford distribution
        benford_distribution = np.array(
            [np.log10(1 + 1 / n) for n in range(1, 10)])

        data_distribution = np.array([(x == n).mean() for n in range(1, 10)])

        # np.corrcoef outputs the normalized covariance (correlation) between benford_distribution and data_distribution.
        # In this case returns a 2x2 matrix, the  [0, 1] and [1, 1] are the values
        # between the two arrays
        return np.corrcoef(benford_distribution, data_distribution)[0, 1]


def interquartile_range(array: np.array, axis=None) -> float:
    return q75(array, axis=axis) - q25(array, axis=axis)


def energy(array: np.array, axis=None) -> float:
    return np.sum(np.power(array, 2), axis=axis) / len(array)


def autocorrelation(array: np.array, axis=None) -> float:
    """Autocorrelation of the time series with its lagged version
    """
    lagged_ts = np.roll(a=array, shift=1, axis=axis)
    corr_coef = np.apply_along_axis(np.corrcoef, axis, lagged_ts) if axis == 2 else np.corrcoef(array, lagged_ts)[0, 1]
    return corr_coef


def zero_crossing_rate(array: np.array, axis=None) -> float:
    """Returns the rate of sign-changes of the time series for a scaled version of it.
    """
    if axis == 2:
        return None
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_array = scaler.fit_transform(array.reshape(-1, 1)).flatten()
        signs = np.sign(scaled_array)
        signs[signs == 0] = -1
        return np.sum((signs[1:] - signs[:-1]) != 0) / len(scaled_array)


def shannon_entropy(array: np.array, axis=None) -> float:
    """Returns the Shannon Entropy of the time series.
    """
    if axis == 2:
        return None
    else:
        p = np.unique(ar=array, return_counts=True)[1] / len(array)
        probs = p * np.log2(p)
        return -np.sum(probs)


def base_entropy(array: np.array, axis=None) -> float:
    """Returns the Shannon Entropy of the time series.
    """
    # Normalize the time series to sum up to 1
    normalized_series = array / np.sum(a=array, axis=axis)
    return entropy(pk=normalized_series, axis=axis)


def ptp_amp(array: np.array, axis=None) -> float:
    """Returns the peak-to-peak amplitude of the time series.
    """
    return np.ptp(a=array, axis=axis)


def crest_factor(array: np.array, axis=None) -> float:
    """Returns the crest factor of the time series.
    """
    return np.max(a=np.abs(array), axis=axis) / np.sqrt(np.mean(np.square(array), axis=axis))


def mean_ema(array: np.array, axis=None) -> float:
    """Returns the exponential moving average of the time series.
    """
    if axis == 2:
        return None
    else:
        span = int(len(array) / 10)
        if span in [0, 1]:
            span = 2
        return pd.Series(array).ewm(span=span).mean().iloc[-1]


def mean_moving_median(array: np.array, axis=None) -> float:
    if axis == 2:
        return None
    else:
        span = int(len(array) / 10)
        if span in [0, 1]:
            span = 2
        return pd.Series(array).rolling(window=span, center=False).median().mean()


def hjorth_mobility(array, axis=None):
    # Compute the first-order differential sequence
    diff_sequence = np.diff(array, axis=axis)
    # Calculate the mean power of the first-order differential sequence
    M2 = np.sum(np.power(diff_sequence, 2), axis=axis) / diff_sequence.shape[axis]
    # Calculate the total power of the time series
    TP = np.sum(np.power(array, 2), axis=axis) / array.shape[axis]
    # Calculate Hjorth mobility
    mobility = np.sqrt(M2 / TP)
    return mobility


def hjorth_complexity(array, axis=None):
    if axis == 2:
        return None
    else:
        # Compute the first-order differential sequence
        diff_sequence = np.diff(a=array, axis=axis)
        # Calculate the mean power of the first-order differential sequence
        M2 = np.sum(np.power(diff_sequence, 2), axis=axis) / diff_sequence.shape[axis]
        # Calculate the total power of the time series
        TP = np.sum(np.power(array, 2), axis=axis) / array.shape[axis]
        # Calculate the fourth central moment of the first-order differential
        # sequence
        try:
            steps = range(1, len(diff_sequence))
            elements_squared_diff = np.array([(diff_sequence[i] - diff_sequence[i - 1]) ** 2 for i in steps])
            M4 = sum(elements_squared_diff) / elements_squared_diff.shape[axis]
        except Exception:
            M4 = 1
        # Calculate Hjorth complexity
        complexity = np.sqrt((M4 * TP) / (M2 * M2))
        # complexity = (M4 * TP) / (M2 * M2)
        return complexity


def hurst_exponent(array, axis=None):
    """ Compute the Hurst Exponent of the time series. The Hurst exponent is used as a measure of long-term memory of
    time series. It relates to the autocorrelations of the time series, and the rate at which these decrease as the
    lag between pairs of values increases. For a stationary time series, the Hurst exponent is equivalent to the
    autocorrelation exponent.

    Args:
        array: Time series

    Returns:
        hurst_exponent: Hurst exponent of the time series

    Notes:
        Author of this function is Xin Liu

    """
    if axis == 2:
        return None
    else:
        X = np.array(array)
        N = X.size
        T = np.arange(1, N + 1)
        Y = np.cumsum(X)
        Ave_T = Y / T

        S_T = np.zeros(N)
        R_T = np.zeros(N)

        for i in range(N):
            S_T[i] = np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = np.ptp(X_T[:i + 1])

        R_S = R_T / S_T
        R_S = np.log(R_S)[1:]
        n = np.log(T)[1:]
        A = np.column_stack((n, np.ones(n.size)))
        [m, c] = np.linalg.lstsq(A, R_S, rcond=None)[0]
        H = m
        return H


def pfd(X, axis=None):
    """The Petrosian fractal dimension (PFD) is a chaotic algorithm used to calculate EEG signal complexity
    Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's difference function.

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if axis == 2:
        return None
    else:
        D = np.diff(X)
        D = D.tolist()
        N_delta = 0  # number of sign changes in derivative of the signal
        for i in range(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(X)
        return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))
