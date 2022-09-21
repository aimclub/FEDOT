import itertools
import numpy as np
from scipy.linalg import toeplitz
import pyemd


def emd(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd_value = pyemd.emd(x, y, distance_mat)
    return emd_value


def l2(x, y):
    dist = np.linalg.norm(x - y, 2)
    return dist


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    ''' Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation
    '''
    emd_val = emd(x, y, distance_scaling=distance_scaling)
    return np.exp(-emd_val ** 2 / (2 * sigma ** 2))


def gaussian(x, y, sigma=1.0):
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist ** 2 / (2 * sigma ** 2))


def discrepancy(samples1, samples2, kernel, *args, **kwargs) -> float:
    d = sum(kernel(s1, s2, *args, **kwargs) for s1, s2 in itertools.product(samples1, samples2))
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel=gaussian_emd, normalize=True, *args, **kwargs) -> float:
    ''' MMD between two samples
    '''
    # normalize histograms into pmf
    if normalize:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    return discrepancy(samples1, samples1, kernel, *args, **kwargs) + \
           discrepancy(samples2, samples2, kernel, *args, **kwargs) - \
           2 * discrepancy(samples1, samples2, kernel, *args, **kwargs)


def compute_emd(samples1, samples2, kernel, normalize=True, *args, **kwargs) -> float:
    ''' EMD between average of two samples
    '''
    if normalize:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return discrepancy(samples1, samples2, kernel, *args, **kwargs)


def test_mmd_compute():
    s1 = np.array([0.2, 0.8])
    s2 = np.array([0.3, 0.7])
    samples1 = [s1, s2]

    s3 = np.array([0.25, 0.75])
    s4 = np.array([0.35, 0.65])
    samples2 = [s3, s4]

    s5 = np.array([0.8, 0.2])
    s6 = np.array([0.7, 0.3])
    samples3 = [s5, s6]

    print('between samples1 and samples2: ', compute_mmd(samples1, samples2, kernel=gaussian_emd, sigma=1.0))
    print('between samples1 and samples3: ', compute_mmd(samples1, samples3, kernel=gaussian_emd, sigma=1.0))


if __name__ == '__main__':
    test_mmd_compute()
