import warnings
import torch


warnings.filterwarnings("ignore")


def mean_torch(x: torch.Tensor, axis=-1):
    mean = torch.mean(x, axis=axis)
    return mean.item() if mean.numel() == 1 else mean


def median_torch(x: torch.Tensor,
                 axis: int = -1,
                 max_elements: int = 10_000_000) -> torch.Tensor:
    """
    This function calculates the median using PyTorch's kthvalue operation, processing the input
    in batches to avoid memory overflow for large tensors. It handles both even and odd lengths
    of the input tensor, returning the median value(s) along the specified axis.

    For even-length tensors, the median is the average of the two central values.
    For odd-length tensors, the median is the central value.

    Args:
        x (torch.Tensor): Input tensor.
        axis (int, optional): Axis along which to compute the median. Defaults to -1.
        max_elements (int, optional): Maximum number of elements to process in a single batch.
                                     Defaults to 10,000,000.

    Returns:
        torch.Tensor: The median value(s) along the specified axis.
    """
    n = x.size(axis)
    k = n // 2

    # Calculate the number of elements per slice
    elems_per_slice = x.select(axis, 0).numel()
    slice_size = max(1, max_elements // elems_per_slice)

    outputs = []
    for i in range(0, x.size(0), slice_size):
        chunk = x[i:i + slice_size]
        if n % 2 == 1:
            median_chunk = chunk.kthvalue(k + 1, dim=axis).values
        else:
            v1 = chunk.kthvalue(k, dim=axis).values
            v2 = chunk.kthvalue(k + 1, dim=axis).values
            median_chunk = (v1 + v2) / 2
        outputs.append(median_chunk)
    return torch.cat(outputs, dim=0)


def std_torch(x: torch.Tensor, axis=-1):
    std = torch.std(x, axis=axis, unbiased=False)
    return std.item() if std.numel() == 1 else std


def max_torch(x: torch.Tensor, axis=-1):
    max = torch.max(x, axis=axis).values
    return max.item() if max.numel() == 1 else max


def min_torch(x: torch.Tensor, axis=-1):
    min = torch.min(x, axis=axis).values
    return min.item() if min.numel() == 1 else min


def q5_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.05, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def q25_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.25, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def q75_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.75, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def q95_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.95, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def lambda_less_zero(array: torch.Tensor, axis=-1) -> int | torch.Tensor:
    mask = (array < 0.01).int()
    return torch.sum(mask).item() if mask.numel() == 1 else torch.sum(mask, dim=axis)


def quantile_torch(array: torch.Tensor, q: float, axis=-1) -> float | torch.Tensor:
    axis = axis % array.ndim
    quant = torch.quantile(input=array, q=q, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def diff(array: torch.Tensor, axis=-1) -> float:
    return (array[-1] - array[0]).item()


def skewness_torch(x: torch.Tensor, axis=-1):
    """
    Skewness measures the asymmetry of the data distribution around the mean.
    Positive skewness indicates a longer right tail, while negative skewness indicates a longer left tail.

    Args:
        x (torch.Tensor): Input tensor.
        axis (int, optional): Axis along which to compute skewness. Defaults to -1.

    Returns:
        float | torch.Tensor: The skewness value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with skewness values along the specified axis.
    """
    mean = x.mean(dim=axis, keepdim=True)
    std = x.std(dim=axis, unbiased=False, keepdim=True)
    skew = ((x - mean) ** 3).mean(dim=axis) / (std.squeeze(axis) ** 3 + 1e-12)
    return skew


def kurtosis_torch(x: torch.Tensor, axis=-1, fisher=True):
    """
    Kurtosis measures the "tailedness" of the data distribution. If `fisher` is True,
    the result is the excess kurtosis (kurtosis minus 3), which compares the distribution
    to a normal distribution. Positive values indicate heavier tails, while negative values
    indicate lighter tails.

    Args:
        x (torch.Tensor): Input tensor.
        axis (int, optional): Axis along which to compute kurtosis. Defaults to -1.
        fisher (bool, optional): If True, return excess kurtosis. Defaults to True.

    Returns:
        float | torch.Tensor: The kurtosis value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with kurtosis values along the specified axis.
    """
    mean = x.mean(dim=axis, keepdim=True)
    std = x.std(dim=axis, unbiased=False, keepdim=True)
    kurt = ((x - mean) ** 4).mean(dim=axis) / (std.squeeze(axis) ** 4 + 1e-12)
    if fisher:
        kurt = kurt - 3.0
    return kurt


def n_peaks_torch(X: torch.Tensor, axis=-1):
    """
    A peak is defined as a point where the signal transitions from increasing to decreasing.
    The function identifies peaks by analyzing sign changes in the first differences of the signal.

    Args:
        X (torch.Tensor): Input tensor containing time series data.
        axis (int, optional): Axis along which to count peaks. Defaults to -1.

    Returns:
        int | torch.Tensor: The number of peaks. Returns a scalar if input is 1D,
                            otherwise a tensor with peak counts along the specified axis.
    """
    if axis != -1:
        x = X.transpose(axis, -1)
    if X.ndim == 1:
        x = X.unsqueeze(0)
    elif X.ndim > 2:
        x = X.reshape(-1, X.shape[-1])
    else:
        x = X
    d = torch.diff(x, dim=-1)
    s = torch.sign(d)
    s[s == 0] = 1
    dd = torch.diff(s, axis=-1)
    n_peaks = torch.count_nonzero(dd == -2, axis=-1)
    if X.ndim > 2:
        n_peaks.reshape(X.shape[0], X.shape[1])
    else:
        return n_peaks.item() if n_peaks.numel() == 1 else n_peaks


def mean_ptp_distance_torch(X: torch.Tensor, axis=-1):
    """
    Peaks are identified as points where the signal transitions from increasing to decreasing.
    The function calculates the average distance (in samples) between these peaks.

    Args:
        X (torch.Tensor): Input tensor containing time series data.
        axis (int, optional): Axis along which to compute the mean peak-to-peak distance. Defaults to -1.

    Returns:
        float | torch.Tensor: The mean peak-to-peak distance. Returns a scalar if input is 1D,
                              otherwise a tensor with distances along the specified axis.
    """
    if axis != -1:
        x = X.transpose(axis, -1)
    if X.ndim == 1:
        x = X.unsqueeze(0)
    elif X.ndim > 2:
        x = X.reshape(-1, X.shape[-1])
    else:
        x = X
    d = torch.diff(x, dim=-1)
    s = torch.sign(d)
    s[s == 0] = 1
    dd = torch.diff(s, dim=-1)
    indices = torch.where(dd == -2)
    rows, cols = indices

    max_count = torch.bincount(rows).max().item()
    B = dd.shape[0]
    count_peaks = torch.bincount(rows, minlength=B)
    max_count = int(count_peaks.max().item())
    offsets = torch.cumsum(torch.cat([count_peaks[:1] * 0, count_peaks[:-1]]), dim=0)
    idx_global = torch.arange(len(rows), device=rows.device)
    idx_in_row = idx_global - offsets[rows]
    peak_indices = torch.zeros((B, max_count), device=X.device, dtype=torch.long)
    peak_indices[rows, idx_in_row] = cols + 1

    if peak_indices.numel() == 0:
        return torch.zeros(X.shape[0], device=x.device)
    count_peaks = torch.tensor(count_peaks).to(X.device) - 1.0
    count_peaks[count_peaks == -1] = 1e-12
    diff_peaks = torch.diff(peak_indices, dim=-1)
    diff_peaks[diff_peaks < 0] = 0
    mean_dist = diff_peaks.sum(dim=-1) / count_peaks.float()
    if X.ndim > 2:
        mean_dist.reshape(X.shape[0], X.shape[1])
    else:
        return mean_dist.item() if mean_dist.numel() == 1 else mean_dist


def slope_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    """
    The slope is calculated using the least squares method, representing the rate of change
    of the data with respect to its index. This is useful for identifying trends in the data.

    Args:
        array (torch.Tensor): Input tensor containing time series data.
        axis (int, optional): Axis along which to compute the slope. Defaults to -1.

    Returns:
        float | torch.Tensor: The slope value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with slope values along the specified axis.
    """
    y = array.to(torch.float32)
    axis = axis % y.ndim
    n = y.shape[axis]
    x = torch.arange(n, device=y.device, dtype=torch.float32)
    x_mean = x.mean()
    y_mean = y.mean(dim=axis, keepdim=True)
    slope = torch.sum((x - x_mean) * (y - y_mean), dim=axis) / (torch.sum((x - x_mean) ** 2 + 1e-12))
    return slope.item() if slope.numel() == 1 else slope


def ben_corr_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when
     compared to the Newcomb-Benford's Law distribution [3][4].

     Args:
            x (torch.Tensor): time series or batch to calculate the feature of

     Returns:
            float | torch.Tensor: the value(s) of the feature

     .. math::

         P(d)=\\log_{10}\\left(1+\\frac{1}{d}\\right)

     where :math:`P(d)` is the Newcomb-Benford distribution for :math:`d` that is the leading digit of the number
     {1, 2, 3, 4, 5, 6, 7, 8, 9}.

     .. rubric:: References

     [1] A Statistical Derivation of the Significant-Digit Law, Theodore P. Hill, Statistical Science, 1995
     [2] The significant-digit phenomenon, Theodore P. Hill, The American Mathematical Monthly, 1995
     [3] The law of anomalous numbers, Frank Benford, Proceedings of the American philosophical society, 1938
     [4] Note on the frequency of use of the different digits in natural numbers, Simon Newcomb, American Journal of
     mathematics, 1881
    """
    if (axis != -1) | (axis != len(x.shape)):
        x = x.transpose(axis, -1)
    *batch_shape, T = x.shape
    B = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
    x_flat = x.reshape(B, T)

    # preprocess and get histogramm of first digits for each ts
    x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
    x_flat = torch.abs(x_flat)
    x_flat = torch.clamp(x_flat, min=1e-8)
    exponents = torch.floor(torch.log10(x_flat))
    mantissas = x_flat / (10**exponents)
    first_digits = torch.floor(mantissas).clamp(1, 9).to(torch.int64)
    offsets = torch.arange(B, device=x.device) * 10
    fd_flat = (first_digits + offsets.unsqueeze(1)).reshape(-1)
    counts = torch.bincount(fd_flat, minlength=B * 10)
    counts = counts.reshape(B, 10)[:, 1:10]
    data_dist = counts / (counts.sum(dim=1, keepdim=True) + 1e-12)

    # Benford's distribution
    digits = torch.arange(1, 10, device=x.device, dtype=torch.float32)
    benford = torch.log10(1 + 1 / digits).unsqueeze(0).expand(B, -1)

    # corr coef
    x_mean = benford.mean(dim=1, keepdim=True)
    y_mean = data_dist.mean(dim=1, keepdim=True)
    num = ((benford - x_mean) * (data_dist - y_mean)).sum(dim=1)
    den = torch.sqrt(
        ((benford - x_mean)**2).sum(dim=1) *
        ((data_dist - y_mean)**2).sum(dim=1)
    )
    corr = num / (den + 1e-12)
    corr = torch.nan_to_num(corr, nan=0.0)
    return corr.reshape(batch_shape) if batch_shape else corr.item()


def interquantile_range_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    """
    The IQR is the difference between the 75th and 25th percentiles, providing a measure of
    statistical dispersion and robustness to outliers.

    Args:
        array (torch.Tensor): Input tensor.
        axis (int, optional): Axis along which to compute the IQR. Defaults to -1.

    Returns:
        float | torch.Tensor: The interquartile range value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with IQR values along the specified axis.
    """
    return quantile_torch(array, 0.75, axis) - quantile_torch(array, 0.25, axis)


def energy_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    """
    Computes the sum of squared values divided by the length of the axis, representing
    the average power or signal strength.

    Args:
        array (torch.Tensor): Input tensor.
        axis (int, optional): Axis along which to compute energy. Defaults to -1.

    Returns:
        float | torch.Tensor: The energy value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with energy values along the specified axis.
    """
    axis = axis % array.ndim
    energy = torch.sum(array ** 2, dim=axis) / array.shape[axis]
    return energy.item() if energy.numel() == 1 else energy


def autocorrelation_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    Measures the Pearson correlation between the time series and its lag-1 shifted version,
    capturing linear dependencies in sequential data.

    Args:
        x (torch.Tensor): Input tensor containing time series data.
        axis (int, optional): Axis along which to compute autocorrelation. Defaults to -1.

    Returns:
        float | torch.Tensor: The autocorrelation value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with autocorrelation values along the specified axis.
    """
    axis = axis % x.ndim
    lagged = torch.roll(x, shifts=1, dims=axis)
    x_centered = x - x.mean(dim=axis, keepdim=True)
    lagged_centered = lagged - lagged.mean(dim=axis, keepdim=True)
    num = (x_centered * lagged_centered).sum(dim=axis)
    denom = torch.sqrt((x_centered ** 2).sum(dim=axis) *
                       (lagged_centered ** 2).sum(dim=axis))
    corr = num / denom.clamp(min=1e-12)
    return corr.item() if corr.numel() == 1 else corr


def zero_crossing_rate_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    The zero-crossing rate is the frequency at which the signal changes sign, normalized by the
    length of the signal. The input is scaled to the range [-1, 1] before computation.

    Args:
        x (torch.Tensor): Input tensor containing time series data.
        axis (int, optional): Axis along which to compute the zero-crossing rate. Defaults to -1.

    Returns:
        float | torch.Tensor: The zero-crossing rate value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with zero-crossing rates along the specified axis.
    """
    if (axis != -1) | (axis != len(x.shape)):
        x = x.transpose(axis, -1)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x_min = x.min(dim=axis, keepdim=True).values
    x_max = x.max(dim=axis, keepdim=True).values
    x_scaled = 2 * (x - x_min) / (x_max - x_min + 1e-12) - 1
    signs = torch.sign(x_scaled)
    signs[signs == 0] = -1
    diff = torch.diff(signs, dim=axis)
    crossings = (diff != 0).sum(dim=axis)
    rate = crossings / x.shape[axis]
    return rate.item() if rate.numel() == 1 else rate


def shannon_entropy_torch(X: torch.Tensor, axis=None):
    """
    The Shannon entropy is calculated by sorting the input, identifying unique value groups,
    and computing the entropy of the resulting probability distribution. This provides
    a measure of uncertainty or randomness in the data.

    Args:
        X (torch.Tensor): Input tensor.
        axis (int, optional): Axis along which to compute entropy. Defaults to -1.

    Returns:
        float | torch.Tensor: The Shannon entropy value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with entropy values along the specified axis.
    """
    if X.ndim == 1:
        x = X.unsqueeze(0)
    elif X.ndim > 2:
        x = X.reshape(-1, X.shape[-1])
    else:
        x = X.clone()
    B, N = x.shape
    x_sorted, _ = torch.sort(x, dim=axis)
    new_group = torch.cat([
        torch.ones(B, 1, device=x.device, dtype=torch.bool),
        x_sorted[:, 1:] != x_sorted[:, :-1]
    ], dim=axis)
    group_starts_list = [row.nonzero(as_tuple=False).squeeze(1) for row in new_group]
    group_sizes_list = []
    for starts in group_starts_list:
        ends = torch.cat([starts[1:], torch.tensor([N], device=x.device)])
        group_sizes = (ends - starts)
        group_sizes_list.append(group_sizes)
    max_groups = max(gs.numel() for gs in group_sizes_list)
    group_sizes = torch.zeros((B, max_groups), device=x.device)
    for i, gs in enumerate(group_sizes_list):
        group_sizes[i, : gs.numel()] = gs
    probs = group_sizes / N
    entropy = torch.sum(-probs * torch.log2(probs + 1e-12), dim=1)
    if X.ndim > 2:
        entropy = entropy.reshape([X.shape[0], X.shape[1]])
    return entropy.item() if entropy.numel() == 1 else entropy


def base_entropy(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    Estimate the Shannon entropy of a probability distribution along a specified axis.

    Normalizes input values to form a probability distribution, then computes the entropy
    using the formula: -sum(p * log(p)). Handles numerical stability with small epsilon values.

    Args:
        x (torch.Tensor): Input tensor representing a distribution or counts.
        axis (int, optional): Axis along which to compute entropy. Defaults to -1.

    Returns:
        float | torch.Tensor: Computed entropy value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with entropy values along the specified axis.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    axis = axis % x.ndim
    probs = x / (x.sum(dim=axis, keepdim=True) + 1e-12)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=axis)
    return entropy.item() if entropy.numel() == 1 else entropy


def ptp_amp_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    Measures the difference between the maximum and minimum values, effectively capturing
    the signal's dynamic range or amplitude variation.

    Args:
        x (torch.Tensor): Input tensor containing signal data.
        axis (int, optional): Axis along which to compute peak-to-peak amplitude. Defaults to -1.

    Returns:
        float | torch.Tensor: Peak-to-peak amplitude value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with amplitude values along the specified axis.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if axis < 0:
        axis = x.ndim + axis
    diff = x.max(dim=axis).values - x.min(dim=axis).values
    return diff.item() if diff.numel() == 1 else diff


def crest_factor_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    The crest factor is the ratio of the peak amplitude to the root mean square (RMS) value,
    indicating the presence of peaks in the signal. Useful for detecting impulsive or transient events.

    Args:
        x (torch.Tensor): Input tensor containing signal data.
        axis (int, optional): Axis along which to compute the crest factor. Defaults to -1.

    Returns:
        float | torch.Tensor: Crest factor value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with crest factor values along the specified axis.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if not torch.is_floating_point(x):
        x = x.float()
    num = x.abs().max(dim=axis).values
    den = torch.sqrt((x ** 2).mean(dim=axis)) + 1e-12
    res = num / den
    return res.item() if res.numel() == 1 else res


def mean_ema_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    Applies an exponential weighting to the signal values, where the span of the EMA
    is dynamically set to 10% of the signal length. Useful for smoothing and trend analysis.

    Args:
        x (torch.Tensor): Input tensor containing signal data.
        axis (int, optional): Axis along which to compute the EMA. Defaults to -1.

    Returns:
        float | torch.Tensor: EMA value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with EMA values along the specified axis.
    """
    T = x.shape[axis]
    span = max(int(T / 10), 2)
    alpha = 2 / (span + 1)
    weights = (1 - alpha) ** torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    ema = torch.sum(x * weights, dim=axis)
    return ema.item() if ema.numel() == 1 else ema


def mean_moving_median_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    Applies a median filter over sliding windows of the signal, where the window size
    is dynamically set to 10% of the signal length. Effective for robust smoothing and outlier suppression.

    Args:
        x (torch.Tensor): Input tensor containing signal data.
        axis (int, optional): Axis along which to compute the moving median. Defaults to -1.

    Returns:
        float | torch.Tensor: Moving median value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with median values along the specified axis.
    """
    if axis != -1:
        x = x.transpose(axis, -1)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    T = x.shape[axis]
    span = max(int(T / 10), 2)
    windows = x.unfold(dimension=axis, size=span, step=1)
    medians = median_torch(windows)
    res = medians.mean(dim=axis)
    return res.item() if res.numel() == 1 else res


def hjorth_mobility_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    Mobility measures the mean frequency or the rate of change in the signal.
    It is computed as the square root of the variance of the first derivative divided by the variance of the signal.

    Args:
        x (torch.Tensor): Input tensor containing signal data.
        axis (int, optional): Axis along which to compute mobility. Defaults to -1.

    Returns:
        float | torch.Tensor: Hjorth mobility value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with mobility values along the specified axis.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if not torch.is_floating_point(x):
        x = x.float()
    diff = torch.diff(x, dim=axis)
    M2 = (diff ** 2).mean(dim=axis)
    TP = (x ** 2).mean(dim=axis)
    mobility = torch.sqrt(M2 / (TP + 1e-12))
    return mobility.item() if mobility.numel() == 1 else mobility


def hjorth_complexity_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    Complexity measures the change in frequency or how similar the signal is to pure sine waves.
    It is computed as the ratio of the mobility of the derivative to the mobility of the signal,
    normalized by the variance of the signal.

    Args:
        x (torch.Tensor): Input tensor containing signal data.
        axis (int, optional): Axis along which to compute complexity. Defaults to -1.

    Returns:
        float | torch.Tensor: Hjorth complexity value(s). Returns a scalar if input is 1D,
                              otherwise a tensor with complexity values along the specified axis.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if not torch.is_floating_point(x):
        x = x.float()
    diff = torch.diff(x, dim=axis)
    M2 = (diff ** 2).mean(dim=axis)
    TP = (x ** 2).mean(dim=axis)
    M4 = (torch.diff(diff, dim=axis) ** 2).mean(dim=axis)
    complexity = torch.sqrt((M4 * TP) / (M2 * M2 + 1e-12))
    return complexity.item() if complexity.numel() == 1 else complexity


def hurst_exponent_torch(X: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """ Compute the Hurst Exponent of the time series. The Hurst exponent is used as a measure of long-term memory of
    time series. It relates to the autocorrelations of the time series, and the rate at which these decrease as the
    lag between pairs of values increases. For a stationary time series, the Hurst exponent is equivalent to the
    autocorrelation exponent.

    Args:
        X (torch.Tensor): time series or batch to calculate the feature of

    Returns:
        H: Hurst exponent of the time series

    Notes:
        Author of this function is Xin Liu

    """
    if (axis != -1) | (axis != len(X.shape)):
        x = X.transpose(axis, -1)
    if X.ndim > 2:
        x = X.reshape(-1, X.shape[-1])
    if X.ndim == 1:
        x = X.unsqueeze(0)

    B, T = x.shape
    device = x.device
    dtype = x.dtype

    t = torch.arange(1, T + 1, device=device, dtype=dtype).unsqueeze(0)
    y = torch.cumsum(x, dim=-1)
    ave_t = y / t
    S_T = torch.zeros((B, T), device=device, dtype=dtype)
    R_T = torch.zeros((B, T), device=device, dtype=dtype)
    for i in range(T):
        segment = x[:, :i + 1]
        S_T[:, i] = segment.std(dim=-1, unbiased=False)
        X_T = y - t * ave_t[:, i].unsqueeze(1)
        R_T[:, i] = X_T[:, :i + 1].amax(dim=-1) - X_T[:, :i + 1].amin(dim=-1)
    RS = R_T / (S_T + 1e-12)
    RS = torch.log(RS[:, 1:])
    n = torch.log(t[:, 1:]).squeeze(0)

    ones = torch.ones_like(n)
    A = torch.stack([n, ones], dim=1)
    H = torch.empty(B, device=device, dtype=dtype)
    for b in range(B):
        sol = torch.linalg.lstsq(A, RS[b].unsqueeze(1)).solution.squeeze()
        H[b] = sol[0]
    if X.ndim > 2:
        H = H.reshape(X.shape[0], X.shape[1])
    return H if H.numel() > 1 else H.item()


def pfd_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """
    The Petrosian fractal dimension (PFD) is a chaotic algorithm used to calculate EEG signal complexity

    Args:
        x (torch.Tensor): time series or batch to calculate the feature of

    Returns:
        pfd: Petrosian fractal dimension of the time series
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)

    D = torch.diff(x, dim=axis)
    N_delta = ((D[..., 1:] * D[..., :-1]) < 0).sum(dim=axis)
    n = x.shape[axis]
    n = torch.tensor(float(n), device=x.device)
    num = torch.log10(n)
    den = torch.log10(n) + torch.log10(n / (n + 0.4 * N_delta))
    pfd = num / den
    return pfd.item() if pfd.numel() == 1 else pfd
