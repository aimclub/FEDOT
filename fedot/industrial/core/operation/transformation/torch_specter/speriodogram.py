import torch
import math


def get_window_torch(window,
                     N,
                     device=None,
                     dtype=torch.float64) -> torch.Tensor:
    """
    This function creates a window of a given type and length, which can be used
    for spectral analysis and signal processing tasks. Supported window types
    include rectangular, Hann, Hamming, Blackman, and Kaiser windows.

    Args:
        window (str): Type of window to generate. Supported values are:
                      - 'rectangular' or None: Rectangular window.
                      - 'hann': Hann window.
                      - 'hamming': Hamming window.
                      - 'blackman': Blackman window.
                      - 'kaiser': Kaiser window.
        N (int): Length of the window.
        device (Optional[torch.device]): Device on which to create the window
        tensor (e.g., 'cpu' or 'cuda'). Defaults to None.
        dtype (torch.dtype): Data type of the window tensor. Defaults to
        torch.float64.

    Returns:
        torch.Tensor: A 1D tensor representing the window function of length N.

    Raises:
        ValueError: If an unsupported window type is provided.
    """
    n = torch.arange(N, device=device, dtype=dtype)

    if window in (None, 'rectangular'):
        return torch.ones(N, device=device, dtype=dtype)

    if window == 'hann':
        return 0.5 - 0.5 * torch.cos(2 * math.pi * n / (N - 1))

    if window == 'hamming':
        return 0.54 - 0.46 * torch.cos(2 * math.pi * n / (N - 1))

    if window == 'blackman':
        return (0.42 - 0.5 * torch.cos(2 * math.pi * n / (N - 1))
                + 0.08 * torch.cos(4 * math.pi * n / (N - 1))
                )

    if window == 'kaiser':
        beta = 8.6
        return torch.kaiser_window(N, periodic=False, beta=beta,
                                   device=device, dtype=dtype)

    raise ValueError(f"Unsupported window type: {window}")


def speriodogram_torch(
    x,
    NFFT=None,
    detrend=None,
    sampling=4096,
    scale_by_freq=True,
    window='hamming',
    axis=0,
) -> torch.Tensor:
    """
    Compute the periodogram power spectral density estimate of a signal using a
    short-time Fourier transform.

    This function computes the periodogram, which is an estimate of the power
    spectral density (PSD) of a signal. It supports both 1D and 2D input tensors
    and allows for detrending and windowing of the input signal. The function
    can handle both real and complex input signals.

    Args:
        x (torch.Tensor): Input signal tensor. Can be 1D (single signal) or 2D
        (multiple signals).
        NFFT (Optional[int]): Number of FFT points. If None, NFFT is set to the
        length of the input signal.
        detrend (Optional[bool]): If True, remove the mean of the signal before
        computing the periodogram.
        sampling (int): Sampling frequency of the signal. Defaults to 4096.
        scale_by_freq (bool): If True, scale the PSD by frequency. Defaults to
        True.
        window (str): Window function to apply to the signal. Defaults to
        'hamming'.
                    Supported windows: 'rectangular', 'hann', 'hamming',
                    'blackman', 'kaiser'.
        axis (int): Axis along which to compute the FFT. Defaults to 0.

    Returns:
        torch.Tensor: The periodogram power spectral density estimate of the
                      signal.
                      For 1D input, the output is transposed (1D tensor of
                      PSD values).
                      For 2D input, the output is a 2D tensor where each column
                      corresponds to the PSD of a signal.

    Raises:
        ValueError: If the input tensor is not 1D or 2D.
    """
    x = torch.as_tensor(x, dtype=torch.float64)

    if x.ndim == 1:
        r = x.shape[0]
        axis = 0
        w = get_window_torch(window, r, device=x.device, dtype=x.dtype)
    elif x.ndim == 2:
        r, c = x.shape
        w = torch.stack(
            [get_window_torch(window, r, device=x.device, dtype=x.dtype)
             for _ in range(c)],
            dim=1
        )
    else:
        raise ValueError("x must be 1D or 2D")

    if NFFT is None:
        NFFT = r

    if detrend is True:
        m = torch.mean(x, dim=axis, keepdim=True)
    else:
        m = 0.0

    xw = x * w - m

    # FFT
    isreal = torch.isreal(x).all()
    if isreal:
        if x.ndim == 2:
            psd = torch.abs(
                torch.fft.rfft(xw, n=NFFT, dim=0)
            ) ** 2 / r
        else:
            psd = torch.abs(
                torch.fft.rfft(xw, n=NFFT, dim=-1)
            ) ** 2 / r
    else:
        if x.ndim == 2:
            psd = torch.abs(
                torch.fft.fft(xw, n=NFFT, dim=0)
            ) ** 2 / r
        else:
            psd = torch.abs(
                torch.fft.fft(xw, n=NFFT, dim=-1)
            ) ** 2 / r

    if scale_by_freq:
        df = sampling / float(NFFT)
        psd = psd * (2 * math.pi / df)

    return psd.T if x.ndim == 1 else psd
