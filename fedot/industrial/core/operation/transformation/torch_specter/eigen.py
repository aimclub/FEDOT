import torch


def aic_eigen_torch(s: torch.Tensor, N: int) -> torch.Tensor:
    """
    Calculate the Akaike Information Criterion (AIC) for eigenanalysis.

    This function computes the AIC values for a given sequence of eigenvalues to
    determine the optimal number of signal components. The AIC is a measure of
    the relative quality of a statistical model for a given set of data.

    Args:
        s (torch.Tensor): Tensor of eigenvalues.
        N (int): Number of data points.

    Returns:
        torch.Tensor: AIC values for each possible number of signal components.
    """
    s = torch.as_tensor(s, dtype=torch.float64)
    kaic = []
    n = s.numel()

    for k in range(n - 1):
        tail = s[k + 1:]
        denom = n - k

        ak = torch.sum(tail) / denom
        gk = torch.exp(torch.sum(torch.log(tail)) / denom)

        val = (
            -2.0 * denom * N * torch.log(gk / ak)
            + 2.0 * k * (2.0 * n - k)
        )
        kaic.append(val)

    return torch.stack(kaic)


def mdl_eigen_torch(s: torch.Tensor, N: int) -> torch.Tensor:
    """
    Calculate the Minimum Description Length (MDL) for eigenanalysis.

    This function computes the MDL values for a given sequence of eigenvalues to
    determine the optimal number of signal components. The MDL is a measure of
    the model complexity and goodness of fit.

    Args:
        s (torch.Tensor): Tensor of eigenvalues.
        N (int): Number of data points.

    Returns:
        torch.Tensor: MDL values for each possible number of signal components.
    """
    s = torch.as_tensor(s, dtype=torch.float64)

    kmdl = []
    n = s.numel()

    for k in range(0, n - 1):
        tail = s[k + 1:]
        ak = torch.mean(tail)
        gk = torch.exp(torch.mean(torch.log(tail)))
        val = (
            -(n - k) * N * torch.log(gk / ak)
            + 0.5 * k * (2.0 * n - k) * torch.log(torch.tensor(float(N)))
        )
        kmdl.append(val)

    return torch.stack(kmdl)


def get_signal_space_torch(
    S,
    NP,
    NSIG=None,
    threshold=None,
    criteria="aic",
):
    """
    Determine the number of signal components based on eigenvalues.

    This function estimates the number of significant signal components using
    either a fixed threshold, a fixed number of components, or information
    criteria (AIC or MDL).

    Args:
        S (torch.Tensor): Tensor of eigenvalues.
        NP (int): Number of data points parameter.
        NSIG (Optional[int]): Fixed number of signal components. If provided,
        this value is returned.
        threshold (Optional[float]): Threshold for determining significant
        eigenvalues.
        criteria (str): Information criterion to use ('aic' or 'mdl') if neither
        NSIG nor threshold is provided.

    Returns:
        int: Number of significant signal components.
    """
    if NSIG is not None:
        if NSIG <= 0:
            raise ValueError("NSIG must be positive")
        return NSIG

    if threshold is not None:
        m = threshold * torch.min(S)
        NSIG = int((S > m).sum().item())
        if NSIG == 0:
            NSIG = 1
        return NSIG

    if criteria == "aic":
        crit = aic_eigen_torch(S, NP * 2)
    elif criteria == "mdl":
        crit = mdl_eigen_torch(S, NP * 2)
    else:
        raise ValueError("criteria must be 'aic' or 'mdl'")
    NSIG = int(torch.argmin(crit).item()) + 1

    return NSIG


def eigen_torch(
    x,
    P,
    NSIG=None,
    threshold=None,
    NFFT=4096,
    method="ev",
    criteria="aic",
) -> tuple:
    """
    Perform eigenanalysis on a signal using forward-backward averaging.

    This function constructs a forward-backward matrix from the input signal,
    performs singular value decomposition (SVD), and estimates the power
    spectral density (PSD) using either the MUSIC or EV method.

    Args:
        x (torch.Tensor): Input signal tensor.
        P (int): Order of the forward-backward matrix.
        NSIG (Optional[int]): Number of signal components. If not provided, it
        is estimated.
        threshold (Optional[float]): Threshold for determining significant
        eigenvalues.
        NFFT (int): Number of FFT points for PSD estimation. Defaults to 4096.
        method (str): Method for PSD estimation ('music' or 'ev'). Defaults to
        'ev'.
        criteria (str): Information criterion to use ('aic' or 'mdl') if NSIG is
        not provided. Defaults to 'aic'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the PSD estimate
        and eigenvalues.

    Raises:
        ValueError: If P is too large for the given input signal length.
    """
    x = torch.as_tensor(x, dtype=torch.complex128)
    N = x.numel()

    NP = N - P
    if 2 * NP <= P - 1:
        raise ValueError("Decrease P")
    NP = min(NP, 100)

    FB = torch.zeros((2 * NP, P), dtype=torch.complex128, device=x.device)
    for i in range(NP):
        for k in range(P):
            FB[i, k] = x[i - k + P - 1]
            FB[i + NP, k] = torch.conj(x[i + k + 1])

    # SVD
    U, S, Vh = torch.linalg.svd(FB, full_matrices=True)
    V = -Vh.conj().T

    # Determine signal subspace
    NSIG = get_signal_space_torch(
        S,
        2 * NP,
        NSIG=NSIG,
        threshold=threshold,
        criteria=criteria,
    )

    PSD = torch.zeros(NFFT, dtype=torch.float64, device=x.device)

    for i in range(NSIG, P):
        Z = torch.zeros(NFFT, dtype=torch.complex128, device=x.device)
        Z[:P] = V[:P, i]

        Zf = torch.fft.fft(Z)

        if method == "music":
            PSD += Zf.abs() ** 2
        elif method == "ev":
            PSD += (Zf.abs() ** 2) / S[i]

    PSD = 1.0 / PSD
    nby2 = NFFT // 2
    part1 = torch.flip(PSD[1:nby2 + 1], dims=[0])
    part2 = torch.flip(PSD[nby2:nby2 * 2], dims=[0])
    newpsd = torch.cat([part1, part2])

    return newpsd, S


def pev_torch(
    x,
    IP: int,
    NSIG=None,
    threshold=None,
    NFFT=None,
    sampling=4096,
    scale_by_freq=False,
):
    """
    Compute the Pisarenko Eigenvector (PEV) method for power spectral density
    estimation.

    This function estimates the PSD of a signal using the Pisarenko Eigenvector
    method, which is based on eigenanalysis of the signal's covariance matrix.

    Args:
        x (torch.Tensor): Input signal tensor.
        IP (int): Order of the covariance matrix.
        NSIG (Optional[int]): Number of signal components. If not provided, it
        is estimated.
        threshold (Optional[float]): Threshold for determining significant
        eigenvalues.
        NFFT (Optional[int]): Number of FFT points for PSD estimation. If not
        provided, it is set to the length of x.
        sampling (int): Sampling frequency of the signal. Defaults to 4096.
        scale_by_freq (bool): If True, scale the PSD by frequency. Defaults to
        False.

    Returns:
        torch.Tensor: PSD estimate of the signal.
    """
    if NFFT is None:
        NFFT = x.shape[0]
    psd, eigenvalues = eigen_torch(
        x,
        IP,
        NSIG=NSIG,
        threshold=threshold,
        NFFT=NFFT,
        method="ev",
    )

    if not torch.is_complex(torch.as_tensor(x)):
        if NFFT % 2 == 0:
            psd = psd[: NFFT // 2 + 1] * 2
        else:
            psd = psd[: (NFFT + 1) // 2] * 2
        psd = torch.flip(psd, dims=[0])

    if scale_by_freq:
        df = sampling / NFFT
        psd = psd * (2 * torch.pi / df)

    return psd
