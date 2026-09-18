import torch

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.constanst_repository import SINGULAR_VALUE_BETA_THR, SINGULAR_VALUE_MEDIAN_THR


def sv_to_explained_variance_ratio(singular_values):
    """Calculate the explained variance ratio of the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.

    Returns:
        explained_variance (int): Explained variance percent.
        n_components (int): Number of singular values to use.

    """
    singular_values = [abs(x) for x in singular_values]
    variance = [x / sum(singular_values) * 100 for x in singular_values]
    variance_selected = [x for x in variance if x > 3]
    return variance_selected if len(variance_selected) != 0 else variance[:2]


def sv_to_explained_variance_ratio_torch(singular_values: torch.Tensor) -> list:
    """Calculate the explained variance ratio of the singular values.
    Filter values which is less than 3%. Return first two values, if there
    is empty tensor after filtering.

    Args:
        singular_values (torch.Tensor): Singular values.

    Returns:
        list: List of values, filtered by explained variance.
    """
    singular_values = torch.abs(singular_values)
    sum_sv = torch.sum(singular_values)
    variance = (singular_values / sum_sv) * 100

    variance_selected = variance[variance > 3]

    if variance_selected.numel() == 0:
        variance_selected = variance[:2]

    return variance_selected.tolist()


def transform_eigen_to_ts(X_elem):
    X_rev = X_elem[::-1]
    eigenvector_to_ts = list(X_rev.diagonal(
        j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1]))
    return eigenvector_to_ts


def eigencorr_matrix(U, S, V,
                     n_components: int = None,
                     correlation_level: float = 0.4):
    d = S.shape[0]
    L = S.shape[0]
    K = V.shape[1]
    if n_components is None:
        n_components = d
    corellated_components = {}
    components_iter = range(n_components)

    X_elem = np.array([S[i] * np.outer(U[:, i], V[i, :]) for i in range(0, d)])

    w = np.array(
        # returns the sequence 1 to L (first line in definition of w)
        list(np.arange(L) + 1) +
        # repeats L K-L-1 times (second line in w definition)
        [L] * (K - L - 1) +
        # reverses the first list (equivalent to the third line)
        list(np.arange(L) + 1)[::-1]
    )

    # Get all the components of the toy series, store them as columns in F_elem array.
    F_elem = np.array([transform_eigen_to_ts(X_elem[i]) for i in range(d)])

    # Calculate the individual weighted norms,
    # ||F_i||_w, first, then take inverse square-root so we don't have to later.
    vector_list = []
    for i in range(d):
        squared_vector = F_elem[i] ** 2
        normed_vector = w.dot(squared_vector)
        vector_list.append(normed_vector)
    F_wnorms = np.array(vector_list)
    F_wnorms = F_wnorms ** -0.5

    # Calculate the w-corr matrix. The diagonal elements are equal to 1, so we can start with an identity matrix
    # and iterate over all pairs of i's and j's (i != j), noting that Wij = Wji.
    Wcorr = np.identity(d)
    for i in range(d):
        for j in range(i + 1, d):
            eigen_vector = F_elem[i]
            next_eigen_vector = F_elem[j]
            Wcorr[i, j] = abs(w.dot(eigen_vector * next_eigen_vector)
                              * F_wnorms[i] * F_wnorms[j])
            Wcorr[j, i] = Wcorr[i, j]

    component_set = [x for x in components_iter]
    for i in components_iter:
        component_idx = np.where(Wcorr[i] > correlation_level)[0]
        intersect = set(component_set).intersection(component_idx)
        have_intersection = len(intersect) != 0
        if have_intersection:
            for j in component_idx.tolist():
                if j in component_set:
                    component_set.remove(j)
            corellated_components.update({f'{i}_component': component_idx})
        else:
            continue

    return corellated_components


def singular_value_hard_threshold(singular_values,
                                  rank=None,
                                  beta=None,
                                  threshold=SINGULAR_VALUE_MEDIAN_THR) -> list:
    """Calculate the hard threshold for the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.
        rank (int): Number of singular values to use.
        beta (float): Beta value.
        threshold (float): Threshold value.

    Returns:
        adjusted singular values array (array-like, shape (n_components,)): Adjusted array of singular values.

    """
    if rank is not None:
        return singular_values[:rank]
    else:
        # Find the median of the singular values
        singular_values = [s_val for s_val in singular_values if s_val > 0.01]
        if len(singular_values) == 1:
            return singular_values[:1]
        median_sv = np.median(singular_values)
        # Find the adjusted rank
        if threshold is None:
            threshold = SINGULAR_VALUE_BETA_THR(beta)
        sv_threshold = threshold * median_sv
        # Find the threshold value
        adjusted_rank = np.sum(singular_values >= sv_threshold)
        # If the adjusted rank is 0, recalculate the threshold value
        if adjusted_rank == 0:
            sv_threshold = 2.31 * median_sv
            adjusted_rank = max(np.sum(singular_values >= sv_threshold), 1)
        return singular_values[:adjusted_rank]


def singular_value_hard_threshold_torch(
    singular_values: torch.Tensor,
    rank: int = None,
    beta: float = None,
    threshold: float = SINGULAR_VALUE_MEDIAN_THR
) -> list:
    """Calculate the hard threshold for the singular values.
    If rank is specified, return the first 'rank' singular values.
    Otherwise, calculate the threshold based on the median of singular values
    and return singular values above the threshold. If no singular values
    are above the threshold, recalculate the threshold and ensure at least one
    singular value is returned.

    Args:
        singular_values (torch.Tensor): Singular values.
        rank (int, optional): Number of singular values to use.
        Defaults to None.
        beta (float, optional): Beta value for threshold calculation.
        Defaults to None.
        threshold (float, optional): Threshold value. Defaults to None.

    Returns:
        list: Adjusted list of singular values.
    """
    if rank is not None:
        return singular_values[:rank].tolist()
    else:
        # Filter out small singular values
        singular_values = singular_values[singular_values > 0.01]

        if singular_values.numel() == 1:
            return singular_values[:1].tolist()

        # Calculate median of singular values
        median_sv = torch.median(singular_values)

        # Calculate threshold
        if threshold is None:
            threshold = SINGULAR_VALUE_BETA_THR(beta)

        sv_threshold = threshold * median_sv

        # Calculate adjusted rank
        adjusted_rank = torch.sum(singular_values >= sv_threshold).item()

        # If the adjusted rank is 0, recalculate the threshold value
        if adjusted_rank == 0:
            sv_threshold = 2.31 * median_sv
            adjusted_rank = max(torch.sum(singular_values >=
                                          sv_threshold).item(), 1)

        return singular_values[:adjusted_rank].tolist()


def reconstruct_basis(U, Sigma, VT, ts_length):
    # check whether Sigma value is set to 'ill_conditioned'
    if isinstance(Sigma, str):
        # rank = round(len(VT)*0.1)
        rank = len(VT)
        TS_comps = np.zeros((ts_length, rank))
        U, S, V = U[0], U[1], U[2]
        for idx, (comp, eigen_idx) in enumerate(VT.items()):
            X_dominant = np.sum([S[i] * np.outer(U[:, i], V[i, :])
                                for i in eigen_idx], axis=0)
            grouped_eigenvector = transform_eigen_to_ts(X_dominant)
            if idx == rank:
                break
            else:
                TS_comps[:, idx] = grouped_eigenvector
        TS_comps[:, 1] = np.sum(TS_comps[:, 1:], axis=1)
        TS_comps = TS_comps[:, :2]
        return TS_comps
    if len(Sigma.shape) > 1:
        def multi_reconstruction(x):
            return reconstruct_basis(U=U, Sigma=x, VT=VT, ts_length=ts_length)

        TS_comps = list(map(multi_reconstruction, Sigma))
    else:
        rank = Sigma.shape[0]
        TS_comps = np.zeros((ts_length, rank))
        for i in range(rank):
            X_elem = Sigma[i] * np.outer(U[:, i], VT[i, :])
            X_rev = X_elem[::-1]
            eigenvector = [X_rev.diagonal(
                j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
            TS_comps[:, i] = eigenvector
    return TS_comps


def reconstruct_basis_torch(U: torch.Tensor,
                            Sigma: torch.Tensor,
                            VT: torch.Tensor,
                            ts_length: int) -> torch.Tensor:
    """Reconstruct time-series basis components from SVD factors
    using diagonal averaging (SSA hankelization).

    This function reconstructs temporal components from a rank-1
    SVD expansion of a trajectory (Hankel) matrix by averaging
    anti-diagonals. The implementation is numerically equivalent
    to the classical SSA reconstruction procedure.

    Args:
        U (torch.Tensor): Left singular vectors of shape (L, rank).
        Sigma (torch.Tensor): Singular values of shape (rank,) or
        batched singular values of shape (B, rank).
        VT (torch.Tensor): Right singular vectors of shape (rank, K).
        ts_length (int): Length of the reconstructed time series
            (L + K - 1).

    Returns:
        torch.Tensor: Reconstructed time-series components of shape
            (ts_length, rank), or (B, ts_length, rank) for batched input.
    """

    if Sigma.dim() > 1:
        return torch.stack([
            reconstruct_basis_torch(U, s, VT, ts_length)
            for s in Sigma
        ])

    device = U.device
    dtype = U.dtype

    rank = Sigma.shape[0]
    L = U.shape[0]
    K = VT.shape[1]

    TS_comps = torch.zeros(ts_length, rank, device=device, dtype=dtype)

    # indexes for diagonal's means
    r = torch.arange(L, device=device).unsqueeze(1)
    c = torch.arange(K, device=device).unsqueeze(0)
    diag_idx = (L - 1 - r) + c

    for i in range(rank):
        # find reverse conversion
        X_elem = Sigma[i] * torch.outer(U[:, i], VT[i, :])
        X_rev = torch.flip(X_elem, dims=[0])

        # find diagonal's means
        diag_sum = torch.zeros(ts_length, device=device, dtype=dtype)
        diag_cnt = torch.zeros(ts_length, device=device, dtype=dtype)
        diag_sum.scatter_add_(0, diag_idx.flatten(), X_rev.flatten())
        diag_cnt.scatter_add_(0, diag_idx.flatten(),
                              torch.ones_like(X_rev).flatten())
        TS_comps[:, i] = diag_sum / diag_cnt

    return TS_comps
