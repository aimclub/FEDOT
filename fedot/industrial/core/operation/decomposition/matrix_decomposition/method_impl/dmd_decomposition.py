from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.constanst_repository import DEFAULT_SVD_SOLVER, DEFAULT_QR_SOLVER


def rq(A):
    n, m = A.shape
    Q, R = DEFAULT_QR_SOLVER(np.flipud(A).T, mode='complete')
    R = np.rot90(R.T, 2)
    Q = np.flipud(Q.T)
    if n > m:
        R = np.hstack((np.zeros((n, n - m)), R))
        Q = np.vstack((np.zeros((n - m, m)), Q))
    return R, Q


def tls(A, B):
    n = A.shape[1]
    if A.shape[0] != B.shape[0]:
        raise ValueError('Matrices are not conformant.')
    R1 = np.hstack((A, B))
    U, S, V = DEFAULT_SVD_SOLVER(R1)
    r = B.shape[1]
    R, Q = rq(V[:, r:])
    Gamma = R[n:, n - r:]
    Z = R[:n, n - r:]
    Xhat = -np.dot(Z, np.linalg.inv(Gamma))
    return Xhat


def exact_dmd_decompose(X, Y, rank):
    Ux, Sx, Vx = DEFAULT_SVD_SOLVER(X)
    Ux = Ux[:, :rank]
    Sx = Sx[:rank]
    Sx = np.diag(Sx)
    Vx = Vx[:, :rank]
    # Project A onto the leading r principal components of X.
    Atilde = (Ux.T @ Y) @ Vx @ np.linalg.pinv(Sx)
    def A(v): return np.dot(a=Ux, b=np.dot(a=Atilde, b=np.dot(a=Ux.T, b=v)))
    # Diagonalise linear operator
    eigen_vals, eigen_vectors = np.linalg.eig(Atilde)
    eigen_vals = np.diag(eigen_vals)
    # Approximation of operator eigenvectors
    eigen_vectors = Y @ Vx @ np.linalg.pinv(Sx) @ eigen_vectors / eigen_vals.T

    return A, eigen_vals, eigen_vectors


def orthogonal_dmd_decompose(X, Y, rank):
    Ux, _, _ = DEFAULT_SVD_SOLVER(X)
    Ux = Ux[:, :rank]
    # Project X (current state) and Y (future state) on leading components of X
    Yproj = Ux.T @ Y
    Xproj = Ux.T @ X
    # A_proj is constrained to be a unitary matrix and the minimization problem is argmin (A.T @ A = I) |Y-AX|_frob
    # The solution of A_proj is obtained by Schonemann A = Uyx,@ Vyx.T
    Uyx, _, Vyx = DEFAULT_SVD_SOLVER(Yproj @ Xproj.T)
    Aproj = Uyx @ Vyx.T
    def A(x): return np.dot(a=Ux, b=np.dot(a=Aproj, b=np.dot(a=Ux.T, b=x)))
    # Diagonalise unitary operator
    eVals, eVecs = np.linalg.eig(Aproj)
    eigen_vals = np.diag(eVals)
    # Approximation of operator eigenvectors
    eigen_vectors = Ux @ eVecs
    return A, eigen_vals, eigen_vectors


def symmetric_decompose(X, Y, rank):
    Ux, S, V = DEFAULT_SVD_SOLVER(X)
    C = np.dot(Ux.T, np.dot(Y, V))
    C1 = C
    if rank is None:
        np.linalg.matrix_rank(X)
    Ux = Ux[:, :rank]
    Yf = np.zeros((rank, rank))
    for i in range(rank):
        Yf[i, i] = np.real(C1[i, i]) / S[i]
        for j in range(i + 1, rank):
            Yf[i, j] = (S[i] * np.conj(C1[j, i]) + S[j] *
                        C1[i, j]) / (S[i] ** 2 + S[j] ** 2)
    Yf = Yf + Yf.T - np.diag(np.diag(np.real(Yf)))
    # elif method == 'skewsymmetric':
    #     for i in range(r):
    #         Yf[i, i] = 1j * np.imag(C1[i, i]) / S[i, i]
    #         for j in range(i + 1, r):
    #             Yf[i, j] = (-S[i, i] * np.conj(C1[j, i]) + S[j, j] * (C1[i, j])) / (S[i, i] ** 2 + S[j, j] ** 2)
    #     Yf = Yf - Yf.T - 1j * np.diag(np.diag(np.imag(Yf)))
    def A(v): return np.dot(Ux, np.dot(Yf, np.dot(Ux.T, v)))
    return A


def hankel_decompose(X, Y, rank):
    nx, nt = X.shape
    # J = np.eye(nx)
    J = np.fliplr(np.eye(nx))
    # Define the left matrix
    A = np.fft.fft(np.concatenate(
        (np.eye(nx), np.zeros((nx, nx))), axis=1), axis=0) / np.sqrt(2 * nx)
    # Define the right matrix
    B = np.fft.fft(np.vstack((np.dot(J, X), np.zeros((nt, nx)))),
                   axis=0) / np.sqrt(2 * nx)
    BtB = B.T @ B
    # Fast computation of A * A
    AAt = np.fft.ifft(np.fft.fft(np.concatenate(
        (np.eye(nx), np.zeros((nx, nx))), axis=1)).T).T
    # Construct the RHS of the linear system
    y = np.diag(np.dot(np.dot(AAt.T, np.conj(Y)), B))
    # Construct the matrix for the linear system
    L = AAt @ BtB
    # Solve the linear system
    d = np.append(y[0:len(y) - 1] / L[0:len(L) - 1, 0:len(L) - 1], 0)
    # Convert the eigenvalues into the circulant matrix
    newA = np.fft.ifft(np.fft.fft(np.diag(d))).T
    # Extract the Toeplitz  matrix from the circulant matrix
    A = newA[0:nx, 0:nx] * J
    return A
