from scipy.sparse import csc_matrix
import numpy as np

sqrt2 = np.sqrt(2)
sqrt2 = 1


def _symmetric2vec(A):
    B = A * sqrt2
    B[np.diag_indices_from(B)] /= sqrt2
    return B.T[np.triu_indices_from(B)]


def merge_psd_plus_diag(theta_size, z_size, p):
    s_size = theta_size + 3*z_size - p
    A = np.zeros((s_size, theta_size+z_size))

    # === THETA ===
    i, j = 0, 0
    for k in range(p, 0, -1):
        for k_ in range(k):
            A[i, j] = 1 if k_ == 0 else sqrt2
            i += 1
            j += 1
        i += p

    # === Z ===
    i, j = p, theta_size+p
    for d in range(p):
        zd_ix = theta_size + d
        A[i, zd_ix] = sqrt2
        for d_ in range(d):
            A[i-d_-1, j] = sqrt2
            j += 1
        i += (2*p - d)

    # === DIAGONAL ===
    i = int(p**2 + p*(p+1)/2)
    for d in range(p):
        z_ix = theta_size + d
        A[i, z_ix] = 1
        i += (p-d)
    return A


# TODO add lambda constraint
def write_glasso_cone_program(S, lambda_):
    """
    S: empirical covariance matrix.
    lambda: sparsity-inducting regularization term. Currently not included for debugging purposes.
        For now, optimal solution should be the inverse covariance.
    """
    p = S.shape[0]
    theta_size = int(p*(p+1)/2)
    z_size = int(p*(p+1)/2)

    A1 = -merge_psd_plus_diag(theta_size, z_size, p)

    # log-det inequality constraint on t_i
    A2 = np.zeros((3*p, z_size+p))
    b2 = np.zeros(3*p)
    for d in range(p):
        z_ix = d
        t_ix = z_size+d
        A2[d*3+2, z_ix] = -1
        b2[d*3+1] = 1
        A2[d*3, t_ix] = -1
    print(A2)

    # Equality constraint on t
    A3 = np.zeros((1, p+1))
    A3[0, :-1] = -1
    A3[0, -1] = 1

    # combine constraint matrices
    B1 = np.hstack([A1, np.zeros((A1.shape[0], p+1))])
    B2 = np.hstack([np.zeros((A2.shape[0], theta_size)), A2, np.zeros((A2.shape[0], 1))])
    B3 = np.hstack([np.zeros((1, theta_size+z_size)), A3])
    # should all have same number of columns (theta_size + z_size + t_size + 1)
    A = np.vstack([
        B1,
        B2,
        B3
    ])
    b = np.zeros(A.shape[0])
    b[A1.shape[0]:(A1.shape[0]+len(b2))] = b2

    c = np.zeros(A.shape[1])
    c[:theta_size] = _symmetric2vec(S)
    c[-1] = -1

    cone_dict = {
        "f": 1,  # zero cone
        "s": 2*p,  # PSD
        "ep": p,  # exponential cone
    }
    A = csc_matrix(A)

    # reshape so that zero cone is first, then PSD, then exponential
    A_ = A.copy()
    A_[1:] = A[0:-1]
    A_[0] = A[-1]
    b_ = b.copy()
    b_[1:] = b[:-1]
    b_[0] = b[-1]
    return A_, b_, c, cone_dict

