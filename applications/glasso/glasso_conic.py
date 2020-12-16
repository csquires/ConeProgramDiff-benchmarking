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
    z_size = theta_size
    m_size = theta_size

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

    # Equality constraint on t
    A3 = np.zeros((1, p+1))
    A3[0, :-1] = -1
    A3[0, -1] = 1

    # Absolute value constraint on M
    A4 = np.zeros((2*theta_size, theta_size + z_size + p + 1 + m_size))
    # M_ij - Theta_ij >= 0
    A4[:theta_size, :theta_size] = -np.eye(theta_size)
    A4[:theta_size, -m_size:] = np.eye(m_size)
    # M_ij + Theta_ij >= 0
    A4[theta_size:, :theta_size] = np.eye(theta_size)
    A4[theta_size:, -m_size:] = np.eye(m_size)

    # combine constraint matrices
    B1 = np.hstack([A1, np.zeros((A1.shape[0], p+1+m_size))])
    B2 = np.hstack([np.zeros((A2.shape[0], theta_size)), A2, np.zeros((A2.shape[0], 1+m_size))])
    B3 = np.hstack([np.zeros((1, theta_size+z_size)), A3, np.zeros((1, m_size))])
    # should all have same number of columns (theta_size + z_size + t_size + 1)
    A = np.vstack([
        B1,
        B2,
        B3,
        -A4
    ])
    b = np.zeros(A.shape[0])
    b[A1.shape[0]:(A1.shape[0]+len(b2))] = b2

    c = np.zeros(A.shape[1])
    c[:theta_size] = _symmetric2vec(S)
    c[-m_size:] = lambda_
    c[theta_size+z_size+p] = -1

    cone_dict = {
        "f": 1,  # zero cone
        "l": 2*theta_size,  # nonnegative cone
        "s": 2*p,  # PSD
        "ep": p,  # exponential cone
    }
    A = csc_matrix(A)

    # reshape so that zero cone is first, then nonnegative, then PSD, then exponential
    zero_plus_nonnegative_dims = 2*theta_size + 1
    A_ = A.copy()
    A_[zero_plus_nonnegative_dims:] = A[0:-zero_plus_nonnegative_dims]
    A_[:zero_plus_nonnegative_dims] = A[-zero_plus_nonnegative_dims:]
    b_ = b.copy()
    b_[zero_plus_nonnegative_dims:] = b[:-zero_plus_nonnegative_dims]
    b_[:zero_plus_nonnegative_dims] = b[-zero_plus_nonnegative_dims:]
    return A_, b_, c, cone_dict


if __name__ == '__main__':
    def _vec2symmetric(a, dim):
        A = np.zeros((dim, dim))
        A[np.triu_indices_from(A)] = a
        A = A + A.T
        A[np.diag_indices_from(A)] /= 2
        return A

    import sympy
    from sympy import Matrix
    import scs
    import cvxpy as cp
    import ecos
    a = np.random.normal(size=(100, 3))
    S = np.cov(a, rowvar=False)
    lambda_ = 1

    A, b, c, cone_dict = write_glasso_cone_program(S, lambda_)
    x = Matrix(sympy.symbols([
        "theta11", "theta21", "theta31", "theta22", "theta32", "theta33",
        "z11", "z22", "z33", "z21", "z31", "z32",
        "t1", "t2", "t3", "t",
        "m11", "m21", "m31", "m22", "m32", "m33"
    ]))

    print("Constraints:")
    print(Matrix(b) - Matrix(A.toarray()) @ x)

    print("Objective:")
    print(Matrix(c).T @ x)

    K = np.linalg.inv(S)
    sol = scs.solve(dict(A=A, b=b, c=c), cone_dict, eps=1e-15, max_iters=10000, verbose=True, acceleration_lookback=1)
    x = sol["x"]
    p = S.shape[0]
    d = int(S.shape[0]*(S.shape[0]+1)/2)
    theta = _vec2symmetric(x[:d], p)
    print("Conic form")
    print(theta)

    print("cvxpy form")
    X = cp.Variable((p, p), symmetric=True)
    constraints = [X >> 0]
    objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - cp.log_det(X) + lambda_*cp.pnorm(X, 1))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    sol = X.value
    print(sol)
