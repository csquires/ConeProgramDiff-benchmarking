import cvxpy as cp
import numpy as np
import diffcp
from scipy.sparse import csc_matrix
import numpy as np

sqrt2 = np.sqrt(2)

p = 5
lambda_ = 0
A = np.random.normal(size=(p*2, p))
S = A.T @ A
X = cp.Variable((p, p), symmetric=True)
constraints = [X >> 0]
objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - cp.log_det(X) + lambda_ * cp.norm(X, 1))
prob = cp.Problem(objective, constraints)
prob.solve(requires_grad=True)
sol = X.value


def _symmetric2vec(A):
    B = A + A.T
    B[np.diag_indices_from(B)] /= 2
    return B.T[np.triu_indices_from(B)]


def _vec2symmetric(a, dim):
    A = np.zeros((dim, dim))
    A[np.triu_indices_from(A)] = a
    A = A + A.T
    A[np.diag_indices_from(A)] /= 2
    return A


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
    i, j = p, theta_size
    for k in range(p, 0, -1):
        for k_ in range(k):
            A[i, j] = sqrt2
            i += 1
            j += 1
        i += p

    # === DIAGONAL ===
    j = theta_size  # start at first z1
    i = int(p**2 + p*(p+1)/2)
    for d in range(p):
        A[i, j] = 1
        j += p-d
        i += (p-d)
    return A


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
    z_ix = 0
    b2 = np.zeros(3*p)
    for d in range(p):
        t_ix = z_size+d
        A2[d*3, z_ix] = -1
        b2[d*3+1] = 1
        A2[d*3+2, t_ix] = -1
        z_ix += p-d
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


if __name__ == '__main__':
    import sympy
    from sympy import Matrix
    import scs
    import ecos
    a = np.random.normal(size=(100, 3))
    S = np.corrcoef(a, rowvar=False)
    A, b, c, cone_dict = write_glasso_cone_program(S, 1.)
    x = Matrix(sympy.symbols([
        "theta11", "theta21", "theta31", "theta22", "theta32", "theta33",
        "z11", "z21", "z31", "z22", "z32", "z33",
        "t1", "t2", "t3", "t"
    ]))

    print(Matrix(b) - Matrix(A.toarray()) @ x)
    print(Matrix(c).T @ x)
    # sol = diffcp.solve_and_derivative(A, b, c, cone_dict)
    K = np.linalg.inv(S)
    sol = scs.solve(dict(A=A, b=b, c=c), cone_dict, eps=1e-12, max_iters=2000, verbose=True, acceleration_lookback=10)
    x = sol["x"]
    theta = _vec2symmetric(x[:6], 3)

    # theta = [1, 2, 3, 4, 5, 6]
    # z = [7, 8, 9, 10, 11, 12]
    # x = np.array([*theta, *z])
    # A = merge_psd_plus_diag(len(theta), len(z), 3)
    # merged = A @ x
    # merged_true = np.array([1, 2, 3, 7, 8, 9, 4, 5, 0, 10, 11, 6, 0, 0, 12, 7, 0, 0, 10, 0, 12])
    # m = _vec2symmetric(merged, 6)
    # merged2 = _symmetric2vec(m)
    # print(np.all(merged == merged2))
    # print(np.all(merged == merged_true))
