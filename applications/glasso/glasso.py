import cvxpy as cp
import numpy as np
import diffcp
from scipy.sparse import csc_matrix

p = 5
lambda_ = 100
A = np.random.normal(size=(p*2, p))
S = A.T @ A
X = cp.Variable((p, p), symmetric=True)
constraints = [X >> 0]
objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - cp.log_det(X) + lambda_ * cp.norm(X, 1))
prob = cp.Problem(objective, constraints)
prob.solve(requires_grad=True)
sol = X.value


def _symmetric2vec(A):
    return A.T[np.triu_indices_from(A)]


def _vec2symmetric(a, dim):
    A = np.zeros((dim, dim))
    A[np.triu_indices_from(A)] = a
    return A.T


def merge_psd_plus_diag(theta_size, z_size, p):
    s_size = theta_size + 3*z_size - p
    A = np.zeros((s_size, theta_size+z_size))

    # === THETA ===
    i, j = 0, 0
    for k in range(p, 0, -1):
        for k_ in range(k):
            A[i, j] = 1
            i += 1
            j += 1
        i += p

    # === Z ===
    i, j = p, theta_size
    for k in range(p, 0, -1):
        for k_ in range(k):
            A[i, j] = 1
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


def cone_form_glasso(S, lambda_):
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
        A2[d*3, z_ix] = 1
        b2[d*3+1] = 1
        A2[d*3+2, t_ix] = 1
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
        "s": 2*p,  # PSD
        "ep": p,  # exponential cone
        "f": 1,  # zero cone
    }
    A = csc_matrix(A)
    return A, b, c, cone_dict


if __name__ == '__main__':
    a = np.random.normal(size=(10, 3))
    S = a.T @ a
    A, b, c, cone_dict = cone_form_glasso(S, 1.)
    sol = diffcp.solve_and_derivative(A, b, c, cone_dict)

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
