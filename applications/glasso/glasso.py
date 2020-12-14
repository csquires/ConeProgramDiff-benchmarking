import cvxpy as cp
import numpy as np

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


def merge_psd_plus_diag(theta, z, p):
    theta_size = len(theta)
    z_size = len(z)
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


def cone_form_glasso(S):
    p = S.shape[0]
    theta_size = p*(p+1)/2
    z_size = p*(p+1)/2
    t_size = p

    A1 = merge_psd_plus_diag(theta_size, z_size, p)

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
    pass

    # combine constraint matrices
    A = np.hstack([
        np.vstack([A1, np.zeros(A1.shape[0], p+1)]),
        np.vstack([np.zeros(), A2, np.zeros()]),
        np.vstack([np.zeros(), 0, None])
    ])


if __name__ == '__main__':
    theta = [1, 2, 3, 4, 5, 6]
    z = [7, 8, 9, 10, 11, 12]
    x = np.array([*theta, *z])
    A = merge_psd_plus_diag(theta, z, 3)
    merged = A @ x
    merged_true = np.array([1, 2, 3, 7, 8, 9, 4, 5, 0, 10, 11, 6, 0, 0, 12, 7, 0, 0, 10, 0, 12])
    print(np.all(merged == merged_true))
