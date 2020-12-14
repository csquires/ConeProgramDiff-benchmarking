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


def cone_form_glasso(S):
    p = S.shape[0]
    theta_size = int(p*(p+1)/2)
    z_size = int(p*(p+1)/2)
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

    # Equality constraint on t
    A3 = np.zeros((1, p+1))
    A3[0, :-1] = -1
    A3[0, -1] = 1

    # combine constraint matrices
    B1 = np.hstack([A1, np.zeros((A1.shape[0], p+1))])
    B2 = np.hstack([np.zeros((A2.shape[0], theta_size)), A2, np.zeros((A2.shape[0], 1))])
    B3 = np.hstack([np.zeros((1, theta_size+z_size)), A3])
    print(B1.shape)
    print(B2.shape)
    print(B3.shape)
    # should all have same number of columns (theta_size + z_size + t_size + 1)
    A = np.vstack([
        B1,
        B2,
        B3
    ])
    b = np.zeros(A.shape[0])
    # TODO: fill in b2

    # TODO: fill in c from S
    c = np.zeros(A.shape[1])

    return A, b, c


if __name__ == '__main__':
    A = np.random.normal(size=(10, 3))
    S = A.T @ A
    cone_form_glasso(S)

    theta = [1, 2, 3, 4, 5, 6]
    z = [7, 8, 9, 10, 11, 12]
    x = np.array([*theta, *z])
    A = merge_psd_plus_diag(len(theta), len(z), 3)
    merged = A @ x
    merged_true = np.array([1, 2, 3, 7, 8, 9, 4, 5, 0, 10, 11, 6, 0, 0, 12, 7, 0, 0, 10, 0, 12])
    print(np.all(merged == merged_true))
