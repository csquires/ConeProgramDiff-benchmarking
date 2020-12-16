import cvxpy as cp
import numpy as np
import diffcp
from scipy.sparse import csc_matrix
import numpy as np
import itertools as itr

sqrt2 = np.sqrt(2)
sqrt2 = 1
# sqrt2 = 1/np.sqrt(2)

p = 3
lambda_ = 1
A = np.random.normal(size=(p*2, p))
S = A.T @ A
print("true inverse")
print(np.linalg.inv(S))

print("original form, no l1 penalty")
X = cp.Variable((p, p), symmetric=True)
constraints = [X >> 0]
objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - cp.log_det(X))
prob = cp.Problem(objective, constraints)
prob.solve()
sol = X.value
print(sol)

print("original form, with l1 penalty")
X = cp.Variable((p, p), symmetric=True)
constraints = [X >> 0]
objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - cp.log_det(X) + lambda_*cp.pnorm(X, 1))
prob = cp.Problem(objective, constraints)
prob.solve()
sol = X.value
print(sol)

print("form with l1 penalty expressed via extra variables")
X = cp.Variable((p, p), symmetric=True)
M = cp.Variable((p, p), symmetric=True)
constraints = [
    X >> 0,
    *[M[i, j] >= X[i, j] for i, j in itr.product(range(p), range(p))],
    *[M[i, j] >= -X[i, j] for i, j in itr.product(range(p), range(p))],
]
objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - cp.log_det(X) + lambda_*cp.sum(M))
prob = cp.Problem(objective, constraints)
prob.solve()
sol = X.value
print(sol)

# print("form with t added")
# X = cp.Variable((p, p), symmetric=True)
# t = cp.Variable()
# constraints = [X >> 0, t <= cp.log_det(X)]
# objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - t)
# prob = cp.Problem(objective, constraints)
# prob.solve()
# sol = X.value
# print(sol)

# print("form with log det expanded")
# X = cp.Variable((p, p), symmetric=True)
# Z = cp.Variable((p, p))
# D = cp.diag(cp.diag(Z))
# A = cp.vstack([
#     cp.hstack([X, Z]),
#     cp.hstack([cp.transpose(Z), D])]
# )
# t = cp.Variable()
# constraints = [
#     A >> 0,
#     t <= cp.sum(cp.log(cp.diag(Z))),
#     Z[0, 1] == 0,
#     Z[0, 2] == 0,
#     Z[1, 2] == 0,
# ]
# objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - t)
# prob = cp.Problem(objective, constraints)
# prob.solve()
# sol = X.value
# print(sol)
#
# print("form with t as a sum")
# X = cp.Variable((p, p), symmetric=True)
# Z = cp.Variable((p, p))
# D = cp.diag(cp.diag(Z))
# A = cp.vstack([
#     cp.hstack([X, Z]),
#     cp.hstack([cp.transpose(Z), D])]
# )
# t = cp.Variable()
# t1 = cp.Variable()
# t2 = cp.Variable()
# t3 = cp.Variable()
# constraints = [
#     A >> 0,
#     Z[0, 1] == 0,
#     Z[0, 2] == 0,
#     Z[1, 2] == 0,
#     # t1 <= cp.log(Z[0, 0]),
#     # t2 <= cp.log(Z[1, 1]),
#     # t3 <= cp.log(Z[2, 2]),
#     cp.constraints.exponential.ExpCone(t1, 1, Z[0, 0]),
#     cp.constraints.exponential.ExpCone(t2, 1, Z[1, 1]),
#     cp.constraints.exponential.ExpCone(t3, 1, Z[2, 2]),
#     t == t1 + t2 + t3,
# ]
# objective = cp.Minimize(cp.sum(cp.multiply(S, X)) - t)
# prob = cp.Problem(objective, constraints)
# prob.solve()
# sol = X.value
# print(sol)


def _symmetric2vec(A):
    B = A * sqrt2
    B[np.diag_indices_from(B)] /= sqrt2
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


# if __name__ == '__main__':
#     import sympy
#     from sympy import Matrix
#     import scs
#     import ecos
#     a = np.random.normal(size=(100, 3))
#     S = np.cov(a, rowvar=False)
#     A, b, c, cone_dict = write_glasso_cone_program(S, 1.)
#     x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
#     print(_vec2symmetric((A @ x)[1:22], 6))
#     x = Matrix(sympy.symbols([
#         "theta11", "theta21", "theta31", "theta22", "theta32", "theta33",
#         "z11", "z22", "z33", "z21", "z31", "z32",
#         "t1", "t2", "t3", "t"
#     ]))
#
#     print(Matrix(b) - Matrix(A.toarray()) @ x)
#     print(Matrix(c).T @ x)
#     sol = diffcp.solve_and_derivative(A, b, c, cone_dict)
#     K = np.linalg.inv(S)
#     sol = scs.solve(dict(A=A, b=b, c=c), cone_dict, eps=1e-15, max_iters=10000, verbose=True, acceleration_lookback=1)
#     x = sol["x"]
#     p = S.shape[0]
#     d = int(S.shape[0]*(S.shape[0]+1)/2)
#     theta = _vec2symmetric(x[:d], p)
#     print(theta[:3, :3])
#     print(np.linalg.inv(S)[:3, :3])
#     print(np.linalg.norm(theta - np.linalg.inv(S), "fro"))

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
