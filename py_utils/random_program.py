from scipy import sparse
import cvxpy as cp
import diffcp
import numpy as np


def random_cone_prog(m, n, cone_dict, dense=False):
    """Returns the problem data of a random cone program."""
    cone_list = diffcp.cones.parse_cone_dict(cone_dict)
    z = np.random.randn(m)
    s_star = diffcp.cones.pi(z, cone_list, dual=False)
    y_star = s_star - z

    A = np.random.randn(m, n)
    if not dense:
        A = sparse.csr_matrix(A)
    x_star = np.random.randn(n)
    b = A @ x_star + s_star
    c = -A.T @ y_star
    return dict(A=A, b=b, c=c, x_star=x_star, y_star=y_star, s_star=s_star)


def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data["dims"])
    return data["A"], data["b"], data["c"], cone_dims


def randn_symm(n):
    A = np.random.randn(n, n)
    return (A + A.T) / 2


def randn_psd(n):
    A = 1. / 10 * np.random.randn(n, n)
    return A@A.T


def random_sdp(n, p, dense=False):
    C = randn_psd(n)
    As = [randn_symm(n) for _ in range(p)]
    Bs = np.random.randn(p)

    # Extract problem data using cvxpy
    X = cp.Variable((n, n), PSD=True)
    objective = cp.trace(C@X)
    constraints = [cp.trace(As[i]@X) == Bs[i] for i in range(p)]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    A, b, c, cone_dims = scs_data_from_cvxpy_problem(prob)

    if dense:
        A = A.toarray()
    return dict(A=A, b=b, c=c), cone_dims






