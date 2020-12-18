import cvxpy as cp
import diffcp
import numpy as np
from py_utils.loaders import save_cone_program, save_derivative_and_adjoint, load_derivative_and_adjoint
import time


def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims


def randn_symm(n):
    A = np.random.randn(n, n)
    return (A + A.T) / 2


def randn_psd(n):
    A = 1. / 10 * np.random.randn(n, n)
    return A@A.T


def main(n=3, p=3):
    # Generate problem data
    C = randn_psd(n)
    As = [randn_symm(n) for _ in range(p)]
    Bs = np.random.randn(p)

    # Extract problem data using cvxpy
    X = cp.Variable((n, n), PSD=True)
    objective = cp.trace(C@X)
    constraints = [cp.trace(As[i]@X) == Bs[i] for i in range(p)]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    A, b, c, cone_dims = scs_data_from_cvxpy_problem(prob)
    cone_dims = {'f': 25, 's': [50]}
    print(cone_dims)

    # Compute solution and derivative maps
    print(A.shape, b.shape, c.shape, cone_dims)
    x, y, s, derivative, adjoint_derivative = diffcp.solve_and_derivative(
        A, b, c, cone_dims, eps=1e-5)

    return dict(A=A, b=b, c=c), cone_dims


if __name__ == '__main__':
    # np.random.seed(0)
    from py_utils.loaders import load_cone_program
    program, _ = main(50, 25)
    save_cone_program("programs/test.txt", program)
    program2 = load_cone_program("programs/test.txt")

    A = program["A"]
    A2 = program2["A"]
    assert (A != A2).nnz == 0
    # d = load_derivative_and_adjoint("test_programs/sdp_test_derivatives.txt")
