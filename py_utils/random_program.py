import numpy as np
from scipy import sparse
import diffcp
import os
import ipdb
from scipy.sparse import csr_matrix


def random_cone_prog(m, n, cone_dict):
    """Returns the problem data of a random cone program."""
    cone_list = diffcp.cones.parse_cone_dict(cone_dict)
    z = np.random.randn(m)
    s_star = diffcp.cones.pi(z, cone_list, dual=False)
    y_star = s_star - z
    A = sparse.csr_matrix(np.random.randn(m, n))
    x_star = np.random.randn(n)
    b = A @ x_star + s_star
    c = -A.T @ y_star
    return dict(A=A, b=b, c=c, x_star=x_star, y_star=y_star, s_star=s_star)


def _vec2str(v):
    return "\t".join(map(str, v))


def _str2vec(s, t):
    return np.array([t(val) for val in s[:-1].split("\t")])


def save_cone_program(file, program, dense=False):
    A = program["A"]
    with open(file, "w") as file:
        if dense:
            vals = A.T.flatten()  # column major order
            file.write(_vec2str(vals))
            file.write("\n")
        else:
            row_ixs, col_ixs = A.nonzero()
            file.write(_vec2str(row_ixs+1))
            file.write("\n")
            file.write(_vec2str(col_ixs+1))
            file.write("\n")
            file.write(_vec2str(A.data))
            file.write("\n")

        file.write(_vec2str(program['b']))
        file.write("\n")
        file.write(_vec2str(program['c']))
        file.write("\n")
        if "x_star" in program:
            file.write(_vec2str(program["x_star"]))
            file.write("\n")
            file.write(_vec2str(program["y_star"]))
            file.write("\n")
            file.write(_vec2str(program["s_star"]))


def load_cone_program(file, dense=False):
    with open(file, "r") as file:
        lines = file.readlines()
        if dense:
            b = _str2vec(lines[1], float)
            c = _str2vec(lines[2], float)
            vals = _str2vec(lines[0], float)
            A = vals.reshape(len(c), len(b)).T  # column major order?
        else:
            b = _str2vec(lines[3], float)
            c = _str2vec(lines[4], float)
            row_ixs, col_ixs, vals = lines[:3]
            row_ixs = _str2vec(row_ixs, int)
            col_ixs = _str2vec(col_ixs, int)
            vals = _str2vec(vals, float)
            A = csr_matrix((vals, (row_ixs-1, col_ixs-1)), shape=(len(b), len(c)))

        if (dense and len(lines) == 6) or len(lines) == 8:  # x_star, y_star, s_star exist
            x_star = _str2vec(lines[-3], float)
            y_star = _str2vec(lines[-2], float)
            s_star = _str2vec(lines[-1], float)
            return dict(A=A, b=b, c=c, x_star=x_star, y_star=y_star, s_star=s_star)
        else:
            return dict(A=A, b=b, c=c)


# do all dense
def save_derivative_and_adjoint(file, input_sensitivities, reverse_sensitivities):
    dA, db, dc = input_sensitivities
    dx, dy, ds = reverse_sensitivities

    # first line: dA
    # second line: db
    # third line: dc

    # fourth line: dx
    # fifth line: dy
    # sixth line: ds

    # compute dx, dy, ds from dA, db, dc

    # compute dA, db, dc from dx, dy, dx
    pass


# do all dense
def load_derivative_and_adjoint(file):
    # first line: dA
    # second line: db
    # third line: dc

    # fourth line: dx
    # fifth line: dy
    # sixth line: ds

    # compute dx, dy, ds from dA, db, dc

    # compute dA, db, dc from dx, dy, dx
    pass


if __name__ == '__main__':
    import random
    np.random.seed(120312)
    random.seed(120931)

    cone_dict = {
        diffcp.ZERO: 3,
        diffcp.POS: 3,
        diffcp.SOC: [5]
    }

    m = 3 + 3 + 5
    n = 5

    folder = "/home/csquires/Desktop/"
    program = random_cone_prog(m, n, cone_dict)
    save_cone_program(f"{folder}/test_sparse_py.txt", program)
    program_dense = program.copy()
    program_dense["A"] = program["A"].toarray()
    save_cone_program(f"{folder}/test_dense_py.txt", program_dense, dense=True)

    # in Julia, read and write back to files

    # p = load_cone_program(f"{folder}/test_sparse_jl.txt")
    p = load_cone_program(f"{folder}/test_sparse_jl.txt")
    p_dense = load_cone_program(f"{folder}/test_dense_jl.txt", dense=True)

    for k in program:
        print(np.max(program[k] - p[k]))
        # print(np.all(program_dense[k] == p_dense[k]))
