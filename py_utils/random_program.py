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
    A = sparse.csc_matrix(np.random.randn(m, n))
    x_star = np.random.randn(n)
    b = A @ x_star + s_star
    c = -A.T @ y_star
    return dict(A=A, b=b, c=c, x_star=x_star, y_star=y_star, s_star=s_star)


def _vec2str(v):
    return "\t".join(map(str, v))


def _str2vec(s, t):
    return np.array([t(val) for val in s[:-1].split("\t")])


# TODO: dense
def save_cone_program(file, program, dense=False):
    A = program["A"]
    with open(file, "w") as file:
        if dense:
            vals = A.T.flatten()  # column major order
            file.write(_vec2str(vals))
            file.write("\n")
        else:
            row_ixs, col_ixs = A.nonzero()
            file.write(_vec2str(row_ixs))
            file.write("\n")
            file.write(_vec2str(col_ixs))
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


# TODO: dense
def load_cone_program(file, dense=False):
    with open(file, "r") as file:
        lines = file.readlines()
        if dense:
            b = _str2vec(lines[1], float)
            c = _str2vec(lines[2], float)
            vals = _str2vec(lines[0], float)
            A = vals.reshape(len(b), len(c))  # column major order?
        else:
            b = _str2vec(lines[3], float)
            c = _str2vec(lines[4], float)
            row_ixs, col_ixs, vals = lines[:3]
            row_ixs = _str2vec(row_ixs, int)
            col_ixs = _str2vec(col_ixs, int)
            vals = _str2vec(vals, float)
            A = csr_matrix((vals, (row_ixs, col_ixs)), shape=(len(b), len(c)))

        if (dense and len(lines) == 6) or len(lines) == 8:  # x_star, y_star, s_star exist
            x_star = _str2vec(lines[-3], float)
            y_star = _str2vec(lines[-2], float)
            s_star = _str2vec(lines[-1], float)
            return dict(A=A, b=b, c=c, x_star=x_star, y_star=y_star, s_star=s_star)
        else:
            return dict(A=A, b=b, c=c)


if __name__ == '__main__':
    cone_dict = {
        diffcp.ZERO: 3,
        diffcp.POS: 3,
        diffcp.SOC: [5]
    }

    m = 3 + 3 + 5
    n = 5

    program = random_cone_prog(m, n, cone_dict)
    save_cone_program("random_programs/test.txt", program)
    p = load_cone_program("random_programs/test.txt")

    program_dense = program.copy()
    program_dense["A"] = program["A"].toarray()
    save_cone_program("random_programs/test_dense.txt", program_dense, dense=True)
    p_dense = load_cone_program("random_programs/test_dense.txt", dense=True)
