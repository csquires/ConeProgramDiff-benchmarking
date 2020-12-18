import numpy as np
from scipy.sparse import csr_matrix, issparse, csc_matrix
import os


def _vec2str(v):
    return "\t".join(map(str, v))


def _str2vec(s, t):
    return np.array([t(val) for val in s[:-1].split("\t")])


def ensure_folder(file):
    folder = os.path.dirname(file)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def save_cone_program(file, program, dense=False):
    if dense and issparse(program["A"]):
        raise ValueError("Asked for saving dense, but A is sparse")

    ensure_folder(file)
    A = program["A"]
    with open(file, "w") as file:
        if dense:
            vals = A.T.flatten()  # column major order
            file.write(_vec2str(vals))
            file.write("\n")
        else:
            row_ixs, col_ixs = A.nonzero()
            data = np.array(A[(row_ixs, col_ixs)]).squeeze()
            file.write(_vec2str(row_ixs+1))
            file.write("\n")
            file.write(_vec2str(col_ixs+1))
            file.write("\n")
            file.write(_vec2str(data))
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


def save_derivative_and_adjoint(file, derivative, adjoint_derivative, forward_sensitivities, reverse_sensitivities):
    ensure_folder(file)
    # output order is dA, db, dc, dx, dy, ds.

    with open(file, "w") as file:
        # Forward
        dA, db, dc = forward_sensitivities
        dx, dy, ds = derivative(dA, db, dc)
        #   dA, db, dc
        vals = dA.T.flatten()  # column major order
        file.write(_vec2str(vals))
        file.write("\n")
        file.write(_vec2str(db))
        file.write("\n")
        file.write(_vec2str(dc))
        file.write("\n")
        #   dx, dy, ds
        file.write(_vec2str(dx))
        file.write("\n")
        file.write(_vec2str(dy))
        file.write("\n")
        file.write(_vec2str(ds))
        file.write("\n")

        # Adjoint
        dx, dy, ds = reverse_sensitivities
        dA, db, dc = adjoint_derivative(dx, dy, ds)
        #   dA, db, dc
        vals = dA.toarray().T.flatten()  # column major order
        file.write(_vec2str(vals))
        file.write("\n")
        file.write(_vec2str(db))
        file.write("\n")
        file.write(_vec2str(dc))
        file.write("\n")
        #   dx, dy, ds
        file.write(_vec2str(dx))
        file.write("\n")
        file.write(_vec2str(dy))
        file.write("\n")
        file.write(_vec2str(ds))


def load_derivative_and_adjoint(file):
    # assume input order is dA, db, dc, dx, dy, ds.
    res = dict()
    with open(file, "r") as file:
        lines = file.readlines()
        db = _str2vec(lines[1], float)
        dc = _str2vec(lines[2], float)
        dA = _str2vec(lines[0], float).reshape((len(db), len(dc)))

        dx = _str2vec(lines[3], float)
        dy = _str2vec(lines[4], float)
        ds = _str2vec(lines[5], float)
        res["forward"] = dict(dA=dA, db=db, dc=dc, dx=dx, dy=dy, ds=ds)

        db = _str2vec(lines[7], float)
        dc = _str2vec(lines[8], float)
        dA = _str2vec(lines[6], float).reshape((len(db), len(dc)))

        dx = _str2vec(lines[9], float)
        dy = _str2vec(lines[10], float)
        ds = _str2vec(lines[11], float)
        res["backward"] = dict(dA=dA, db=db, dc=dc, dx=dx, dy=dy, ds=ds)

        return res


if __name__ == '__main__':
    import random
    np.random.seed(120312)
    random.seed(120931)
    import diffcp
    from py_utils.random_program import random_cone_prog

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
