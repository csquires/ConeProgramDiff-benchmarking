import numpy as np
from scipy import sparse
import diffcp
import os


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


def save_cone_program(folder, program):
    os.makedirs(folder, exist_ok=True)

    A = program["A"]
    row_ixs, col_ixs = A.nonzero()
    with open(f"{folder}/A.txt", "w") as file:
        file.write("\t".join(map(str, row_ixs)))
        file.write("\n")
        file.write("\t".join(map(str, col_ixs)))
        file.write("\n")
        file.write("\t".join(map(str, A.data)))
        file.write("\n")
    np.savetxt(f'{folder}/b.txt', program['b'])
    np.savetxt(f'{folder}/c.txt', program['c'])
    if "x_star" in program:
        np.savetxt(f'{folder}/x_star.txt', program["x_star"])
        np.savetxt(f'{folder}/y_star.txt', program["y_star"])
        np.savetxt(f'{folder}/s_star.txt', program["s_star"])


def load_cone_program(folder):
    with open(f"{folder}/A.txt", "r") as file:
        row_ixs, col_ixs, vals = file.readlines()
        # TODO: finish formatting into sparse matrix
    b = np.loadtxt(f'{folder}/b.txt')
    c = np.loadtxt(f'{folder}/c.txt')
    if os.path.exists(f'{folder}/x_star.txt'):
        x_star = np.loadtxt(f'{folder}/x_star.txt')
        y_star = np.loadtxt(f'{folder}/y_star.txt')
        s_star = np.loadtxt(f'{folder}/s_star.txt')
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
    save_cone_program("random_programs/test", program)
    p = load_cone_program("random_programs/test")
