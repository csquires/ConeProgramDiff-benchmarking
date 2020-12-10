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
    return A, b, c


def save_cone_program(folder, A, b, c):
    os.makedirs(folder, exist_ok=True)
    sparse.save_npz(f'{folder}/A.npz', A)
    np.savetxt(f'{folder}/b.txt', b)
    np.savetxt(f'{folder}/c.txt', c)


def load_cone_program(folder):
    A = sparse.load_npz(f'{folder}/A.npz')
    b = np.loadtxt(f'{folder}/b.txt')
    c = np.loadtxt(f'{folder}/c.txt')
    return A, b, c


if __name__ == '__main__':
    cone_dict = {
        diffcp.ZERO: 3,
        diffcp.POS: 3,
        diffcp.SOC: [5]
    }

    m = 3 + 3 + 5
    n = 5

    A, b, c = random_cone_prog(m, n, cone_dict)
    save_cone_program("random_programs/test/", A, b, c)
