import diffcp
import numpy as np
from scipy import sparse
from py_utils.random_program import random_cone_prog
np.set_printoptions(precision=5, suppress=True)


cone_dict = {
    'f': 3,
    'l': 3,
    'q': [5]
}

m = 3 + 3 + 5
n = 5

np.random.seed(0)

program = random_cone_prog(m, n)
A, b, c = program["A"], program["b"], program["c"]

m, n = A.shape
x, y, s, D, DT = diffcp.solve_and_derivative(A, b, c, cone_dict)

# evaluate the derivative
nonzeros = A.nonzero()
data = 1e-4 * np.random.randn(A.size)
dA = sparse.csc_matrix((data, nonzeros), shape=A.shape)
db = 1e-4 * np.random.randn(m)
dc = 1e-4 * np.random.randn(n)
dx, dy, ds = D(dA, db, dc)
print(dx)

# evaluate the adjoint of the derivative
dx = c
dy = np.zeros(m)
ds = np.zeros(m)
dA, db, dc = DT(dx, dy, ds)
print(dc)
