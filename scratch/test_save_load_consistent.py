from py_utils.random_program import random_sdp
from py_utils.loaders import save_cone_program, load_cone_program
import os
import random
import numpy as np
np.random.seed(12093821)
random.seed(12093821)

# === delete
if os.path.exists("scratch/sdp_test.txt"):
    os.remove("scratch/sdp_test.txt")
    os.remove("scratch/sdp_test_jl.txt")

# === create
program, cone_dict = random_sdp(50, 25)
print(program["A"][:5, :5])
print(program["b"][:5])
print(program["c"][:5])
print("Saving")
save_cone_program("scratch/sdp_test.txt", program, dense=False)

# === to Julia
print("Loading/Saving in Julia")
os.system("julia scratch/load_test_sdp.jl")

# === and back
print("Loading")
program2 = load_cone_program("scratch/sdp_test_jl.txt")

# === and check
A, b, c = program["A"], program["b"], program["c"]
A2, b2, c2 = program2["A"], program2["b"], program2["c"]
assert (A != A2).nnz == 0
assert (b == b2).all()
assert (c == c2).all()
