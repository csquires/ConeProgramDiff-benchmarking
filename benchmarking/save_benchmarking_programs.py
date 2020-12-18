from py_utils.random_program import random_cone_prog
from py_utils.loaders import save_cone_program
import random
import numpy as np

np.random.seed(1298731)
random.seed(1298731)

num_programs = 30

K = {
    'f': 3,  # ZERO
    'l': 3,  # POS
    'q': [5]  # SOC
}
m = 3 + 3 + 5
n = 5

for program_num in range(num_programs):
    program = random_cone_prog(m, n, K)
    save_cone_program(f"benchmarking/programs/program{program_num}.txt", program)


