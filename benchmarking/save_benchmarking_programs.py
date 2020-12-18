from py_utils.random_program import random_cone_prog, random_sdp
from py_utils.loaders import save_cone_program
import random
import numpy as np
from tqdm import trange

np.random.seed(31231)
random.seed(321)

# === SOC SMALL
print("=== GENERATING SMALL SOCs ===")
num_programs = 30
K = {
    'f': 3,  # ZERO
    'l': 3,  # POS
    'q': [3]  # SOC
}
m = 3 + 3 + 3
n = 5
name = "soc-small"
for program_num in trange(num_programs):
    program = random_cone_prog(m, n, K)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === SOC LARGE
print("=== GENERATING LARGE SOCs ===")
num_programs = 30
K = {
    'f': 3,  # ZERO
    'l': 3,  # POS
    'q': [20]  # SOC
}
m = 3 + 3 + 20
n = 50
name = "soc-large"
for program_num in trange(num_programs):
    program = random_cone_prog(m, n, K)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === SDP SMALL
print("=== GENERATING SMALL SDPs ===")
num_programs = 30
n = 10
p = 5
name = "sdp-small"
for program_num in trange(num_programs):
    program, cone_dims = random_sdp(n, p)
    print(f"Cone dimensions: {cone_dims}")
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === SDP LARGE
print("=== GENERATING LARGE SDPs ===")
num_programs = 30
n = 20
p = 10
name = "sdp-large"
for program_num in trange(num_programs):
    program, cone_dims = random_sdp(n, p)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === EXPONENTIAL SMALL
print("=== GENERATING SMALL EXPONENTIAL ===")
num_programs = 30
K = {
    'ep': 2  # EXPONENTIAL
}
m = 2*3
n = 5
name = "exponential-small"
for program_num in trange(num_programs):
    program = random_cone_prog(m, n, K)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === EXPONENTIAL LARGE
print("=== GENERATING LARGE EXPONENTIAL ===")
num_programs = 30
K = {
    'ep': 4  # EXPONENTIAL
}
m = 4*3
n = 10
name = "exponential-large"
for program_num in trange(num_programs):
    program = random_cone_prog(m, n, K)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)

