from py_utils.random_program import random_cone_prog, random_sdp
from py_utils.loaders import save_cone_program
import random
import numpy as np
from tqdm import trange
30
np.random.seed(1298731)
random.seed(1298731)

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
num_programs = 2
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
random.seed(1231)
np.random.seed(1231)
num_programs = 2
n = 10
p = 5
name = "sdp-small"
for program_num in trange(num_programs):
    program, cone_dims = random_sdp(n, p)
    print(f"Cone dimensions: {cone_dims}")
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === SDP LARGE
print("=== GENERATING LARGE SDPs ===")
num_programs = 50
n = 50
p = 25
name = "sdp-large"
for program_num in trange(num_programs):
    program, cone_dims = random_sdp(n, p)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === EXPONENTIAL SMALL
print("=== GENERATING SMALL EXPONENTIAL ===")
num_programs = 2
K = {
    'ep': 2  # EXPONENTIAL
}
m = 2*3
n = 5
name = "exponential-small"
for program_num in trange(num_programs):
    program = random_cone_prog(m, n, K)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)


# === EXPONENTIAL SMALL
print("=== GENERATING LARGE EXPONENTIAL ===")
num_programs = 2
K = {
    'ep': 20  # EXPONENTIAL
}
m = 20*3
n = 20
name = "exponential-large"
for program_num in trange(num_programs):
    program = random_cone_prog(m, n, K)
    save_cone_program(f"benchmarking/programs/{name}/{program_num}.txt", program)

