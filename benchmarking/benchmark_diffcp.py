from py_utils.loaders import load_cone_program
import diffcp
from time import perf_counter
from tqdm import trange
import numpy as np
from scipy.sparse import csc_matrix


def solve_and_time(folder, cone_dict, num_programs):
    print(f"Solving programs in {folder}")

    times = np.zeros(num_programs)
    for program_num in trange(num_programs):
        program = load_cone_program(f"{folder}/{program_num}.txt")
        A, b, c = program["A"], program["b"], program["c"]
        A = csc_matrix(A)

        start = perf_counter()
        diffcp.solve_and_derivative(A, b, c, cone_dict, eps=1e-5)
        time_taken = perf_counter() - start

        times[program_num] = time_taken

    print(f"Took an average of {np.mean(times)} seconds")
    np.savetxt(f"{folder}_diffcp_times.txt", times)


# === SOC SMALL
print("=== SOLVING SMALL SOCs WITH DIFFCP ===")
num_programs = 30
K = {
    'f': 3,  # ZERO
    'l': 3,  # POS
    'q': [3]  # SOC
}
name = "soc-small"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)

# === SOC LARGE
print("=== SOLVING LARGE SOCs WITH DIFFCP ===")
num_programs = 2
K = {
    'f': 3,  # ZERO
    'l': 3,  # POS
    'q': [20]  # SOC
}
name = "soc-large"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === SDP SMALL
print("=== SOLVING SMALL SDPs WITH DIFFCP ===")
num_programs = 2
K = {
    'f': 5,  # ZERO
    's': [10],  # SD
}
name = "sdp-small"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === SDP LARGE
print("=== SOLVING LARGE SDPs WITH DIFFCP ===")
num_programs = 2
K = {
    'f': 25,  # ZERO
    's': [50],  # SD
}
name = "sdp-large"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === EXPONENTIAL SMALL
print("=== SOLVING SMALL EXPONENTIAL WITH DIFFCP ===")
num_programs = 2
K = {
    'ep': 2
}
name = "exponential-small"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === EXPONENTIAL SMALL
print("=== SOLVING LARGE EXPONENTIAL WITH DIFFCP ===")
num_programs = 2
K = {
    'ep': 20
}
name = "exponential-large"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)
