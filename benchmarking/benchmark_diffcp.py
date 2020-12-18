from py_utils.loaders import load_cone_program
import diffcp
from time import perf_counter
from tqdm import trange
import numpy as np
from scipy.sparse import csc_matrix


def solve_and_time(folder, cone_dict, num_programs):
    print(f"Solving programs in {folder}")

    solve_times = np.zeros(num_programs)
    deriv_times = np.zeros(num_programs)
    adjoint_times = np.zeros(num_programs)
    for program_num in trange(num_programs):
        program = load_cone_program(f"{folder}/{program_num}.txt")
        A, b, c = program["A"], program["b"], program["c"]
        A = csc_matrix(A)

        start = perf_counter()
        x, y, s, D, DT = diffcp.solve_and_derivative(A, b, c, cone_dict, eps=1e-5)
        solve_times[program_num] = perf_counter() - start

        start = perf_counter()
        D(np.zeros(A.shape), np.zeros(b.shape), np.ones(c.shape))
        deriv_times[program_num] = perf_counter() - start

        start = perf_counter()
        DT(np.ones(x.shape), np.zeros(y.shape), np.ones(s.shape))
        adjoint_times[program_num] = perf_counter() - start

    print(f"Solving took an average of {np.mean(solve_times)} seconds")
    print(f"Derivatives took an average of {np.mean(solve_times)} seconds")
    print(f"Adjoints took an average of {np.mean(solve_times)} seconds")
    np.savetxt(f"{folder}_diffcp_solve_times.txt", solve_times)
    np.savetxt(f"{folder}_diffcp_deriv_times.txt", deriv_times)
    np.savetxt(f"{folder}_diffcp_adjoint_times.txt", adjoint_times)


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
num_programs = 30
K = {
    'f': 3,  # ZERO
    'l': 3,  # POS
    'q': [20]  # SOC
}
name = "soc-large"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === SDP SMALL
print("=== SOLVING SMALL SDPs WITH DIFFCP ===")
num_programs = 30
K = {
    'f': 5,  # ZERO
    's': [10],  # SD
}
name = "sdp-small"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === SDP LARGE
print("=== SOLVING LARGE SDPs WITH DIFFCP ===")
num_programs = 30
K = {
    'f': 10,  # ZERO
    's': [20],  # SD
}
name = "sdp-large"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === EXPONENTIAL SMALL
print("=== SOLVING SMALL EXPONENTIAL WITH DIFFCP ===")
num_programs = 30
K = {
    'ep': 2
}
name = "exponential-small"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)


# === EXPONENTIAL LARGE
print("=== SOLVING LARGE EXPONENTIAL WITH DIFFCP ===")
num_programs = 30
K = {
    'ep': 4
}
name = "exponential-large"
solve_and_time(f"benchmarking/programs/{name}", K, num_programs)
