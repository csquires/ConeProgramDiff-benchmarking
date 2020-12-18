from py_utils.loaders import load_cone_program
import diffcp
from time import perf_counter
from tqdm import trange
import numpy as np

K = {
    'f': 3,  # ZERO
    'l': 3,  # POS
    'q': [5]  # SOC
}


num_programs = 30
times = np.zeros(num_programs)
for program_num in trange(num_programs):
    program = load_cone_program(f"benchmarking/programs/program{program_num}.txt")
    A, b, c = program["A"], program["b"], program["c"]

    start = perf_counter()
    diffcp.solve_and_derivative(A, b, c, K)
    time_taken = perf_counter() - start

    print(f"Took {time_taken} seconds")
    times[program_num] = time_taken

np.savetxt("diffcp_times.txt", times)

