import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
os.makedirs("benchmarking/figures/", exist_ok=True)
sns.set()

names = [
    "soc-small",
    "soc-large",
    "sdp-small",
    "sdp-large",
    "exponential-small",
    "exponential-large"
]

names2labels = {
    "soc-small": "SOC-small",
    "soc-large": "SOC-large",
    "sdp-small": "SDP-small",
    "sdp-large": "SDP-large",
    "exponential-small": "exp-small",
    "exponential-large": "exp-large"
}

SOLVE_PLOT = True
DERIV_PLOT = True
ADJOINT_PLOT = True

if SOLVE_PLOT:
    plt.clf()
    diffcp_times = {name: np.loadtxt(f"benchmarking/programs/{name}_diffcp_solve_times.txt") for name in names}
    cpd_times = {name: np.loadtxt(f"benchmarking/programs/{name}_cpd_solve_times.txt") for name in names}
    df = pd.DataFrame()
    for name, times in diffcp_times.items():
        d = pd.DataFrame(times, columns=["time"])
        d["setting"] = names2labels[name]
        d["Library"] = "diffcp"
        df = df.append(d)
    for name, times in cpd_times.items():
        d = pd.DataFrame(times, columns=["time"])
        d["setting"] = names2labels[name]
        d["Library"] = "ConeProgramDiff"
        df = df.append(d)

    plt.yscale('log')
    ax = sns.barplot(x="setting", y="time", hue="Library", data=df)
    plt.ylabel("Time (s)")
    plt.xlabel("")
    plt.title("Time to solve program")
    plt.tight_layout()
    plt.savefig("benchmarking/figures/solve_times.png")


if DERIV_PLOT:
    plt.clf()
    diffcp_times = {name: np.loadtxt(f"benchmarking/programs/{name}_diffcp_deriv_times.txt") for name in names}
    cpd_times = {name: np.loadtxt(f"benchmarking/programs/{name}_cpd_deriv_times.txt") for name in names}
    df = pd.DataFrame()
    for name, times in diffcp_times.items():
        d = pd.DataFrame(times, columns=["time"])
        d["setting"] = names2labels[name]
        d["Library"] = "diffcp"
        df = df.append(d)
    for name, times in cpd_times.items():
        d = pd.DataFrame(times, columns=["time"])
        d["setting"] = names2labels[name]
        d["Library"] = "ConeProgramDiff"
        df = df.append(d)

    plt.yscale('log')
    ax = sns.barplot(x="setting", y="time", hue="Library", data=df)
    plt.ylabel("Time (s)")
    plt.xlabel("")
    plt.title("Time to compute derivative")
    plt.tight_layout()
    plt.savefig("benchmarking/figures/deriv_times.png")


if ADJOINT_PLOT:
    plt.clf()
    diffcp_times = {name: np.loadtxt(f"benchmarking/programs/{name}_diffcp_adjoint_times.txt") for name in names}
    cpd_times = {name: np.loadtxt(f"benchmarking/programs/{name}_cpd_adjoint_times.txt") for name in names}
    df = pd.DataFrame()
    for name, times in diffcp_times.items():
        d = pd.DataFrame(times, columns=["time"])
        d["setting"] = names2labels[name]
        d["Library"] = "diffcp"
        df = df.append(d)
    for name, times in cpd_times.items():
        d = pd.DataFrame(times, columns=["time"])
        d["setting"] = names2labels[name]
        d["Library"] = "ConeProgramDiff"
        df = df.append(d)

    plt.yscale('log')
    ax = sns.barplot(x="setting", y="time", hue="Library", data=df)
    plt.ylabel("Time (s)")
    plt.xlabel("")
    plt.title("Time to compute adjoint")
    plt.tight_layout()
    plt.savefig("benchmarking/figures/adjoint_times.png")
