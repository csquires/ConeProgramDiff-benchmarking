import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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

diffcp_times = {name: np.loadtxt(f"benchmarking/programs/{name}_diffcp_times.txt") for name in names}
df = pd.DataFrame()
for name, times in diffcp_times.items():
    d = pd.DataFrame(times, columns=["time"])
    d["setting"] = names2labels[name]
    df = df.append(d)

plt.yscale('log')
ax = sns.barplot(x="setting", y="time", data=df)
plt.ylabel("Time (s)")
plt.xlabel("")
plt.tight_layout()
plt.show()
