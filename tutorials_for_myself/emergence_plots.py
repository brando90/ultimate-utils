#%%
"""
In Python, plot
    acc = exp( -CE(N) )^L
where L (Length of Seq) goes from L=1 to 5 linearly in increments of 1 and,
    CE(N) = N^alpha / c^alpha.
So the y-axsis is acc = exp( -CE(N) ) [linear] and x-axis is log N (num. of parameters) [log]
-- in a log linear plot, so use plt.xscale("log") and plt.yscale("linear").
Where alpha = -1 (note alpha < 0),
N goes from 1 to 1_000 (logarithmically) (note N > 0),
c = 1 (note c > 0).
Bound the y-axis between 0 and 1.
Include in the legends the values of L.
Use grids.
Use a title.
Use labels for the axes.
Use a legend.
Make sure to add comments explaining the code.
Use python matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

def CE(N, alpha, c):
    return (N**alpha) / (c**alpha)

def acc(CE, L):
    return np.exp(-CE * L)

# Parameters
alpha = -1
c = 1
N = np.logspace(0, 3, 1000)  # N values from 1 to 1000 logarithmically
L_values = range(1, 6)  # L values from 1 to 5

# Create the plot
plt.figure()
plt.xscale("log")
plt.yscale("linear")

for L in L_values:
    CE_values = CE(N, alpha, c)
    acc_values = acc(CE_values, L)
    plt.plot(N, acc_values, label=f"L={L}")

plt.xlabel("N")
plt.ylabel("acc")
plt.title("Log-Linear Plot of acc vs N")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()


