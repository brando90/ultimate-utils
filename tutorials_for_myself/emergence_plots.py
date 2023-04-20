#%%
"""
Plot acc = exp( -CE(N) ) where CE(N) = N^alpha / c^alpha.
So y-xsis is acc = exp( -CE(N) ) [linear] and x-axis is N [log], in a log linear plot.
Where Alpha = -1 < 0, N goes from 0 to 1_000, c = 2 > 0.
Bound the y-axis between 0 and 1.
Use python matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = -1
c = 2
N = np.logspace(0, 3, 1000)  # Logarithmically spaced N values from 1 to 1000

# Calculate CE(N)
CE_N = (N ** alpha) / (c ** alpha)

# Calculate Accuracy
acc = np.exp(-CE_N)

# Create the plot
plt.figure()
plt.semilogx(N, acc, label="acc = exp(-CE(N))")
plt.xlim(1, 1000)
plt.ylim(0, 1)
plt.xlabel("N [log]")
plt.ylabel("acc [linear]")
plt.title("Log-Linear Plot of Accuracy vs N")
plt.legend()
plt.grid(True)
plt.show()
