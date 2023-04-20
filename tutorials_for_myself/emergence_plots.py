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

# Define the range of N values, from 0 to 1000
N = np.linspace(0, 1000, 1001)

# Define the alpha and c values
alpha = -1
c = 2

# Calculate the CE(N) values using the formula N^alpha / c^alpha
CE_N = (N ** alpha) / (c ** alpha)

# Calculate the acc values using the formula exp(-CE(N))
acc = np.exp(-CE_N)

# Create a new figure
plt.figure()

# Plot the acc values against N values on a log-linear scale
plt.plot(N, acc)

# Set the x-axis scale to logarithmic
plt.xscale("log")

# Set the y-axis scale to linear
plt.yscale("linear")

# Set the y-axis limits between 0 and 1
plt.ylim(0, 1)

# Add x-axis label
plt.xlabel("N")

# Add y-axis label
plt.ylabel("acc = exp(-CE(N))")

# Add a title to the plot
plt.title("Log-linear plot of acc = exp(-CE(N))")

# Display the plot
plt.show()

