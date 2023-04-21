#%%
"""
# In Python, plot
#     acc = exp( -CE(N) )^L
# where L (Length of Seq) goes from L=1 to 5 linearly in increments of 1 and,
#     CE(N) = N^alpha / c^alpha.
# So the y-axsis is acc = exp( -CE(N) ) [linear] and x-axis is log N (num. of parameters) [log]
# -- in a log linear plot, so use plt.xscale("log") and plt.yscale("linear").
# Where alpha = -1 (note alpha < 0),
# N goes from 1 to 1_000 (logarithmically) (note N > 0),
# c = 4 (note c > 0).
# Bound the y-axis between 0 and 1.
# Include in the legends the values of L.
# Use grids.
# Use a title Accuracy vs. Number of Parameters.
# Use labels for the axes, y-axis is Accuracy (acc = exp( -CE(N) )),
# x-axis log N (log Number of Parameters).
# Use a legends.
# Add code to save the plot in png, svg, pdf with a good filename at my desktop path from home ~, expand the path in py.
# Make sure to add comments explaining the code.
# Use python matplotlib and give me all the code at once.

In Python, create one figure with two plots.
First plot (right)
    acc = exp( -CE(N) )^L
where L (Length of Seq) goes from L=1 to 5 linearly in increments of 1, starting at 1.
For the second plot (left)
    CE(N) = N^alpha / c^alpha.
Where alpha = -1 (note alpha < 0),
N goes from 1 to 1_000 (logarithmically) its important N starts at 1=10^0,
c = 4 (note c > 0).

The parameters for the first plot (right) are:
the y-axsis is acc = exp( -CE(N) ) [linear] and x-axis is log N (log Number of parameters) [log],
use a log-linear plot, so use plt.xscale("log") and plt.yscale("linear"),
bound the y-axis between 0 and 1,
include in the legends the values of L,
use grids,
use labels for the axes, y-axis is Accuracy (acc),
use a legends,
in a log linear plot, so use plt.xscale("log") and plt.yscale("linear").

The parameters for the second plot (left) are:
The y-axis is log CE(N) = log( N^alpha / c^alpha) (in proper latex for matplotlib),
label y-axis log CE(N) = N^alpha / c^alpha,
label x-axis is log N (Number of parameters),
use grids,
use legends log y = log CE(N),
use a log-log plot  plt.xscale("log") and plt.yscale("log").

Add code to save the plot in png, svg, pdf with a good filename at my desktop path from home ~, expand the path in py.
Make sure to add comments explaining the code.
Use python matplotlib and give me all the code at once.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left plot (ax1)
alpha = -1
N = np.logspace(0, 3, num=1000)
c = 4
CE = (N**alpha) / (c**alpha)

ax1.plot(N, CE, label=r'$\log y = \log \left(\frac{N^{\alpha}}{c^{\alpha}}\right)$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\log N$ (Number of parameters)')
ax1.set_ylabel(r'$\log CE(N) = \frac{N^{\alpha}}{c^{\alpha}}$')
ax1.legend()
ax1.grid()

# Right plot (ax2)
L = np.arange(1, 6)
for l in L:
    acc = np.exp(-CE)**l
    ax2.plot(N, acc, label=f'L = {l}')

ax2.set_xscale('log')
ax2.set_yscale('linear')
ax2.set_ylim(0, 1)
ax2.set_xlabel(r'$\log N$ (Number of parameters)')
ax2.set_ylabel('Accuracy (acc = exp(-CE(N)))')
ax2.legend()
ax2.grid()

# Save the figure in PNG, SVG, and PDF formats on your desktop
desktop_path = os.path.expanduser("~/Desktop")
plt.savefig(os.path.join(desktop_path, "figure.png"))
plt.savefig(os.path.join(desktop_path, "figure.svg"))
plt.savefig(os.path.join(desktop_path, "figure.pdf"))

# Show the figure
plt.show()



