#%%
"""
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

Add code to save the plot in png, svg, pdf with filename scaling_law_vs_emergence_plot at my desktop path from home ~, expand the path in py.
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

ax1.plot(N, CE, label=r'$ CE(N) = \log \left(\frac{N^{\alpha}}{c^{\alpha}}\right)$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$N$ (Number of parameters)')
ax1.set_ylabel(r'$CE(N) = \frac{N^{\alpha}}{c^{\alpha}}$')
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
ax2.set_xlabel(r'$N$ (Number of parameters)')
ax2.set_ylabel('Accuracy $acc = exp(-CE(N))^L$')
ax2.legend()
ax2.grid()

# Save the figure in PNG, SVG, and PDF formats on your desktop
desktop_path = os.path.expanduser("~/Desktop")
plt.savefig(os.path.join(desktop_path, "scaling_law_vs_emergence_plot.png"))
plt.savefig(os.path.join(desktop_path, "scaling_law_vs_emergence_plot.svg"))
# plt.savefig(os.path.join(desktop_path, "scaling_law_vs_emergence_plot.pdf"))

# Show the figure
plt.show()

#%%
"""
In Python using matplotlib, create one figure with two plots.
First plot (right)
    acc = exp( -CE(N) )^L
where L (Length of Seq) goes from L=1 to 5 linearly in increments of 1, starting at 1.
For the second plot (left)
    CE(N) = m * N + c
Where m is negative,
N goes from 1 to 1_000 (logarithmically) its important N starts at 1=10^0,
c is such CE(N) (the y value) starts at 8.0. 
CE(N) crosses the x-axis at 1_000, so CE(1_000) = 0.0. Stop the plot there. 

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
The y-axis is CE(N) = m * N + c (in proper latex for matplotlib),
label y-axis CE(N) = m * N + c,
label x-axis is N (Number of parameters),
use grids,
use legends log y = CE(N).

Add code to save the plot in png, svg, with filename linear_decreasing_loss_vs_emergence_plot at my desktop path ~Desktop/ 
expand the user path in python.
Use numpy instead of math library.
Always use proper latex in matplotlib when appropriate.
Use f-strings when appropriate.
Make sure to add comments explaining the code.
Use python matplotlib and give me all the code at once.
"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Function to calculate CE(N)
def CE(N, m, c):
    return m * N + c

# Function to calculate accuracy
def acc(CE_N, L):
    return np.exp(-CE_N) ** L

# Find the slope 'm' and the intercept 'c'
N_start = 1
N_end = 1000
CE_start = 3.0
CE_end = 0.0

m = (CE_end - CE_start) / (N_end - N_start)
c = CE_start - m * N_start

# Create the N values logarithmically
N_values = np.logspace(0, 3, 1000)

# Calculate the corresponding CE(N) values
CE_values = CE(N_values, m, c)

# Prepare the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot (left)
ax1.plot(N_values, CE_values, label=r'${CE(N)}=m \cdot N + c$')
ax1.set_xlabel('N (Number of parameters)')
ax1.set_ylabel(r'${CE(N)}=m \cdot N + c$')
ax1.grid()
ax1.legend()

# Second plot (right)
L_values = np.arange(1, 6)
for L in L_values:
    accuracy_values = acc(CE_values, L)
    ax2.plot(N_values, accuracy_values, label=f'L={L}')

ax2.set_xscale('log')
ax2.set_yscale('linear')
ax2.set_ylim(0, 1)
ax2.set_xlabel('N (Number of parameters)')
ax2.set_ylabel('Accuracy $acc = exp(-CE(N))^L$')
ax2.grid()
ax2.legend()

# Save the figure to your desktop
desktop_path = str(Path.home() / 'Desktop')
fig.savefig(f"{desktop_path}/linear_decreasing_loss_vs_emergence_plot.png", dpi=300)
fig.savefig(f"{desktop_path}/linear_decreasing_loss_vs_emergence_plot.svg", dpi=300)
# fig.savefig(f"{desktop_path}/scaling_law_vs_emergence_plot.pdf", dpi=300)

# Show the figure
plt.show()

#%%
"""
In Python using matplotlib, create one figure with two plots.
First plot (right)
    acc = exp( -CE(N) )^L
where L (Length of Seq) goes from L=1 to 5 linearly in increments of 1, starting at 1.
For the second plot (left)
    CE(N) = a * N^2 + b * x + c
Where the quadratic is concave up and 
a, b , c are such CE(N) (the y value) starts at 3.0
in particular the value of CE(N) (y value) is decreasing and starts at 3.0 so CE(1) = 3.0,
N goes from 1 to 1_000 (logarithmically) its important N starts at 1=10^0,
CE(N) crosses the x-axis at 1_000, so CE(1_000) = 0.0. Stop the plot there. 
CE(N) is never negative so CE(N) >= 0.0, so CE(N) (the y-axis) are always positive.
Make sure CE(N) is a quadratic concave up.
To find the values of a, b and c use a quadratic solver in python that satisfies the constraints:
CE(1) = 3.0, CE(500) = 0.68, CE(1_000) = 0.0, CE(N) >= 0.0.

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
The y-axis is CE(N) = a * N^2 + b * x + c (in proper latex for matplotlib),
label y-axis CE(N) = a * N^2 + b * x + c,
label x-axis is N (Number of parameters),
use grids,
use legends y = CE(N).

Add code to save the plot in png, svg, pdf with filename quadratic_decreasing_loss_vs_emergence_plot at path ~/Desktop expand user in python.
Always show me the plot too in addition to saving them. 
Use numpy instead of math library.
Make sure to add comments explaining the code.
Use python f-strings when appropriate.
Always expand the user in paths that have home ~. 
Use python matplotlib and give me all the code at once.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define CE function with a, b, c as parameters
def ce(N, a, b, c):
    return a * N**2 + b * N + c

# Objective function for minimizing the quadratic difference
def objective(params, *data):
    a, b, c = params
    N, CE = data
    return np.sum((ce(N, a, b, c) - CE)**2)

# Define the constraints for the solver
def constraint1(params):
    a, b, c = params
    return ce(1, a, b, c) - 3.0

def constraint2(params):
    a, b, c = params
    return ce(500, a, b, c) - 0.68

def constraint3(params):
    a, b, c = params
    return ce(1000, a, b, c) - 0.0

# Initial guess for a, b, c
params0 = [0, 0, 0]
N = np.array([1, 500, 1000])
CE = np.array([3.0, 0.5, 0.0])

# Apply constraints and solve for a, b, c
con1 = {'type': 'eq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': constraint3}
cons = [con1, con2, con3]

solution = minimize(objective, params0, args=(N, CE), constraints=cons)
a, b, c = solution.x

# Calculate CE(N) for N between 1 and 1000
N_values = np.logspace(0, 3, num=1000)
CE_values = ce(N_values, a, b, c)

# Calculate accuracy values for L between 1 and 5
L_values = np.arange(1, 6)
acc_values = np.exp(-np.outer(CE_values, L_values))

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot CE(N) on the left subplot
ax1.plot(N_values, CE_values)
ax1.set_xlabel('N (Number of parameters)')
ax1.set_ylabel(r'$CE(N) = a \cdot N^2 + b \cdot N + c$')
# ax1.set_title('CE(N) vs N')
ax1.grid()
ax1.legend([r'$CE(N) = a \cdot N^2 + b \cdot N + c$'])

# Plot accuracy on the right subplot
for i, L in enumerate(L_values):
    ax2.plot(N_values, acc_values[:, i], label=f'L = {L}')

ax2.set_xscale('log')
ax2.set_xlabel('N (Number of parameters)')
ax2.set_ylabel('Accuracy $acc = exp(-CE(N))^L$')
ax2.set_ylim(0, 1)
# ax2.set_title('Accuracy vs log N')
ax2.grid()
ax2.legend()

# Save the plot to the desktop
desktop_path = '~/Desktop'
plt.savefig(os.path.expanduser(f"{desktop_path}/quadratic_decreasing_loss_vs_emergence_plot.png"))
plt.savefig(os.path.expanduser(f"{desktop_path}/quadratic_decreasing_loss_vs_emergence_plot.svg"))
# plt.savefig(os.path.expanduser(f"{desktop_path}/quadratic_decreasing_loss_vs_emergence_plot.pdf"))
plt.show()

# """
# lol, gpt-3.5 doesn't work at all.
# """
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the quadratic function CE(N) = a * N^2 + b * N + c
# # such that CE(1) = 3.0, CE(500) = 0.68, and CE(1000) = 0.0
# # and CE(N) >= 0.0 for all N.
# a, b, c = np.polyfit([1, 500, 1000], [3.0, 0.68, 0.0], 2)
# CE = lambda N: np.maximum(0.0, a * N**2 + b * N + c)
#
# # Define the accuracy function acc(L, N) = exp(-CE(N))^L
# # where L varies from 1 to 5 in increments of 1, and N
# # varies logarithmically from 1 to 1000.
# acc = lambda L, N: np.exp(-CE(N))**L
#
# # Define the range of N values to plot.
# N_vals = np.logspace(0, 3, num=1000, base=10)
#
# # Define the range of L values to plot.
# L_vals = np.arange(1, 6)
#
# # Create the figure and subplots.
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# # Plot the first graph (right).
# for L in L_vals:
#     ax1.plot(N_vals, acc(L, N_vals), label=f'L={L}')
# ax1.set_xlabel('Number of parameters (N)')
# ax1.set_ylabel('Accuracy (acc)')
# ax1.set_xscale('log')
# ax1.set_yscale('linear')
# ax1.set_ylim(0, 1)
# ax1.grid(True)
# ax1.legend()
#
# # Plot the second graph (left).
# ax2.plot(N_vals, np.log(CE(N_vals)), label='log CE(N)')
# ax2.set_xlabel('Number of parameters (N)')
# ax2.set_ylabel(r'log CE(N) = $a N^2 + b N + c$')
# ax2.set_xscale('log')
# ax2.grid(True)
# ax2.legend()
#
# # Save the figure in PNG, SVG, and PDF formats.
# filename = '~/Desktop/plot.png'
# filename = os.path.expanduser(filename)
# plt.savefig(filename)
#
# # Show the plot.
# plt.show()

