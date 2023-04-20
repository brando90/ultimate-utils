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
Use a title Accuracy vs. Number of Parameters.
Use labels for the axes, y-axis is Accuracy (acc = exp( -CE(N) )),
x-axis log N (log Number of Parameters).
Use a legends.
Make sure to add comments explaining the code.
Use python matplotlib and give me all the code at once.
"""

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Define the function CE(N)
# def CE(N, alpha=-1, c=1):
#     return (N ** alpha) / (c ** alpha)
#
#
# # Define the function acc(CE, L)
# def acc(CE, L):
#     return np.exp(-CE * L)
#
#
# # Set up the plot
# plt.figure()
#
# # Generate N values logarithmically from 1 to 1000
# N_values = np.logspace(0, 3, 1000)
#
# # Loop through L values from 1 to 5
# for L in range(1, 6):
#     # Calculate the acc values for each N and L
#     acc_values = acc(CE(N_values), L)
#
#     # Plot the results
#     plt.plot(N_values, acc_values, label=f"L = {L}")
#
# # Set the x and y scales
# plt.xscale("log")
# plt.yscale("linear")
#
# # Set the y-axis limits
# plt.ylim(0, 1)
#
# # Add a grid
# plt.grid()
#
# # Add a title
# plt.title("Accuracy vs Number of Parameters for Different Sequence Lengths")
#
# # Add labels for the axes
# plt.xlabel("Number of Parameters (log scale)")
# plt.ylabel("Accuracy (linear scale)")
#
# # Add a legend
# plt.legend()
#
# # Show the plot
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the functions
# def CE(N, alpha, c):
#     return N**alpha / c**alpha
#
# def acc(CE, L):
#     return np.exp(-CE * L)
#
# # Set up the plot
# fig, ax = plt.subplots()
#
# # Define the parameters
# alpha = -1
# c = 1
# N_values = np.logspace(0, 3, 100)  # Logarithmically spaced N values from 1 to 1000
#
# # Loop over sequence lengths L and plot the curves
# for L in range(1, 6):
#     y_values = acc(CE(N_values, alpha, c), L)
#     ax.plot(N_values, y_values, label=f'L={L}')
#
# # Set the title and axis labels
# ax.set_title('Accuracy vs. Number of Parameters')
# ax.set_xlabel('N (Number of Parameters)')
# ax.set_ylabel('Accuracy (acc)')
#
# # Set the axis scales and bounds
# ax.set_xscale('log')
# ax.set_ylim(0, 1)
#
# # Add grid and legend
# ax.grid()
# ax.legend()
#
# # Add arrow and text
# arrow_start = (N_values[0], acc(CE(N_values[0], alpha, c), 1))
# arrow_end = (N_values[-1], acc(CE(N_values[-1], alpha, c), 5))
# ax.annotate('',
#             xy=arrow_end, xycoords='data',
#             xytext=arrow_start, textcoords='data',
#             arrowprops=dict(arrowstyle="->",
#                             lw=2, color='red'))
# ax.text(15, 0.5, 'increasing seq length', rotation=15, color='red', fontsize=12)
#
# # Show the plot
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the CE function
# def CE(N, alpha, c):
#     return (N**alpha) / (c**alpha)
#
# # Define the acc function
# def acc(CE, L):
#     return np.exp(-CE * L)
#
# # Set parameters
# alpha = -1
# c = 1
# N_values = np.logspace(0, 3, 100)  # N goes from 1 to 1000 logarithmically
# L_values = np.arange(1, 6)  # L goes from 1 to 5 linearly in increments of 1
#
# # Set up the plot
# plt.figure()
# plt.xscale("log")  # Set x-axis scale to log
# plt.yscale("linear")  # Set y-axis scale to linear
# plt.ylim(0, 1)  # Bound the y-axis between 0 and 1
# plt.xlabel("log N (num. of parameters)")  # x-axis label
# plt.ylabel("acc = exp( -CE(N) )")  # y-axis label
# plt.title("Log-Linear Plot of acc vs log N")  # Title of the plot
# plt.grid()  # Add grid lines
#
# # Plot the curves for different L values
# for L in L_values:
#     CE_values = CE(N_values, alpha, c)
#     acc_values = acc(CE_values, L)
#     plt.plot(N_values, acc_values, label=f"L = {L}")
#
# # # Add the horizontal arrow and text
# # arrow_start = (N_values[0], acc(CE(N_values[0], alpha, c), L_values[0]))
# # arrow_end = (N_values[-1], acc(CE(N_values[-1], alpha, c), L_values[-1]))
# # plt.annotate("increasing seq length",
# #              xy=arrow_start,
# #              xytext=arrow_end,
# #              textcoords='data',
# #              arrowprops=dict(facecolor='black', arrowstyle='<->'))
#
# # Add the legend
# plt.legend()
#
# # Show the plot
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha = -1
c = 1
L_values = np.arange(1, 6, 1)  # L goes from 1 to 5 in increments of 1
N_values = np.logspace(0, 3, 1000)  # N goes from 1 to 1000 logarithmically

# Function to calculate CE(N)
def CE(N, alpha, c):
    return (N ** alpha) / (c ** alpha)

# Function to calculate accuracy (acc)
def accuracy(N, L, alpha, c):
    return np.exp(-CE(N, alpha, c) * L)

# Creating the plot
plt.figure()

for L in L_values:
    acc = accuracy(N_values, L, alpha, c)
    plt.plot(N_values, acc, label=f'L={L}')

plt.xscale('log')
plt.yscale('linear')
plt.title('Accuracy vs. Number of Parameters')
plt.xlabel('log N (log Number of Parameters)')
plt.ylabel('Accuracy (acc = exp( -CE(N) ))')
plt.legend()
plt.grid()
plt.ylim(0, 1)

# Display the plot
plt.show()




