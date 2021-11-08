#%%
"""
https://stackoverflow.com/questions/55368485/draw-error-shading-bands-on-line-plot-python?noredirect=1&lq=1
"""

# import numpy as np                 # v 1.19.2
# import matplotlib.pyplot as plt    # v 3.3.2
#
# rng = np.random.default_rng(seed=1)
#
# x = np.linspace(0, 5*np.pi, 50)
# y = np.sin(x)
# # error = np.random.normal(0.1, 0.02, size=x.shape) # I leave this out
# nb_yfuncs = 25
# ynoise = rng.normal(1, 0.1, size=(nb_yfuncs, y.size))
# yfuncs = nb_yfuncs*[y] + ynoise
#
# # fig, ax = plt.subplots(figsize=(10,4))
# # for yfunc in yfuncs:
# #     plt.plot(x, yfunc, 'k-')
# #
# # plt.show()
#
# ymean = yfuncs.mean(axis=0)
# ymin = yfuncs.min(axis=0)
# ymax = yfuncs.max(axis=0)
# yerror = np.stack((ymean-ymin, ymax-ymean))
#
# fig, ax = plt.subplots(figsize=(10, 4))
# ax.grid()
# plt.fill_between(x, ymin, ymax, alpha=0.2, label='error band')
# plt.errorbar(x, ymean, yerror, color='tab:blue', ecolor='tab:blue',
#              capsize=3, linewidth=1, label='mean with error bars')
# plt.legend()
#
# plt.show()

#%%

import numpy as np                 # v 1.19.2
import matplotlib.pyplot as plt    # v 3.3.2

# the number of x values to consider in a given range e.g. [0,1] will sample 10 raw features x sampled at in [0,1] interval
num_x: int = 10
# the repetitions for each x feature value e.g. multiple measurements for sample x=0.0 up to x=1.0 at the end
rep_per_x: int = 5
total_size_data_set: int = num_x * rep_per_x
print(f'{total_size_data_set=}')
# - create fake data set
# only consider 10 features from 0 to 1
x = np.linspace(start=0.0, stop=1.0, num=num_x)

# to introduce fake variation add uniform noise to each feature and pretend each one is a new observation for that feature
noise_uniform: np.ndarray = np.random.rand(rep_per_x, num_x)
# same as above but have the noise be the same for each x (thats what the 1 means)
noise_normal: np.ndarray = np.random.randn(rep_per_x, 1)
# signal function
sin_signal: np.ndarray = np.sin(x)
cos_signal: np.ndarray = np.cos(x)
# [rep_per_x, num_x]
y1: np.ndarray = sin_signal + noise_uniform + noise_normal
y2: np.ndarray = cos_signal + noise_uniform + noise_normal

# - since seaborn expects a an x value paired with it's y value, let's flatten the y's and make sure the corresponding
# x value is alined with it's y value.
# x: np.ndarray = np.tile(x, rep_per_x)  # np.tile = Construct an array by repeating A the number of times given by reps.
# y: np.ndarray = np.ravel(y1)  # flatten the y's to match the x values to have the x to it's corresponding y
# y2: np.ndarray = np.ravel(y2)  # flatten the y's to match the x values to have the x to it's corresponding y

ymean = y1.mean(axis=0)
# ymin = yfuncs.min(axis=0)
# ymax = yfuncs.max(axis=0)
# yerror = np.stack((ymean-ymin, ymax-ymean))
yerror = y1.std(axis=0)

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, tight_layout=True)
axs.grid()
plt.errorbar(x=x, y=ymean, yerr=yerror, color='tab:blue', ecolor='tab:blue',
             capsize=3, linewidth=1, label='mean with error bars')
plt.fill_between(x, ymean-yerror, ymean+yerror, alpha=0.2, label='error band')
plt.legend()

# fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, tight_layout=True)
# axs.grid()
# plt.errorbar(x=x, y=ymean, yerr=yerror, color='tab:orange', ecolor='tab:orange',
#              capsize=3, linewidth=1, label='mean with error bars')
# plt.fill_between(x, ymean-yerror, ymean+yerror, alpha=0.2, label='error band')
# plt.legend()

plt.show()