#%%
"""
An example of how to plot my [B, L] for each metric.


https://seaborn.pydata.org/tutorial/relational.html#relational-tutorial

https://seaborn.pydata.org/examples/errorband_lineplots.html
https://www.youtube.com/watch?v=G3F0EZcW9Ew
https://github.com/knathanieltucker/seaborn-weird-parts/commit/3e571fd8e211ea04b6c9577fd548e7e532507acf
https://github.com/knathanieltucker/seaborn-weird-parts/blob/3e571fd8e211ea04b6c9577fd548e7e532507acf/tsplot.ipynb

https://seaborn.pydata.org/examples/errorband_lineplots.html
"""
from collections import OrderedDict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd

print(sns)

np.random.seed(22)
sns.set(color_codes=True)

# the number of x values to consider in a given range e.g. [0,1] will sample 10 raw features x sampled at in [0,1] interval
num_x: int = 4
# the repetitions for each x feature value e.g. multiple measurements for sample x=0.0 up to x=1.0 at the end
rep_per_x: int = 5  # B
total_size_data_set: int = num_x * rep_per_x
print(f'{total_size_data_set=}')
# - create fake data set
# only consider 10 features from 0 to 1
# x = np.linspace(start=0.0, stop=1.0, num=num_x)
x = np.arange(start=1, stop=5, step=num_x)
# to introduce fake variation add uniform noise to each feature and pretend each one is a new observation for that feature
noise_uniform: np.ndarray = np.random.rand(rep_per_x, num_x)
# same as above but have the noise be the same for each x (thats what the 1 means)
noise_normal: np.ndarray = np.random.randn(rep_per_x, 1)
# signal function
sin_signal: np.ndarray = np.sin(x)
cos_signal: np.ndarray = np.cos(x)
# [rep_per_x, num_x]
data1: np.ndarray = sin_signal + noise_uniform + noise_normal
data2: np.ndarray = cos_signal + noise_uniform + noise_normal

# data_od: OrderedDict = OrderedDict()
# for idx_x in range(num_x):
#     # [rep_per_x, 1]
#     samples_for_x: np.ndarray = data[:, idx_x]
#     data_od[str(x[idx_x])] = samples_for_x
#

column_names = ["layer_name", "metric", "sample_val"]
data_df = pd.DataFrame(columns=column_names)

data = data1
metric = 'sin'
for row in range(data.shape[0]):  # b
    for col in range(data.shape[1]):  # l
        df_row = {'layer_name': f'Layer{col}', 'metric': metric, 'sample_val': data[row, col]}
        data_df = data_df.append(df_row, ignore_index=True)

data = data2
metric = 'cos'
for row in range(data.shape[0]):  # b
    for col in range(data.shape[1]):  # l
        df_row = {'layer_name': f'Layer{col}', 'metric': metric, 'sample_val': data[row, col]}
        data_df = data_df.append(df_row, ignore_index=True)

print(data_df)
sns.lineplot(x='layer_name', y='sample_val', hue='metric', data=data_df, err_style='band')
# ax = sns.lineplot(x=x, y=data)
# ax = sns.lineplot(data=data, err_style='band')
# ax = sns.lineplot(data=data, err_style='bars')
# ax = sns.lineplot(data=data, ci='sd', err_style='band')
# ax = sns.lineplot(data=data, ci='sd', err_style='bars')

# ax = sns.relplot(data=data)

plt.show()

#%%
"""
https://seaborn.pydata.org/examples/errorband_lineplots.html
"""

# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# from pandas import DataFrame
#
# fmri: DataFrame = sns.load_dataset("fmri")
# print(fmri)
# sns.lineplot(x="timepoint", y="signal",  hue="region", style="event", data=fmri)
# plt.show()
