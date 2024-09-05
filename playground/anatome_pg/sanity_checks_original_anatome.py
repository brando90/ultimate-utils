#%%
"""
The similarity of the same network should always be 1.0 on same input.
"""
from copy import deepcopy

import torch
import torch.nn as nn

import uutils.torch_uu
from uutils.torch_uu import cxa_sim, approx_equal
from uutils.torch_uu.models import get_named_identity_one_layer_linear_model

print('--- Sanity check: sCCA = 1.0 when using same net twice with same input. --')

Din: int = 10
Dout: int = Din
B: int = 2000
mdl1: nn.Module = get_named_identity_one_layer_linear_model(D=Din)
mdl2: nn.Module = deepcopy(mdl1)  # use deepcopy otherwise the models will collect the data in the hook twice
layer_name = 'fc0'

# - ends up comparing two matrices of size [B, Dout], on same data, on same model
cxa_dist_type = 'svcca'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
print(f'Is it close to 1.0? {approx_equal(sim, 1.0)}')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

cxa_dist_type = 'pwcca'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
print(f'Is it close to 1.0? {approx_equal(sim, 1.0)}')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

cxa_dist_type = 'lincka'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
print(f'Is it close to 1.0? {approx_equal(sim, 1.0)}')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

cxa_dist_type = 'opd'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
print(f'Is it close to 1.0? {approx_equal(sim, 1.0)}')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

#%%
"""
Reproducing: How many data points: https://github.com/google/svcca/blob/master/tutorials/001_Introduction.ipynb

As n increases, the cca sim should decrease until it converges to the true max linear correlation in the data.
This is because when D is small it's easy to correlate via Xw, Yw since there are less equations (m data) than unknown (D features). 
Similarly, the similarity decreases because the more data there is, the more variation has to be captured and thus the less
correlation there will be.
This is correct because 1/4*E[|| Xw - Yw||^2]^2 is proportional the pearson's correlation (assuming Xw, Yw is standardized).

"""
import torch
import torch.nn as nn

import uutils
from uutils.torch_uu import cxa_sim, approx_equal
from uutils.torch_uu.models import get_named_one_layer_random_linear_model

import uutils.plot as uulot

print('\n--- Sanity check: when number of data points B is smaller than D, then it should be trivial to make similiarty 1.0 '
      '(even if nets/matrices are different)')
B: int = 10
Dout: int = 300
mdl1: nn.Module = get_named_one_layer_random_linear_model(B, Dout)
mdl2: nn.Module = get_named_one_layer_random_linear_model(B, Dout)  # no need for deep copy since models are already different
layer_name = 'fc0'
# cxa_dist_type = 'svcca'
cxa_dist_type = 'pwcca'
# cxa_dist_type = 'lincka'
# cxa_dist_type = 'opd'

# - get sim for B << D e.g. [B=10, D=300] easy to "fit", to many degrees of freedom
X: torch.Tensor = uutils.torch_uu.get_identity_data(B)
# mdl1(X) : [B, Dout] = [B, B] [B, Dout]
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} (since we have many features to match the two Xw1, Yw2).')
print(f'Is it close to 1.0? {approx_equal(sim, 1.0)}')
# assert(approx_equal(sim, 1.0))

print('\n-- Santity: just makes sure that when low data is present sim is high and afterwards (as n->infty) sim (CCA) '
      'converges to the "true" cca value (eventually)')
Dout: int = 64
# data_sizes: list[int] = [10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000]
data_sizes: list[int] = [10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
# data_sizes: list[int] = [10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000]
# data_sizes: list[int] = [10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
sims: list[float] = []
for b in data_sizes:
    # print('---')
    X: torch.Tensor = uutils.torch_uu.get_identity_data(b)
    mdl1: nn.Module = get_named_one_layer_random_linear_model(b, Dout)
    mdl2: nn.Module = get_named_one_layer_random_linear_model(b, Dout)
    # print(f'{b=}')
    sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
    # print(f'{sim=}')
    sims.append(sim)

print(f'{sims=}')
uulot.plot(x=data_sizes, y=sims, xlabel='number of data points (n)', ylabel=f'similarity ({cxa_dist_type})', show=True, save_plot=True, plot_filename=f'ndata_vs_sim_{cxa_dist_type}', title=f'Features (D) vs Sim ({cxa_dist_type})', x_hline=Dout, x_hline_label=f'B=D={Dout}')

#%%

from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import uutils
from uutils.torch_uu import cxa_sim, approx_equal
from uutils.torch_uu.models import get_named_one_layer_random_linear_model

from uutils.plot import plot, save_to_desktop
import uutils.plot as uuplot

B: int = 10  # [101, 200, 500, 1000, 2000, 5000, 10000]
Din: int = B
Dout: int = 300
mdl1: nn.Module = get_named_one_layer_random_linear_model(Din, Dout)
mdl2: nn.Module = get_named_one_layer_random_linear_model(Din, Dout)  # no need for deep copy since models are already different
layer_name = 'fc0'
cxa_dist_type = 'svcca'
# cxa_dist_type = 'pwcca'
# cxa_dist_type = 'lincka'
# cxa_dist_type = 'opd'

X: torch.Tensor = uutils.torch_uu.get_identity_data(B)
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=}')
print(f'Is it close to 1.0? {approx_equal(sim, 1.0)}')
# assert(approx_equal(sim, 1.0))

# data_sizes: list[int] = [10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000]
B: int = 64
D_feature_sizes: list[int] = [10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
if cxa_dist_type == 'opd':
    D_feature_sizes: list[int] = [10, 25, 50, 100, 200, 500, 1_000, 2_000]
sims: list[float] = []
for d in D_feature_sizes:
    print(f'{d=}')
    X: torch.Tensor = uutils.torch_uu.get_identity_data(B)
    mdl1: nn.Module = get_named_one_layer_random_linear_model(B, d)
    mdl2: nn.Module = get_named_one_layer_random_linear_model(B, d)
    sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=None, iters=1, cxa_dist_type=cxa_dist_type)
    # print(f'{d=}, {sim=}')
    sims.append(sim)

print(f'{sims=}')
uuplot.plot(x=D_feature_sizes, y=sims, xlabel='number of features/size of dimension (D)', ylabel=f'similarity ({cxa_dist_type})', show=True, save_plot=True, plot_filename=f'D_vs_sim_{cxa_dist_type}', title=f'Features (D) vs Sim ({cxa_dist_type})', x_hline=B, x_hline_label=f'B=D={B}')

