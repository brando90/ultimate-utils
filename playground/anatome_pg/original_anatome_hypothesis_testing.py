"""
This file is for testing the properties of anatome (CCA in particular) that I'd expect.
Mainly to test it's working.
Hypothesis:
1) H1: for a single model, if the downsampling size is decreased then the similarity should increase.
    Or the dCCA should decrease (since there is less information to distiguish the models in the data
    since the data is collapsing. Or the variation in the data decreases so the variation in the output
    of the model decreases too - especially since CCA measure correlation of the (linear) prediction function.
    The assumption (which might be wrong) is that the downsampling makes there be less diference between the
    data points, so the predictions between the models should also decrease.
    Note data means the data wrt CCA, so the input to CCA, so the activation vectors (layers of size [M, D]).
    In the extreme case M=1, D=1 CCA = 1.0 exactly.


btw, perhaps this is wrong...sim might increase since interaction of rvs when they are averaged.
It might depend how much neurons (rvs) are correlated. If they are very correlated, which they usually
are in NNs, then it means the sim might increase more than it "should" compared ot no downsampling.
Need to test empirically.
"""
#%%
from typing import Union

import torch
from anatome import SimilarityHook
from torch import nn, Tensor

"""
Testing H1: does collapse of data (in this
"""

def dCCA(mdl1: nn.Module, mdl2: nn.Module, X: Tensor, layer_name: str, downsample_size: Union[None, int], iters: int = 1) -> float:
    hook1 = SimilarityHook(mdl1, layer_name, 'pwcca')
    hook2 = SimilarityHook(mdl2, layer_name, 'pwcca')
    mdl1.eval()
    mdl2.eval()
    for _ in range(iters):  # might make sense to go through multiple is NN is stochastic e.g. BN, dropout layers
        mdl1(X)
        mdl2(X)
    # - size: size of the feature map after downsampling
    dist: float = hook1.distance(hook2, size=downsample_size)
    return dist

def sCCA(mdl1: nn.Module, mdl2: nn.Module, X: Tensor, layer_name: str, downsample_size: Union[None, int], iters: int = 1) -> float:
    return 1.0 - dCCA(mdl1, mdl2, X, layer_name, downsample_size, iters)

def hardcoded_3_layer_model(in_features: int, out_features: int) -> nn.Module:
    """
    Returns a nn sequential model with 3 layers (2 hidden and 1 output layers).
    ReLU activation. Final layer are the raw scores (so final layer is a linear layer).

    """
    from collections import OrderedDict
    hidden_features = in_features
    params = OrderedDict([
        ('fc0', nn.Linear(in_features=in_features, out_features=hidden_features)),
        ('ReLU0', nn.ReLU()),
        ('fc1', nn.Linear(in_features=hidden_features, out_features=hidden_features)),
        ('ReLU2', nn.ReLU()),
        ('fc2', nn.Linear(in_features=hidden_features, out_features=out_features))
    ])
    mdl = nn.Sequential(params)
    return mdl

def hypothesis1_test():
    # from uutils.torch_uu.models import hardcoded_3_layer_model
    B = 1024
    Din = 524
    downsample_size = 1
    Dout = Din
    mdl1 = hardcoded_3_layer_model(Din, Dout)
    # mdl2 = hardcoded_3_layer_model(Din, Dout)
    # - layer name
    # layer_name = 'fc0'
    # layer_name = 'fc1'
    layer_name = 'fc2'
    # - data
    X: torch.Tensor = torch.distributions.Uniform(low=-1, high=1).sample((B, Din))
    # - since data variation decreases so do similarity
    scca_full: float = sCCA(mdl1, mdl1, X, layer_name, downsample_size=None)
    scca_downample: float = sCCA(mdl1, mdl1, X, layer_name, downsample_size=downsample_size)
    print(f'{scca_full=}, {scca_downample=}')
    assert(scca_full < scca_downample), f'Expected similarity to decrease as less variation in the data is seen e.g.' \
                           f'the data is downampled but got {scca_full=}, {scca_downample=}'



hypothesis1_test()
#%%