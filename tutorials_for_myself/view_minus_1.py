#%%
"""
Summary: view(-1, ...) keeps the remaining dimensions as give and infers the -1 location such that it respects the
original view of the tensor. If it's only .view(-1) then it only has 1 dimension given all the previous ones so it ends
up flattening the tensor.

ref: my answer https://stackoverflow.com/a/66500823/1601580
"""
import torch

x = torch.arange(6)
print(x)

x = x.reshape(3, 2)
print(x)

print(x.view(-1))
