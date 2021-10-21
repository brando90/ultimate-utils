#%%

import torch

D: int = 7
B: int = 4
x: torch.Tensor = torch.randn(B, D)

# this seems to collect the tensors in the data dimension! To make one single big data matrix
out_cat: torch.Tensor = torch.cat([x, x, x])
print(f'{out_cat.size()=}')
assert(out_cat.size() == torch.Size([3*B, D]))

# collects but remembers the index for the identity of each x
out_stack: torch.Tensor = torch.stack([x, x, x])
print(f'{out_stack.size()=}')
assert(out_stack.size() == torch.Size([3, B, D]))
