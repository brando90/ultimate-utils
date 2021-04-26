import torch
import torch.nn as nn

import torch.nn.functional as F

#mini class to add a flatten layer to the ordered dictionary
class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

# class MyBatchNorm1D(nn._NormBase):
#     def __init__(self, old_bn):
#         # note this resets running stats, to avoid that copoy paste the code form
#         super().__init__()
#
#
#
#     def forward(self, input):
#
#         out = F.batch_norm()
#         return out

# class MySequential()

### units tests

# def test_bn1d():
#     # bn1 = nn.BatchNorm1d(1)
#     bn1 = nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
#
#     # 1D data
#     x = torch.randn(1)
#     try:
#         bn1(x)  # this batch norm should fail
#         assert False  # raise error if it does not
#     except:
#         pass
#     # since its regression few shot [k_shot,D] is the input size
#     x = torch.randn([5, 1])
#     out = bn1(x)
#     # mu = # com


if __name__ == '__main__':
    # test_bn1d()
    print('Done! \a')
