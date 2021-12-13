# __all__ = ["Operation"]

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SPP(nn.Module):
    def __init__(self, channels, levels):
        super(SPP, self).__init__()
        self.channels = channels
        self.levels = levels

    def forward(self, x):
        batch_size, height, width = x.size(0), x.size(2), x.size(2)
        out_list = []
        for level in self.levels:
            h_kernel = int(math.ceil(height / level))
            w_kernel = int(math.ceil(width / level))
            h_pad0 = int(math.floor((h_kernel * level - height) / 2))
            h_pad1 = int(math.ceil((h_kernel * level - height) / 2))
            w_pad0 = int(math.floor((w_kernel * level - width) / 2))
            w_pad1 = int(math.ceil((w_kernel * level - width) / 2))
            pool = nn.MaxPool2d(
                kernel_size=(h_kernel, w_kernel),
                stride=(h_kernel, w_kernel),
                padding=(0, 0)
            )
            padded_x = F.pad(x, pad=[w_pad0, w_pad1, h_pad0, h_pad1])
            out = pool(padded_x)
            print(out.size())
            out_list.append(out.view(batch_size, -1))

        return torch.cat(out_list, dim=1)

    @property
    def output_size(self):
        output_size = 0
        for level in self.levels:
            output_size += self.channels * level * level

        return output_size