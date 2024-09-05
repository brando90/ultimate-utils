from __future__ import division, print_function, absolute_import

import pdb
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from types import SimpleNamespace

from uutils.torch_uu.models.spp import SPP


def helloworld(msg="hello"):
    print(f'hello world with mgs: {msg}')

class Kcnn(nn.Module):

    def __init__(self, image_size,
                 bn_eps,
                 bn_momentum,
                 n_classes,
                 filter_size=64,  # 64 is MAML, 32 is meta-lstm
                 nb_feature_layers=4,  # note total layers is + 1 of this (e.g. 4+1 to get the past nets used)
                 act_type='relu',
                 levels=None,
                 spp=False
                 ):
        """
        @param image_size:
        @param bn_eps:
        @param bn_momentum:
        @param n_classes:
        @param filter_size:
        @param nb_feature_layers:
        @param act_type:
        @param levels:
        @param spp:
        """
        super().__init__()
        self.act_type = act_type
        self.spp = spp
        # create the feature part of the network (the k hidden layers)
        self.nb_feature_layers = nb_feature_layers
        if nb_feature_layers < 1:
            raise ValueError(f'Cnn must have at least 1 layer you gave {nb_feature_layers}')
        layers = []
        for feature_layer in range(1, nb_feature_layers+1):
            act = self.get_act(act_type)
            if feature_layer == 1:
                c = (f'conv{feature_layer}', nn.Conv2d(3, filter_size, 3, padding=1))
            else:
                c = (f'conv{feature_layer}', nn.Conv2d(filter_size, filter_size, 3, padding=1))
            n = (f'norm{feature_layer}', nn.BatchNorm2d(filter_size, bn_eps, bn_momentum))
            a = (f'act{feature_layer}', act)
            p = (f'pool{feature_layer}', nn.MaxPool2d(2))
            layers.append(c); layers.append(n); layers.append(a); layers.append(p)
        layers = nn.Sequential(OrderedDict(layers))
        self.model = nn.ModuleDict({'features': layers})

        # insert final classification layer
        if spp:
            spp_ = SPP(filter_size, levels)
            self.model.update({'spp': spp_})
            self.model.update({'cls': nn.Linear(spp_.output_size, n_classes)})
        else:
            clr_in = image_size // 2**(nb_feature_layers)
            self.model.update({'cls': nn.Linear(filter_size * clr_in * clr_in, n_classes)})
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model.features(x)
        if self.spp:
            out = self.model.spp(out)
        else:
            out = torch.reshape(out, [out.size(0), -1])
        outputs = self.model.cls(out)
        return outputs

    def get_act(self, act_type, inplace=False):
        if act_type == 'relu':
            return nn.ReLU(inplace=inplace)
        elif act_type == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f'Activation act = {act_type} not implemented')

    def get_flat_params(self):
        # return torch_uu.cat([p.view(-1) for p in self.model.parameters()], 0)
        pass

    def copy_flat_params(self, cI):
        # idx = 0
        # for p in self.model.parameters():
        #     plen = p.view(-1).size(0)
        #     p.data.copy_(cI[idx: idx+plen].view_as(p))
        #     idx += plen
        pass

    def transfer_params(self, learner_w_grad, cI):
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        # self.load_state_dict(learner_w_grad.state_dict())
        # #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        # idx = 0
        # for m in self.model.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        #         wlen = m._parameters['weight'].view(-1).size(0)
        #         m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
        #         idx += wlen
        #         if m._parameters['bias'] is not None:
        #             blen = m._parameters['bias'].view(-1).size(0)
        #             m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
        #             idx += blen
        pass

    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()


if __name__ == '__main__':
    args = SimpleNamespace()
    args.n_classes = 64
    args.bn_momentum = 0.95
    args.bn_eps = 1e-3
    args.grad_clip_mode = 'clip_all_together'
    args.image_size = 84
    model = Kcnn(args.image_size, args.bn_eps, args.bn_momentum, args.n_classes)
    if torch.cuda.is_available():
        model.cuda()
    x = torch.randn(2, 3, 84, 84)
    out = model(x)
    print(out.size())

    model = Kcnn(args.image_size, args.bn_eps, args.bn_momentum, args.n_classes, filter_size=64, nb_feature_layers=6, act_type='sigmoid')
    if torch.cuda.is_available():
        model.cuda()
    x = torch.randn(2, 3, 84, 84)
    out = model(x)
    print(out.size())

    print('Done!\a')

    torch.nn.ReLU

