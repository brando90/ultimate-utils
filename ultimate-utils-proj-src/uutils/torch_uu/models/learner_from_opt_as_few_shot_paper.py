"""
Notes:
- 1. For the conv layer we have H' = H since H' = H+2p-k+1 = H' for p=1, k=3. i.e. same as previous layer
    - since stride=1 as default (so only moves by 1) since you want to see all the image for a conv.
- 2. For the avg pool layer we have H' = H/2 i.e. half of previous layer
    - since stride=kernel_size as default (so you want to avg more significantly for pool layers e.g. for invariance)
- 3. H = W should the true for this model and data, unless you feed rectangular data for some reason.

For this model if H = 84, input layer H^(l)_{layer_type} = is the H at layer l for layer_type we have:
- H^(l)_{conv} = H/2**(l-1)
- H^(l)_{avg_pool} = H/2**(l)
since, for the conv layer the height don't change and the pooling layer halves for each spatial dimension.

"""
from __future__ import division, print_function, absolute_import

import pdb
import copy
from argparse import Namespace
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from typing import Optional

# from automl.core.operations import SPP
from uutils.torch_uu.models.spp import SPP


def helloworld(msg="hello"):
    print(f'hello world with mgs: {msg}')


def get_defaul_args_for_5cnn() -> Namespace:
    args: Namespace = Namespace()
    args.image_size = 84
    args.bn_eps = 1e-3
    args.bn_momentum = 0.95
    args.n_classes = 5
    args.filter_size = 32
    args.levels = None
    args.spp = False
    return args


def get_learner_from_args(args: Namespace) -> nn.Module:
    return Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_classes)


def get_default_learner(image_size: int = 84,
                        bn_eps: float = 1e-3,
                        bn_momentum: float = 0.95,
                        n_classes: int = 5,
                        filter_size: int = 32,
                        levels: Optional = None,
                        spp: bool = False) -> nn.Module:
    """
    Gets a 5CNN as in the paper "optimization as a model for few shot learning". Cbfin et al. in MAML also use
    the same hps:
        "For MiniImagenet, we used 32 filters per layer to reduce overfitting, as done by (Ravi & Larochelle, 2017)."

    Copy pasted from my script that ran experiments:
        args.bn_momentum = 0.95
        args.bn_eps = 1e-3
        args.grad_clip_mode = 'clip_all_together'
        args.image_size = 84
        args.base_model = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum,
                                  n_classes=args.n_classes).to(args.device)

    """
    return Learner(image_size, bn_eps, bn_momentum, n_classes, filter_size, levels, spp)


def get_default_learner_and_hps_dict(image_size: int = 84,
                                     bn_eps: float = 1e-3,
                                     bn_momentum: float = 0.95,
                                     n_classes: int = 5,
                                     filter_size: int = 32,
                                     levels: Optional = None,
                                     spp: bool = False) -> tuple[nn.Module, dict]:
    model_hps_for_cons_dict: dict = {image_size: image_size, bn_eps: bn_eps, bn_momentum: bn_momentum,
                                     n_classes: n_classes, filter_size: filter_size, levels: levels, spp: spp}
    model: nn.Module = Learner(**model_hps_for_cons_dict)
    return model, model_hps_for_cons_dict


def get_default_learner_from_default_args(args: Optional[Namespace] = None) -> nn.Module:
    if args is None:
        args = get_defaul_args_for_5cnn()
    mdl = get_learner_from_args(args)
    return mdl


def get_feature_extractor_pool_layers(L: int = 4) -> list[str]:
    return [f'model.features.pool{i}' for i in range(1, L + 1)]


def get_feature_extractor_conv_layers(L: int = 4, include_cls: bool = False) -> list[str]:
    """
    Note: if the cls is present then we need B >= s*D since the output for it has shape
        [B, n_c] where n_c so we need, B >= 10*5 = 50 for example.
        s being used for B = 13 is
            s_cls = B/n_c = 13/5 = 2.6
            s_cls = B/n_c = 26/5 = 5.2
    """
    layers: list[str] = [f'model.features.conv{i}' for i in range(1, L + 1)]
    if include_cls:
        layers: list[str] = layers + ['model.cls']
    return layers


def get_head_cls() -> list[str]:
    return ['model.cls']


def get_all_layers_minus_cls(L: int = 4) -> list[str]:
    layer_names: str = []
    for l in range(1, L + 1):
        layer_name1: str = f'model.features.conv{l}'
        layer_name2: str = f'model.features.norm{l}'
        layer_name3: str = f'model.features.relu{l}'
        layer_name4: str = f'model.features.pool{l}'
        layer_names.append(layer_name1)
        layer_names.append(layer_name2)
        layer_names.append(layer_name3)
        layer_names.append(layer_name4)
    return layer_names


def get_last_two_layers(layer_type: str = 'conv', include_cls: bool = True,
                        start_L: int = 4, end_L: int = 4
                        ) -> list[str]:
    assert layer_type in ['conv', 'norm', 'relu', 'pool']
    layers: list[str] = [f'model.features.{layer_type}{i}' for i in range(start_L, end_L + 1)]
    if include_cls:
        layers: list[str] = layers + ['model.cls']
    return layers


class Learner(nn.Module):

    def __init__(self, image_size,
                 bn_eps: float,
                 bn_momentum: float,
                 n_classes: int,
                 filter_size: int = 32,  # Meta-LSTM & MAML use 32 filters
                 levels: Optional = None,
                 spp: bool = False
                 ):
        """[summary]

        Args:
            image_size ([type]): [description]
            bn_eps ([type]): [description]
            bn_momentum ([type]): [description]
            n_classes ([type]): [description]
            levels ([type], optional): [description]. Defaults to None.
            spp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.spp = spp
        # - note: "model" is also a Module
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=filter_size, kernel_size=3, padding=1)),
            ('norm1', nn.BatchNorm2d(filter_size, bn_eps, bn_momentum)),
            ('relu1', nn.ReLU(inplace=False)),
            ('pool1', nn.MaxPool2d(kernel_size=2)),

            ('conv2', nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=3, padding=1)),
            ('norm2', nn.BatchNorm2d(filter_size, bn_eps, bn_momentum)),
            ('relu2', nn.ReLU(inplace=False)),
            ('pool2', nn.MaxPool2d(kernel_size=2)),

            ('conv3', nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=3, padding=1)),
            ('norm3', nn.BatchNorm2d(filter_size, bn_eps, bn_momentum)),
            ('relu3', nn.ReLU(inplace=False)),
            ('pool3', nn.MaxPool2d(kernel_size=2)),

            ('conv4', nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=3, padding=1)),
            ('norm4', nn.BatchNorm2d(filter_size, bn_eps, bn_momentum)),
            ('relu4', nn.ReLU(inplace=False)),
            ('pool4', nn.MaxPool2d(kernel_size=2))]))
        })

        if spp:
            spp_ = SPP(filter_size, levels)
            self.model.update({'spp': spp_})
            self.model.update({'cls': nn.Linear(spp_.output_size, n_classes)})
        else:
            clr_in = image_size // 2 ** 4
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


def load_model_5CNN_opt_as_model_for_few_shot(model_hps_for_cons_dict: dict) -> nn.Module:
    # - get the hps of the model & build the instance
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import Learner
    model: nn.Module = Learner(**model_hps_for_cons_dict)
    return model
