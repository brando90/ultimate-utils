"""
This file is to an example of comparing activations of CCNs using the original pwcca.
"""
#%%

import torch

from pathlib import Path

from meta_learning.base_models.learner_from_opt_as_few_shot_paper import Learner

from uutils.torch_uu.dataloaders import get_miniimagenet_dataloaders_torchmeta
from uutils.torch_uu import process_meta_batch

from argparse import Namespace

args = Namespace()
# args.k_eval = 150
# args.image_size = 84
args.image_size = 32
args.bn_eps = 1e-3
args.bn_momentum = 0.95
args.n_classes = 5
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum, n_classes=args.n_classes).to(args.device)
model2 = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum, n_classes=args.n_classes).to(args.device)

cxa_dist_type = 'pwcca'
layer_name = "model.features.conv1"