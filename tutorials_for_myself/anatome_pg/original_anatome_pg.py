"""
attempt at a colab: https://colab.research.google.com/drive/1GrhWrWFPmlc6kmxc0TJY0Nb6qOBBgjzX#scrollTo=KhUWNu3J_6i4
"""
#%%

# import torch
# import torchvision
# from torch.nn import functional as F
# from torchvision.models import resnet18
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
#
# import matplotlib.pyplot as plt
#
# batch_size = 128
#
# model = resnet18(pretrained=True)
# imagenet = ImageFolder('~/.torch/data/imagenet/val',
#                        transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
#                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
# data = next(iter(DataLoader(imagenet, batch_size=batch_size, num_workers=8)))

#%%

import torch
import torchvision
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


batch_size = 128

# imagenet = ImageFolder('~/.torch/data/imagenet/val',
#                        transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
#                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
# data = next(iter(DataLoader(imagenet, batch_size=batch_size, num_workers=8)))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128
num_workers = 0

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data = next(iter(trainloader))

#%%

import anatome
print(anatome)
# from anatome import CCAHook
from anatome import SimilarityHook

model = resnet18(pretrained=True)
random_model = resnet18()
# random_model = resnet18().cuda()

# hook1 = CCAHook(model, "layer1.0.conv1")
# hook2 = CCAHook(random_model, "layer1.0.conv1")

cxa_dist_type = 'pwcca'
layer_name = "layer1.0.conv1"

hook1 = SimilarityHook(model, layer_name, cxa_dist_type)
hook2 = SimilarityHook(random_model, layer_name, cxa_dist_type)

with torch.no_grad():
    batch_x = data[0]
    print(f'{batch_x.size()=}')
    model(batch_x)
    random_model(batch_x)

# - print distances
print('\n- print distances')
print(f'distance_btw_nets={hook1.distance(hook2, size=8)=}')
print(f'distance_btw_nets={hook1.distance(hook2, size=None)=}')

#%%

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
# layer_name = "model.features.norm1"
# layer_name = "model.features.relu1"
# layer_name = "model.features.pool1"

hook1 = SimilarityHook(model1, layer_name, cxa_dist_type)
hook2 = SimilarityHook(model2, layer_name, cxa_dist_type)

# - get data example for 5CNN
# args.data_path = Path('~/data/').expanduser()  # for some datasets this is enough
# args.n_classes = 5
# args.k_shots = 5
# args.k_eval = 15
# args.meta_batch_size_train = 2
# args.meta_batch_size_eval = 2
# args.num_workers = 0
# # args = get_minimum_args_for_mini_imagenet_from_torchmeta(args)
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_miniimagenet_dataloaders_torchmeta(args)
# meta_batch: dict = next(iter(meta_val_dataloader))
# spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, meta_batch)

# - run anatome analysis
with torch.no_grad():
    # batch_x = qry_x[0]  # first task
    batch_x = data[0]
    print(f'{batch_x.size()=}')
    model1(batch_x)
    model2(batch_x)

# - print distances
print('\n- print distances')
print(f'distance_btw_nets={hook1.distance(hook2, size=8)=}')
print(f'distance_btw_nets={hook1.distance(hook2, size=None)=}')
