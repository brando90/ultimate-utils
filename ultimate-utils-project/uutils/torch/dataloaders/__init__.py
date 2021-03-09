from pathlib import Path

from meta_learning.datasets.mini_imagenet import MetaImageNet, ImageNet

import torch
from torch import nn
from torch.utils.data import DataLoader

def process_batch_sl(args, batch):
    batch_x, batch_y = batch
    if torch.cuda.is_available():
        if args.device:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
        else:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    return batch_x, batch_y

def get_rfs_sl_dataloader(args):
    args.num_workers = 4 if args.num_workers is None else args.num_workers
    args.target_type = 'classification'
    args.trainin_with_epochs = False
    args.data_root = Path('~/data/miniImageNet_rfs/miniImageNet/').expanduser()
    args.data_aug = True
    args.criterion = nn.CrossEntropyLoss()
    # -- get SL dataloaders
    # train_trans, test_trans = transforms_options[opt.transform]
    # train_sl_loader = DataLoader(ImageNet(args=args, partition=args.split),
    #                              batch_size=args.batch_size, shuffle=True, drop_last=True,
    #                              num_workers=args.num_workers)
    train_sl_loader = DataLoader(ImageNet(args=args, partition='train'),
                                 batch_size=args.batch_size, shuffle=True, drop_last=True,
                                 num_workers=args.num_workers)
    val_sl_loader = DataLoader(ImageNet(args=args, partition='val'),
                               batch_size=args.batch_size // 2, shuffle=False, drop_last=False,
                               num_workers=args.num_workers // 2)
    # -- get meta-dataloaders
    args.n_aug_support_samples = 5  # default from rfs
    # meta_testloader = DataLoader(MetaImageNet(args=args, partition='test'),
    #                              batch_size=args.test_batch_size, shuffle=False, drop_last=False,
    #                              num_workers=args.num_workers)
    meta_valloader = DataLoader(MetaImageNet(args=args, partition='val'),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    # if opt.use_trainval:
    #     n_cls = 80
    # else:
    #     n_cls = 64
    return train_sl_loader, val_sl_loader, meta_valloader